"""Depth benchmark -- the reusable harness that decides whether a depth model beats the floor.

Master Spec §7 flags the depth slot "benchmark before lock". This runs ANY depth model behind
the `perception.depth` interface and reports the numbers that actually matter for placing a hand:

  * **wrist z-jitter (mm/frame)** -- the gate. Place the wrist at the depth each model gives
    (Part C's solve_root_depth + hand-cloud fit + temporal smooth) and measure how much the
    placed wrist z moves frame-to-frame. THIS is what becomes the training action, so this is
    what must be quiet. Fair baseline (Part C): UniDepth wrist-pixel 13.4 mm, hand-cloud-fit +
    smooth 11.3 mm. The gate is a 3x reduction from the FAIR baseline -> **< 4 mm/frame**, not
    < 7 mm (that would be grading against the bbox-pseudo-depth strawman).
  * **static-point consistency (%)** -- the depth-map-level temporal noise (consistency.py).
  * **metric scale (m)** -- is the hand depth physically plausible (~0.5-0.7 m at a bench).

A temporal model can win the static-consistency row (whole-scene) yet NOT win the wrist row: the
wrist MOVES, and a moving articulated hand is the hard case. If that happens, it is the finding,
not a bug -- report it and stop before retargeting on a hand we cannot place to < 4 mm.

Model-agnostic: it consumes {name: DepthResult} + a HandResult, so UniDepth, MoGe-2, and a
video-depth model are scored identically on the same hand keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from actuate.config import Side
from actuate.perception.depth.consistency import consistency_score, sample_tracked_depths
from actuate.perception.depth.unidepth import smooth_root_depth, solve_root_depth

#: The gate: 3x reduction from the fair 11.3 mm hand-cloud-fit+smooth baseline.
GATE_WRIST_JITTER_MM = 4.0


@dataclass
class DepthBenchmarkRow:
    name: str
    n_frames: int
    #: wrist z-jitter placing the wrist at raw per-frame solved depth (mm/frame)
    wrist_jitter_raw_mm: float
    #: ...after the hand-cloud fit's temporal smoothing (the fair, best-effort number)
    wrist_jitter_smoothed_mm: float
    median_wrist_depth_m: float
    #: static-scene depth-map consistency (median frame-to-frame wobble %)
    static_wobble_pct: float
    static_wobble_mm: float

    def line(self) -> str:
        return (
            f"{self.name:26s} | wrist z-jit raw {self.wrist_jitter_raw_mm:5.1f}  "
            f"smoothed {self.wrist_jitter_smoothed_mm:5.1f} mm | depth "
            f"{self.median_wrist_depth_m:.2f} m | static {self.static_wobble_pct:4.1f}% "
            f"({self.static_wobble_mm:4.1f} mm)"
        )


def _right_hand(hs):
    return next((h for h in hs if h.side == Side.RIGHT), hs[0] if hs else None)


def wrist_jitter(depth, hands, max_frames: int | None = None) -> tuple[float, float, float]:
    """Place the wrist at `depth` and measure z-jitter. Returns (raw_mm, smoothed_mm, median_m).

    Exactly the Part C placement: per frame, solve the wrist root depth from the whole hand
    keypoint cloud against the depth map, then temporally smooth. The jitter of that placed z is
    the number the gate is about.
    """
    K = depth.intrinsics
    frame_ids = sorted(set(depth.frames) & set(hands.frames))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if len(frame_ids) < 2 or K is None:
        return float("nan"), float("nan"), float("nan")

    n = max(frame_ids) + 1
    raw = np.full(n, np.nan)
    for i in frame_ids:
        h = _right_hand(hands.frames.get(i, []))
        df = depth.frames.get(i)
        if h is None or df is None:
            continue
        raw[i] = solve_root_depth(df.depth_m, df.confidence, h.keypoints_2d, h.keypoints_3d)
    sm = smooth_root_depth(raw)

    z_raw = raw[np.isfinite(raw)]
    z_sm = sm[np.isfinite(sm)]
    raw_mm = float(np.std(np.diff(z_raw)) * 1000) if z_raw.size > 1 else float("nan")
    sm_mm = float(np.std(np.diff(z_sm)) * 1000) if z_sm.size > 1 else float("nan")
    median_m = float(np.nanmedian(raw))
    return raw_mm, sm_mm, median_m


def benchmark_row(name: str, depth, hands, video_frames, max_frames=None) -> DepthBenchmarkRow:
    raw_mm, sm_mm, med_m = wrist_jitter(depth, hands, max_frames=max_frames)
    try:
        td = sample_tracked_depths(depth, video_frames, max_frames=max_frames)
        cs = consistency_score(td)
        static_pct, static_mm = cs.median_wobble_pct, cs.wobble_mm
    except Exception:
        static_pct = static_mm = float("nan")
    n = len(set(depth.frames) & set(hands.frames))
    return DepthBenchmarkRow(
        name=name, n_frames=n,
        wrist_jitter_raw_mm=raw_mm, wrist_jitter_smoothed_mm=sm_mm,
        median_wrist_depth_m=med_m, static_wobble_pct=static_pct, static_wobble_mm=static_mm,
    )


def run_benchmark(models: dict, hands, video_frames, *, baseline="UniDepthV2",
                  max_frames=None) -> str:
    """Score every {name: DepthResult} on the same hand, print the table + gate verdict.

    Returns the formatted report (also suitable for writing to depth_ab.txt on Kaggle).
    """
    rows = {name: benchmark_row(name, dr, hands, video_frames, max_frames=max_frames)
            for name, dr in models.items()}

    lines = [
        "Depth benchmark -- wrist z-jitter is the gate (place the hand, measure the trajectory).",
        f"GATE: temporal model must reach < {GATE_WRIST_JITTER_MM:.0f} mm/frame smoothed "
        "(3x below the fair 11.3 mm hand-cloud-fit baseline; NOT the 20.7 mm bbox strawman).",
        "",
    ]
    lines += [rows[n].line() for n in models]

    base = rows.get(baseline)
    lines.append("")
    for name, r in rows.items():
        if name == baseline:
            continue
        verdict = "PASS" if r.wrist_jitter_smoothed_mm < GATE_WRIST_JITTER_MM else "does NOT pass"
        note = ""
        both_finite = (base and np.isfinite(base.wrist_jitter_smoothed_mm)
                       and np.isfinite(r.wrist_jitter_smoothed_mm))
        if both_finite and r.wrist_jitter_smoothed_mm > 1e-6 and base.wrist_jitter_smoothed_mm > 0:
            fold = base.wrist_jitter_smoothed_mm / r.wrist_jitter_smoothed_mm
            note = f" ({fold:.1f}x vs {baseline})"
            if r.static_wobble_pct < base.static_wobble_pct and r.wrist_jitter_smoothed_mm >= GATE_WRIST_JITTER_MM:
                note += " -- wins static consistency but NOT the wrist; monocular floor stands."
        lines.append(f"GATE [{name}]: {r.wrist_jitter_smoothed_mm:.1f} mm smoothed -> "
                     f"{verdict}{note}")
    report = "\n".join(lines)
    return report
