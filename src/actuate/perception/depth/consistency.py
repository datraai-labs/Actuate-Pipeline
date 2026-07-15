"""Temporal-consistency metric for depth -- the A/B test for UniDepth vs a video-depth model.

Part C established the binding failure of single-image depth by PHYSICS: track background points
that cannot move, and measure how much their depth reading wobbles frame-to-frame. UniDepthV2
wobbled ~2.1% (~13 mm at the hand), which exceeds the real per-frame hand motion, so per-frame
differencing is below the noise floor.

This packages that test as a reusable score so any two depth models can be compared head-to-head
on the same tracked points:

    score(model_A) vs score(model_B)   -- lower wobble == more temporally consistent

A genuinely temporal/video-depth model should score MEASURABLY lower wobble than UniDepth. If it
does not, it does not beat the floor and there is no point paying for it. The metric is the
decision, not a vibe.

Nothing here needs a GPU -- it consumes a DepthResult (already computed) plus the video, so it
runs on Kaggle right after each model, or locally on cached DepthResults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConsistencyReport:
    n_points: int
    n_frames: int
    #: Median frame-to-frame relative change of a STATIC point's depth. 0 == perfectly stable.
    median_wobble_pct: float
    p90_wobble_pct: float
    #: Full peak-to-peak swing of the tracked points' median depth, as % of the mean. The
    #: headline number: a static scene reading e.g. 10% here is 10% pure model noise.
    full_swing_pct: float
    #: Absolute wobble in millimetres at the mean tracked depth -- comparable to hand motion.
    wobble_mm: float
    mean_depth_m: float

    def summary(self) -> str:
        return (
            f"{self.n_points} static pts x {self.n_frames} frames | "
            f"frame-to-frame wobble median {self.median_wobble_pct:.2f}% "
            f"(p90 {self.p90_wobble_pct:.2f}%) | full swing {self.full_swing_pct:.0f}% | "
            f"~{self.wobble_mm:.1f} mm at {self.mean_depth_m:.2f} m"
        )


def consistency_score(tracked_depths: np.ndarray) -> ConsistencyReport:
    """Score temporal consistency from a (T, P) matrix of depths at T frames, P static points.

    Pure and unit-testable: feed a constant matrix (perfect model) or constant+noise (a wobbly
    one). The points are assumed STATIC, so every deviation over time is model error.
    """
    d = np.asarray(tracked_depths, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2:
        raise ValueError("tracked_depths must be (T, P) with T >= 2 frames")
    T, P = d.shape

    # per-point frame-to-frame relative change |d[t]-d[t-1]| / d[t-1], over valid pairs
    prev, cur = d[:-1], d[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
    rel = np.abs(cur[ok] - prev[ok]) / prev[ok]
    median_wobble = float(np.median(rel)) if rel.size else float("nan")
    p90_wobble = float(np.percentile(rel, 90)) if rel.size else float("nan")

    # scene-scale swing: the median depth of the tracked cloud per frame (a static quantity)
    per_frame = np.nanmedian(d, axis=1)
    per_frame = per_frame[np.isfinite(per_frame)]
    mean_depth = float(np.mean(per_frame)) if per_frame.size else float("nan")
    full_swing = (
        100.0 * (per_frame.max() - per_frame.min()) / mean_depth
        if per_frame.size and mean_depth > 0
        else float("nan")
    )
    wobble_mm = median_wobble * mean_depth * 1000.0

    return ConsistencyReport(
        n_points=P, n_frames=T,
        median_wobble_pct=100.0 * median_wobble,
        p90_wobble_pct=100.0 * p90_wobble,
        full_swing_pct=full_swing,
        wobble_mm=wobble_mm,
        mean_depth_m=mean_depth,
    )


def sample_tracked_depths(
    depth_result,
    video_frames: list,
    max_points: int = 300,
    patch: int = 5,
    max_frames: int | None = None,
) -> np.ndarray:
    """Track static background points across the video and read each model's depth at them.

    Returns a (T, P) matrix: the depth every tracked point reads at every frame. Points are
    found once (good corners on frame 0) and followed with Lucas-Kanade optical flow; only
    points that survive the whole clip are kept, so the SAME physical spots are compared over
    time. Hand/moving regions get dropped naturally (they fail to track coherently).

    Model-agnostic: `depth_result.frames[i].depth_m` is the only thing read, so UniDepth and a
    video-depth model are scored identically.
    """
    import cv2

    from actuate.perception.depth import sample_depth

    frame_ids = sorted(depth_result.frames)
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    n = min(len(frame_ids), len(video_frames))
    frame_ids = frame_ids[:n]
    gray = [cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY) for i in frame_ids]

    p0 = cv2.goodFeaturesToTrack(gray[0], maxCorners=max_points, qualityLevel=0.01,
                                 minDistance=20)
    if p0 is None:
        raise RuntimeError("no trackable features on frame 0")
    tracks = [p0]
    alive = np.ones(len(p0), dtype=bool)
    for t in range(1, n):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], tracks[-1], None,
                                             winSize=(21, 21), maxLevel=3)
        alive &= (st.ravel() == 1)
        tracks.append(p1)

    idx = np.flatnonzero(alive)
    out = np.full((n, len(idx)), np.nan)
    for t in range(n):
        df = depth_result.frames[frame_ids[t]]
        pts = tracks[t][idx].reshape(-1, 2)
        for j, p in enumerate(pts):
            out[t, j] = sample_depth(df.depth_m, df.confidence, p, patch=patch)[0]
    return out


def compare(depth_a, depth_b, video_frames: list, *, label_a="A", label_b="B",
            max_frames: int | None = None) -> dict:
    """Score two depth models on the SAME tracked static points and print the verdict.

    Returns {label_a: ConsistencyReport, label_b: ConsistencyReport, "winner": label}. Use it
    on Kaggle to decide whether the video-depth model actually beats UniDepth's noise floor.
    """
    ta = sample_tracked_depths(depth_a, video_frames, max_frames=max_frames)
    tb = sample_tracked_depths(depth_b, video_frames, max_frames=max_frames)
    ra, rb = consistency_score(ta), consistency_score(tb)
    winner = label_a if ra.median_wobble_pct <= rb.median_wobble_pct else label_b
    return {label_a: ra, label_b: rb, "winner": winner}
