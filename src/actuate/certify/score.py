"""L4 certification -- the quality certificate becomes real numbers (Master Spec §4 L4).

Until Phase 5 the certificate slots existed but nothing computed them: `quality` was a shim
rescaling v1's EIS, `speed` was a raw frame count mislabelled "binned", and the components
behind the composite were not carried at all. This module computes each component from its
actual source and publishes them ALL, because a composite nobody can decompose is a score
nobody can dispute.

### The one rule that shapes everything here

`None` means NOT MEASURED, never zero. A bare-hand rig has no grasp sensor, so its
`contact_consistency` is None -- writing 0.0 would claim "measured, catastrophic" about a
channel that was never observed (the same schema discipline the contact fields carry).
The composite renormalises over what WAS measured.

### Layering note

`certify` sits BELOW `retarget` in the import contract, so L5 results (sim validation,
reconciliation) arrive as duck-typed objects -- attribute access only, no imports. The CLI is
what wires the two layers together.

**Consent is untouched.** The fail-closed gate lives in `actuate.io.consent` and runs on
every delivery write. Scoring happens alongside it; quality NEVER substitutes for consent
(gate 4 tests exactly this: quality=5 with consent=pending still blocks).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from actuate.config import Channel, get_rig
from actuate.schema import CanonicalEpisode
from actuate.schema.episode import CertificateComponents, EpisodeMeta, StrategyAlignment

#: Component weights for the 1-5 composite. Perception dominates because on a vision-only
#: rig it IS the data; calibration is next because the guessed-intrinsics bug (fx off 1.7x)
#: silently corrupted every back-projection until measured. Renormalised over the non-None
#: components, so an unmeasured channel neither helps nor hurts.
QUALITY_WEIGHTS = {
    "sync_integrity": 0.15,
    "calibration_completeness": 0.30,
    "perception_confidence": 0.35,
    "contact_consistency": 0.05,
    "ik_convergence_rate": 0.15,
}

# These weights/bands are engineering priors, not calibrated against downstream policy
# success. The flag is carried into every certificate until such a calibration exists.
THRESHOLDS_PROVISIONAL = True
THRESHOLD_SET_VERSION = "eis-v1-uncalibrated"

#: Speed bins over the episode's real duration (length in steps at nominal fps).
#: 1=fast (<15 s), 2=normal (15-60 s), 3=slow (>60 s).
SPEED_FAST_S, SPEED_SLOW_S = 15.0, 60.0

#: A 1-second segment whose mean perception confidence sits below this is a mistake flag.
MISTAKE_CONFIDENCE_FLOOR = 0.4


@dataclass
class CertificationReport:
    episode_id: str
    components: CertificateComponents
    quality: int
    speed: int
    mistakes: tuple[str, ...]
    strategy_alignment: dict[str, StrategyAlignment]
    retarget_eligibility: dict[str, bool]
    episode: CanonicalEpisode                 # the updated (copied) episode
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        c = self.components
        fmt = lambda v: "not measured" if v is None else f"{v:.2f}"  # noqa: E731
        lines = [
            f"episode {self.episode_id}",
            f"  sync_integrity           : {fmt(c.sync_integrity)}",
            f"  calibration_completeness : {fmt(c.calibration_completeness)}",
            f"  perception_confidence    : {fmt(c.perception_confidence)}",
            f"  contact_consistency      : {fmt(c.contact_consistency)}",
            f"  ik_convergence_rate      : {fmt(c.ik_convergence_rate)}",
            f"  quality                  : {self.quality}/5",
            f"  speed                    : {self.speed} (1=fast 2=normal 3=slow)",
            f"  mistakes                 : {len(self.mistakes)} flag(s)",
        ]
        for e, s in self.strategy_alignment.items():
            lines.append(f"  strategy_alignment[{e}]  : "
                         f"{'ok' if s.ok else 'FLAGGED: ' + (s.reason or '')}")
        for e, ok in self.retarget_eligibility.items():
            lines.append(f"  retarget_eligibility[{e}]: {ok}")
        return "\n".join(lines)


# ------------------------------------------------------------------ components
def sync_integrity(session_dir: Path | None) -> float | None:
    """1 - (max cross-stream timestamp drift / one frame period). From the L0 sync report.

    Drift beyond a full frame period scores 0 -- streams that far apart cannot be paired
    frame-to-frame at all. None when there is no sync report to read (not measured).
    """
    if session_dir is None:
        return None
    session_dir = Path(session_dir)
    cert = session_dir / "quality_certificate.json"
    meta = session_dir / "session_meta.json"
    drift_ms = None
    if cert.exists():
        try:
            eps = json.loads(cert.read_text(encoding="utf-8"))["episodes"]
            drift_ms = max(e["components"]["sync_drift"]["max_drift_ms"] for e in eps)
        except (KeyError, ValueError):
            pass
    if drift_ms is None:
        h5_path = session_dir / "session.h5"
        if not h5_path.exists():
            return None
        try:
            import h5py

            with h5py.File(h5_path, "r") as h5:
                sample_ts = np.asarray(h5["imu/timestamp_ns"][:], dtype=np.float64)
                video_ts = np.asarray(h5["imu/video_timestamp_ns"][:], dtype=np.float64)
            if len(sample_ts) != len(video_ts) or not len(sample_ts):
                return None
            drift_ms = float(np.max(np.abs(sample_ts - video_ts)) / 1e6)
        except (KeyError, OSError):
            return None
    fps = 30.0
    if meta.exists():
        fps = float(json.loads(meta.read_text(encoding="utf-8")).get("fps_nominal", 30.0))
    return max(0.0, 1.0 - drift_ms / (1000.0 / fps))


def calibration_completeness(episode: CanonicalEpisode, *,
                             intrinsics_measured: bool = False) -> float | None:
    """Real-vs-approximated intrinsics, weighted by how much of the episode SLAM tracked.

    `intrinsics_measured` defaults to False because nothing in this pipeline measures
    intrinsics today -- the shipped values are guesses, and the guess was a REAL bug
    (fx=1104 assumed vs ~660 measured, scaling every back-projection by 1.7x). Approximated
    intrinsics cap the score at 0.3; a calibrated rig lifts the cap to 1.0.
    """
    if not episode.frames:
        return None
    tracked = sum(1 for f in episode.frames if f.camera_pose is not None)
    slam_frac = tracked / len(episode.frames)
    cap = 1.0 if intrinsics_measured else 0.3
    return cap * (0.5 + 0.5 * slam_frac)


def perception_confidence(episode: CanonicalEpisode) -> float | None:
    """Mean of explicitly calibrated perception probabilities only.

    Detector scores and UniDepth relative weights are useful signals, but calibration
    research shows they cannot be interpreted as likelihoods of correctness without a
    labeled calibration set. Raw channels remain on the frame/artifact; only keys ending in
    ``_calibrated`` may influence this customer-facing certificate component.
    """
    per_frame = []
    for frame in episode.frames:
        values = [
            value for key, value in frame.confidence.items()
            if key.endswith("_calibrated")
        ]
        if values:
            per_frame.append(float(np.mean(values)))
    return float(np.mean(per_frame)) if per_frame else None


def contact_consistency(episode: CanonicalEpisode) -> float | None:
    """Hardware grasp vs vision contact agreement. None on rigs with no contact sensor.

    On this corpus every rig is bare-hand (head_mounted measures nothing), so this is None
    everywhere today -- kept honest rather than filled with a vision-vs-vision tautology.
    """
    if not get_rig(episode.rig).measures(Channel.CONTACT):
        return None
    agree = total = 0
    for f in episode.frames:
        if f.contact is None or f.interaction_state is None:
            continue
        hw = any(r.confidence > 0.5 for fingers in f.contact.values()
                 for r in fingers.values())
        total += 1
        agree += hw == bool(f.interaction_state.is_grasped)
    return agree / total if total else None


# ------------------------------------------------------------------ composite + meta
def composite_quality(components: CertificateComponents) -> tuple[int, list[str]]:
    """Weighted mean over the MEASURED components, mapped to π0.7's 1-5."""
    vals = {k: getattr(components, k) for k in QUALITY_WEIGHTS}
    measured = {k: v for k, v in vals.items() if v is not None}
    if not measured:
        return 1, ["no component measured; quality floors at 1"]
    wsum = sum(QUALITY_WEIGHTS[k] for k in measured)
    score = sum(QUALITY_WEIGHTS[k] * v for k, v in measured.items()) / wsum
    quality = int(np.clip(round(1 + 4 * score), 1, 5))
    notes = [f"{k} not measured" for k, v in vals.items() if v is None]
    return quality, notes


def speed_bin(episode: CanonicalEpisode) -> int | None:
    """Episode length in steps at nominal fps, binned 1=fast / 2=normal / 3=slow.

    Uses the TIMESTAMP span, not len(frames) -- canonical episodes are often built on a
    frame subsample, and a subsample must not reclassify a 95 s demonstration as fast.
    """
    ts = [f.t for f in episode.frames]
    if len(ts) < 2:
        return None
    duration = max(ts) - min(ts)
    return 1 if duration < SPEED_FAST_S else (3 if duration > SPEED_SLOW_S else 2)


def find_mistakes(episode: CanonicalEpisode, *, window_s: float = 1.0) -> tuple[str, ...]:
    """Per-segment flags where perception confidence collapses (L2+L4).

    1-second windows over the timestamp axis; a window whose mean confidence is below the
    floor is flagged with its time span, so a human reviewer can seek straight to it.
    """
    stamped = []
    for frame in episode.frames:
        values = [
            value for key, value in frame.confidence.items()
            if key.endswith("_calibrated")
        ]
        if values:
            stamped.append((frame.t, float(np.mean(values))))
    if not stamped:
        return ()
    flags = []
    t0 = min(t for t, _ in stamped)
    t_end = max(t for t, _ in stamped)
    w = t0
    while w <= t_end:
        seg = [c for t, c in stamped if w <= t < w + window_s]
        if seg and float(np.mean(seg)) < MISTAKE_CONFIDENCE_FLOOR:
            flags.append(f"low_confidence@{w - t0:.0f}s-{w - t0 + window_s:.0f}s"
                         f"(mean {np.mean(seg):.2f})")
        w += window_s
    return tuple(flags)


# ------------------------------------------------------------------ entry point
def score(
    episode: CanonicalEpisode,
    embodiment: str | None = None,
    *,
    session_dir: Path | None = None,
    intrinsics_measured: bool = False,
    sim_result=None,          # duck-typed L5 SimValidationResult (certify may not import retarget)
    reconcile_result=None,    # duck-typed L5 ReconcileResult
) -> CertificationReport:
    """Compute the full L4 certificate for one episode and return the updated copy.

    L5 inputs are optional: without them `ik_convergence_rate` is None and the
    strategy/eligibility maps stay empty -- reported as not measured, never invented.
    """
    components = CertificateComponents(
        sync_integrity=sync_integrity(session_dir),
        calibration_completeness=calibration_completeness(
            episode, intrinsics_measured=intrinsics_measured),
        perception_confidence=perception_confidence(episode),
        contact_consistency=contact_consistency(episode),
        ik_convergence_rate=(
            None if sim_result is None or sim_result.ik_convergence_rate is None
            else float(sim_result.ik_convergence_rate)),
    )
    quality, notes = composite_quality(components)
    speed = speed_bin(episode)
    mistakes = tuple(dict.fromkeys(          # dedup, keep order: new flags then v1's
        find_mistakes(episode) + tuple(episode.episode_meta.mistakes)))

    strategy: dict[str, StrategyAlignment] = dict(episode.strategy_alignment)
    eligibility: dict[str, bool] = dict(episode.retarget_eligibility)
    if embodiment is not None:
        if reconcile_result is not None:
            strategy[embodiment] = StrategyAlignment(
                ok=bool(reconcile_result.ok),
                reason=None if reconcile_result.ok else "; ".join(reconcile_result.reasons))
        if sim_result is not None:
            eligibility[embodiment] = bool(sim_result.eligible)
            if sim_result.reasons:
                mistakes += tuple(f"sim_validate[{embodiment}]: {r}"
                                  for r in sim_result.reasons)

    derivation_notes = dict(episode.derivation_notes)
    derivation_notes["certificate_thresholds"] = (
        f"PROVISIONAL ({THRESHOLD_SET_VERSION}). Component weights and 1-5 quality bands "
        "have not been calibrated against downstream policy performance."
    )
    updated = episode.model_copy(update={
        "episode_meta": EpisodeMeta(quality=quality, speed=speed, mistakes=mistakes,
                                    components=components),
        "strategy_alignment": strategy,
        "retarget_eligibility": eligibility,
        "derivation_notes": derivation_notes,
    })
    return CertificationReport(
        episode_id=episode.episode_id, components=components, quality=quality,
        speed=speed if speed is not None else 0, mistakes=mistakes,
        strategy_alignment=strategy, retarget_eligibility=eligibility,
        episode=updated,
        notes=notes + [
            f"thresholds_provisional={THRESHOLDS_PROVISIONAL} "
            f"threshold_set={THRESHOLD_SET_VERSION}"
        ],
    )
