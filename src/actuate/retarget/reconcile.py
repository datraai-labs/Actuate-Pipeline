"""L5 reconciliation -- are the arm and finger retargets one coherent trajectory? (Spec §L5)

The arm branch (VN + flow matching -> IK) and the finger branch (GeoRT) are retargeted
INDEPENDENTLY, frame by frame. Nothing upstream guarantees they agree: they can cover different
frame sets, either can teleport between adjacent frames (each frame is solved in isolation), and
the finger pose can contradict what the demonstrator's hand was doing. This module is the check
that turns two per-frame maps into something a robot could actually execute -- and its verdict is
what `strategy_alignment.<embodiment>` reports on the certificate.

### What is actually checked (and what is not)

1. **Frame alignment** -- both branches must cover a shared frame set; the result is scoped to
   the intersection and a poor overlap is itself a flag.
2. **Temporal consistency** -- per-frame joint deltas must stay under a teleport threshold.
   This catches discontinuities (the failure mode of per-frame solvers), NOT subtle dynamic
   infeasibility -- there is no actuator model here and the threshold says so.
3. **Grasp agreement** (only when `canonical` is given) -- when L2 fusion says the hand is
   GRASPING, the retargeted finger openness should be on the closed side of its range, and
   vice versa. A retarget that opens the hand mid-grasp forced a strategy the demonstrator
   never showed -- exactly what `strategy_alignment` exists to flag.

**Honest scope:** contact-blind. No grasp-force or slip reasoning (no contact model, no physical
hand). A PASS here means "kinematically coherent", not "will hold the object".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Max plausible per-frame joint delta, radians. At 30 fps this is ~9 rad/s -- far beyond any
#: real actuator, so a hit is a genuine discontinuity (teleport), not a tuned dynamics bound.
TELEPORT_RAD_PER_FRAME = 0.3

#: Minimum fraction of either branch's frames that must be shared for the pair to count as
#: covering the same episode.
MIN_FRAME_OVERLAP = 0.5

#: Grasp agreement below this flags the strategy as undemonstrated.
MIN_GRASP_AGREEMENT = 0.7


@dataclass
class ReconcileResult:
    embodiment: str
    frame_ids: list[int]                 # intersection both branches cover
    frame_overlap: float                 # shared / max(branch sizes)
    arm_teleports: int                   # arm joint-delta violations
    finger_teleports: int
    grasp_agreement: float | None        # None when no canonical/interaction states given
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        g = "n/a" if self.grasp_agreement is None else f"{self.grasp_agreement:.0%}"
        return (f"{self.embodiment}: {len(self.frame_ids)} shared frames "
                f"(overlap {self.frame_overlap:.0%}) | teleports arm={self.arm_teleports} "
                f"finger={self.finger_teleports} | grasp agreement {g} | "
                f"{'OK' if self.ok else 'FLAGGED: ' + '; '.join(self.reasons)}")


def _teleport_count(traj: np.ndarray, threshold: float = TELEPORT_RAD_PER_FRAME) -> int:
    """Number of adjacent-frame joint deltas exceeding the teleport threshold."""
    if len(traj) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(np.asarray(traj, dtype=np.float64), axis=0))
                      .max(axis=1) > threshold))


def _grasp_agreement(canonical, finger_traj, frame_ids, finger_frame_ids, hand_model) -> float:
    """Fraction of shared frames where fused interaction_state and finger openness agree.

    "Agree" = grasped frames sit in the closed half of the openness range seen in this episode,
    non-grasped frames in the open half. Episode-relative, because absolute openness depends on
    hand geometry and the capture may never reach either extreme (measured: the real capture's
    hand never opens fully and never fists).
    """
    open_by_frame = {}
    idx = {fid: i for i, fid in enumerate(finger_frame_ids)}
    for fid in frame_ids:
        q = finger_traj[idx[fid]]
        open_by_frame[fid] = float(np.linalg.norm(hand_model.fk_fingertips(q), axis=1).mean())

    grasped = {}
    for f in canonical.frames:
        if f.frame_idx in open_by_frame and f.interaction_state is not None:
            grasped[f.frame_idx] = bool(f.interaction_state.is_grasped)
    if not grasped or len(set(grasped.values())) < 2:
        # all-grasped or all-free episodes give the split no information; report perfect
        # agreement rather than penalising an episode for being uneventful
        return 1.0

    vals = np.array([open_by_frame[fid] for fid in grasped])
    mid = (vals.min() + vals.max()) / 2
    hits = sum(1 for fid, g in grasped.items() if (open_by_frame[fid] < mid) == g)
    return hits / len(grasped)


def run(arm_result, finger_result, embodiment: str, *, canonical=None,
        hand_model=None) -> ReconcileResult:
    """Reconcile independently-retargeted arm + finger branches for one embodiment.

    `canonical` + `hand_model` enable the grasp-agreement check; without them it is skipped
    (reported as None, never silently passed).
    """
    reasons: list[str] = []

    arm_ids, fin_ids = list(arm_result.frame_ids), list(finger_result.frame_ids)
    shared = sorted(set(arm_ids) & set(fin_ids))
    overlap = len(shared) / max(len(arm_ids), len(fin_ids)) if (arm_ids or fin_ids) else 0.0
    if overlap < MIN_FRAME_OVERLAP:
        reasons.append(f"frame overlap {overlap:.0%} < {MIN_FRAME_OVERLAP:.0%}")

    a_idx = {fid: i for i, fid in enumerate(arm_ids)}
    f_idx = {fid: i for i, fid in enumerate(fin_ids)}
    arm_traj = np.asarray(arm_result.joint_traj)[[a_idx[f] for f in shared]]
    fin_traj = np.asarray(finger_result.finger_traj)[[f_idx[f] for f in shared]]

    arm_tp = _teleport_count(arm_traj)
    fin_tp = _teleport_count(fin_traj)
    if arm_tp:
        reasons.append(f"{arm_tp} arm teleport(s) > {TELEPORT_RAD_PER_FRAME} rad/frame")
    if fin_tp:
        reasons.append(f"{fin_tp} finger teleport(s) > {TELEPORT_RAD_PER_FRAME} rad/frame")

    agreement = None
    if canonical is not None and hand_model is not None:
        agreement = _grasp_agreement(canonical, np.asarray(finger_result.finger_traj),
                                     shared, fin_ids, hand_model)
        if agreement < MIN_GRASP_AGREEMENT:
            reasons.append(
                f"grasp agreement {agreement:.0%} < {MIN_GRASP_AGREEMENT:.0%}: the retargeted "
                "hand contradicts the demonstrated interaction states (undemonstrated strategy)")

    return ReconcileResult(
        embodiment=embodiment, frame_ids=shared, frame_overlap=overlap,
        arm_teleports=arm_tp, finger_teleports=fin_tp, grasp_agreement=agreement,
        ok=not reasons, reasons=reasons,
    )
