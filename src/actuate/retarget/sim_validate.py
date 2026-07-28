"""L5 sim validation -- replay the retargeted trajectory in MuJoCo before certifying it (§L5).

The retarget branches emit joint trajectories that LOOK plausible as numbers. This module is the
only place those numbers meet a physics model: every frame is replayed through the embodiment's
MuJoCo model and checked for joint-limit violations and self-collision. Its verdict is what the
certificate's `retarget_eligibility.<embodiment>` reports -- an episode whose trajectory
violates limits or interpenetrates is NOT eligible for that embodiment, whatever its other
scores say.

**Honest scope.** This is a KINEMATIC replay: limits, self-collision, and the IK convergence the
arm branch measured. It does not simulate dynamics, contact stability, or slip -- there is no
object model and no physical hand, so "eligible" means "kinematically executable", not "the
grasp will hold". Contact-stability validation needs hardware or a contact-rich sim we do not
have; stated rather than implied otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Below this IK convergence the arm branch was guessing too often to certify.
MIN_IK_CONVERGENCE = 0.9
#: A per-source-frame joint jump above this is a solver teleport, not robot motion.  Source
#: frame spacing is respected, so sparse dashboard previews are not judged as 30 Hz samples.
MAX_JOINT_DELTA_RAD_PER_FRAME = 0.3


@dataclass
class SimValidationResult:
    embodiment: str
    n_frames: int
    joint_limit_violations: int          # frames with any joint outside its range
    collision_count: int                 # frames with >=1 interpenetrating pair
    temporal_discontinuities: int        # adjacent samples that teleport after frame-gap scaling
    ik_convergence_rate: float | None    # from the arm branch; None if not provided
    eligible: bool                       # the certificate's retarget_eligibility.<embodiment>
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        ik = "n/a" if self.ik_convergence_rate is None else f"{self.ik_convergence_rate:.0%}"
        verdict = "ELIGIBLE" if self.eligible else "NOT ELIGIBLE: " + "; ".join(self.reasons)
        return (f"{self.embodiment}: {self.n_frames} frames | limit violations "
                f"{self.joint_limit_violations} | collision frames {self.collision_count} | "
                f"teleports {self.temporal_discontinuities} | "
                f"IK convergence {ik} | {verdict}")


def _replay(model, traj: np.ndarray) -> tuple[int, int]:
    """(frames outside joint limits, frames with interpenetration) over a joint trajectory."""
    limit_bad = 0
    collision_bad = 0
    for q in np.asarray(traj, dtype=np.float64):
        if not model.within_limits(q):
            limit_bad += 1
        if model.self_collision_count(q) > 0:
            collision_bad += 1
    return limit_bad, collision_bad


def _temporal_discontinuities(
    traj: np.ndarray,
    frame_ids: list[int] | None,
    threshold: float = MAX_JOINT_DELTA_RAD_PER_FRAME,
) -> int:
    if len(traj) < 2:
        return 0
    if frame_ids is None or len(frame_ids) != len(traj):
        gaps = np.ones(len(traj) - 1, dtype=np.float64)
    else:
        gaps = np.maximum(1.0, np.diff(np.asarray(frame_ids, dtype=np.float64)))
    delta_per_frame = np.abs(np.diff(traj, axis=0)) / gaps[:, None]
    return int(np.sum(np.max(delta_per_frame, axis=1) > threshold))


def run(canonical, embodiment: str, trajectory, *, robot_model=None, hand_model=None,
        finger_traj=None, ik_convergence: float | None = None) -> SimValidationResult:
    """Validate a retargeted trajectory against the embodiment's MuJoCo model.

    `trajectory`: (T, n_arm) arm joint trajectory, or an ArmRetargetResult (its `joint_traj`
    and `convergence` are used). `finger_traj` + `hand_model` add the dexterous-hand replay.
    `canonical` scopes the claim: the result certifies THIS episode's trajectory only.
    """
    frame_ids = None
    if hasattr(trajectory, "joint_traj"):          # ArmRetargetResult
        if ik_convergence is None:
            ik_convergence = float(trajectory.convergence)
        frame_ids = list(getattr(trajectory, "frame_ids", [])) or None
        trajectory = trajectory.joint_traj
    traj = np.asarray(trajectory, dtype=np.float64)

    if robot_model is None:
        from actuate.retarget.arm.robot import load_robot

        robot_model = load_robot(embodiment)

    reasons: list[str] = []
    limit_bad, collision_bad = _replay(robot_model, traj)
    temporal_bad = _temporal_discontinuities(traj, frame_ids)
    n = len(traj)

    if finger_traj is not None:
        if hand_model is None:
            raise ValueError("finger_traj given without a hand_model to replay it in")
        f_limit, f_coll = _replay(hand_model, np.asarray(finger_traj))
        limit_bad += f_limit
        collision_bad += f_coll

    if limit_bad:
        reasons.append(f"{limit_bad} frame(s) violate joint limits")
    if collision_bad:
        reasons.append(f"{collision_bad} frame(s) self-collide")
    if temporal_bad:
        reasons.append(
            f"{temporal_bad} trajectory jump(s) exceed "
            f"{MAX_JOINT_DELTA_RAD_PER_FRAME} rad/source-frame"
        )
    if ik_convergence is not None and ik_convergence < MIN_IK_CONVERGENCE:
        reasons.append(f"IK convergence {ik_convergence:.0%} < {MIN_IK_CONVERGENCE:.0%}")

    return SimValidationResult(
        embodiment=embodiment, n_frames=n,
        joint_limit_violations=limit_bad, collision_count=collision_bad,
        temporal_discontinuities=temporal_bad,
        ik_convergence_rate=ik_convergence,
        eligible=not reasons, reasons=reasons,
    )
