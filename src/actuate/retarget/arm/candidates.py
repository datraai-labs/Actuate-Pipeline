"""Candidate root-frame selection (Master Spec §L5, EgoInfinity).

The flow-matching estimator samples several root hypotheses (the map wrist->root is one-to-many).
This scores each by actually running IK over the wrist trajectory in that root frame and picks
the best. Score components (per the spec): IK convergence rate, residual tracking error,
manipulability (distance from singularity), joint-limit margin, and trajectory smoothness.

Optionally clusters the raw samples first (k-means over root position+rotation) so near-duplicate
hypotheses are not scored repeatedly -- but scoring all is cheap at the default sample count, so
clustering is a de-dup, not a necessity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from actuate.retarget.arm.robot import RobotModel
from actuate.retarget.arm.simdata import rot6d_to_matrix
from actuate.schema import SE3


@dataclass
class CandidateScore:
    root_pos: np.ndarray
    root_R: np.ndarray
    convergence: float          # fraction of frames IK converged
    residual_mm: float          # mean position residual over converged frames
    manipulability: float       # mean, higher = further from singularity
    joint_margin: float         # min normalised distance to a joint limit (0=at limit)
    smoothness: float           # mean joint speed (lower = smoother); NaN if <2 frames
    joint_traj: np.ndarray      # (T, n) solved configs
    ee_traj: list               # (T,) SE3, FK of the solved configs (in root frame)

    def rank_key(self) -> tuple:
        # convergence dominates; then smoother, more manipulable, better margin, lower residual
        return (self.convergence, -self.smoothness, self.manipulability, self.joint_margin,
                -self.residual_mm)


def wrist_to_root(wrist_pos_cam, wrist_rot6d_cam, root_pos, root_R) -> list[SE3]:
    """Express a camera-frame wrist trajectory in the robot base (root) frame: T_root^-1 * wrist."""
    out = []
    for wp, w6 in zip(wrist_pos_cam, wrist_rot6d_cam):
        Rw = rot6d_to_matrix(w6)
        p_root = root_R.T @ (np.asarray(wp) - root_pos)
        R_root = root_R.T @ Rw
        q = Rotation.from_matrix(R_root).as_quat()  # xyzw
        out.append(SE3(position_m=tuple(float(x) for x in p_root),
                       quaternion_wxyz=(float(q[3]), float(q[0]), float(q[1]), float(q[2]))))
    return out


def score_candidate(robot: RobotModel, root_pos, root_R, wrist_pos_cam, wrist_rot6d_cam,
                    ik_restarts: int = 2) -> CandidateScore:
    targets = wrist_to_root(wrist_pos_cam, wrist_rot6d_cam, root_pos, root_R)
    q_prev = None
    js, residuals, manips, margins, conv = [], [], [], [], 0
    ee = []
    for tgt in targets:
        res = robot.ik(tgt, q0=q_prev, restarts=ik_restarts)
        js.append(res.q)
        ee.append(robot.fk(res.q))
        if res.converged:
            conv += 1
            residuals.append(res.pos_err_mm)
            manips.append(robot.manipulability(res.q))
            lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
            margins.append(float(np.min(np.minimum(res.q - lo, hi - res.q) / (hi - lo))))
            q_prev = res.q
    js = np.array(js)
    smooth = float(np.mean(np.linalg.norm(np.diff(js, axis=0), axis=1))) if len(js) > 1 else float("nan")
    return CandidateScore(
        root_pos=np.asarray(root_pos), root_R=np.asarray(root_R),
        convergence=conv / len(targets),
        residual_mm=float(np.mean(residuals)) if residuals else float("nan"),
        manipulability=float(np.mean(manips)) if manips else 0.0,
        joint_margin=float(np.min(margins)) if margins else 0.0,
        smoothness=smooth, joint_traj=js, ee_traj=ee,
    )


def cluster_candidates(cands, k: int) -> list:
    """k-means over root position+rotation to de-duplicate near-identical hypotheses; returns
    one representative (nearest to each centroid) per cluster. Falls back to `cands` if k>=len."""
    if k >= len(cands):
        return cands
    from scipy.cluster.vq import kmeans2

    feats = np.array([np.concatenate([p, R.reshape(-1)]) for p, R in cands])
    _, labels = kmeans2(feats, k, seed=0, minit="++", missing="warn")
    reps = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if members.size:
            reps.append(cands[members[0]])
    return reps


def select_best(robot: RobotModel, candidates, wrist_pos_cam, wrist_rot6d_cam,
                cluster_k: int | None = None) -> CandidateScore:
    """Score candidates (optionally cluster-deduped first) and return the best by rank_key."""
    cands = cluster_candidates(candidates, cluster_k) if cluster_k else candidates
    scored = [score_candidate(robot, p, R, wrist_pos_cam, wrist_rot6d_cam) for p, R in cands]
    return max(scored, key=lambda s: s.rank_key())


def candidate_spread(candidates) -> float:
    """Positional spread of the raw candidate roots (gate 4: >0 means distinct hypotheses)."""
    pos = np.array([p for p, _ in candidates])
    return float(np.linalg.norm(pos.std(0)))
