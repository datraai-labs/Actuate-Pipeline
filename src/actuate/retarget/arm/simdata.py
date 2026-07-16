"""Procedural sim data for the root-frame estimator (Master Spec §L5, EgoInfinity approach).

The estimator learns to predict the robot BASE (root) frame from a wrist trajectory + gravity,
expressed in the ego/camera frame. Training needs no real capture -- it is generated in sim:

  1. sample a robot base pose T_root in a world frame (gravity = world -z),
  2. sample a smooth OU joint trajectory within the robot's limits,
  3. forward-kinematics -> wrist poses relative to the base,
  4. place the base at T_root and a camera at a random tilt -> express the wrist trajectory and
     gravity in the CAMERA frame,
  5. record (wrist_traj_cam, gravity_cam) -> T_root_cam.

Because every wrist pose is a forward-kinematics image of an in-limits joint config, the pairs
are valid BY CONSTRUCTION: reachable wrist, joints within limits. That is gate 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from actuate.retarget.arm.robot import RobotModel


@dataclass
class SimPair:
    wrist_pos_cam: np.ndarray     # (T, 3) wrist positions in the camera frame
    wrist_rot6d_cam: np.ndarray   # (T, 6) wrist orientation, first two rotation-matrix columns
    gravity_cam: np.ndarray       # (3,) gravity direction in the camera frame
    root_pos_cam: np.ndarray      # (3,) robot base position in the camera frame  [TARGET]
    root_rot6d_cam: np.ndarray    # (6,) robot base orientation, 6D                [TARGET]
    joint_traj: np.ndarray        # (T, n) the joint configs (all within limits)


def ou_joint_trajectory(robot: RobotModel, length: int, rng, theta=0.15, sigma=0.3, dt=1.0):
    """Smooth Ornstein-Uhlenbeck random walk in joint space, mean-reverting to mid-range,
    clamped to limits. Produces the correlated, non-degenerate motion real manipulation has."""
    lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
    mid, span = (lo + hi) / 2, (hi - lo) / 2
    q = robot.random_config(rng, margin=0.2)
    out = [q.copy()]
    for _ in range(length - 1):
        q = q + theta * (mid - q) * dt + sigma * span * np.sqrt(dt) * rng.standard_normal(robot.n)
        q = robot.clamp(q)
        out.append(q.copy())
    return np.array(out)


def _rot6d(R: Rotation) -> np.ndarray:
    """First two columns of the rotation matrix (the standard continuous 6D representation)."""
    M = R.as_matrix()
    return np.concatenate([M[:, 0], M[:, 1]])


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the 6D representation back to a valid rotation matrix."""
    a1, a2 = r6[:3], r6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - (b1 @ a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def generate_pair(robot: RobotModel, length: int, rng, base_pos_range=0.6) -> SimPair:
    jtraj = ou_joint_trajectory(robot, length, rng)
    wrist_base = [robot.fk(q) for q in jtraj]  # wrist relative to the robot base

    # robot base pose in the world (gravity = world -z)
    base_yaw = rng.uniform(-np.pi, np.pi)
    R_wb = Rotation.from_euler("z", base_yaw)
    t_wb = rng.uniform(-base_pos_range, base_pos_range, 3)

    # camera pose in the world: a head-cam looks down at a workbench -> random tilt + position
    cam_eul = rng.uniform([-0.6, -0.6, -np.pi], [0.6, 0.6, np.pi])  # pitch,roll,yaw-ish
    R_wc = Rotation.from_euler("xyz", cam_eul)
    t_wc = rng.uniform(-0.5, 0.5, 3)

    R_cw = R_wc.inv()

    def to_cam(pos_w, R_w):
        return R_cw.apply(pos_w - t_wc), R_cw * R_w

    wrist_pos_cam, wrist_rot6d_cam = [], []
    for wb in wrist_base:
        wpos_w = t_wb + R_wb.apply(np.asarray(wb.position_m))
        cw, cx, cy, cz = wb.quaternion_wxyz
        wR_w = R_wb * Rotation.from_quat([cx, cy, cz, cw])
        p_c, R_c = to_cam(wpos_w, wR_w)
        wrist_pos_cam.append(p_c)
        wrist_rot6d_cam.append(_rot6d(R_c))

    root_pos_cam, root_R_cam = to_cam(t_wb, R_wb)
    gravity_cam = R_cw.apply([0, 0, -1.0])

    return SimPair(
        wrist_pos_cam=np.array(wrist_pos_cam),
        wrist_rot6d_cam=np.array(wrist_rot6d_cam),
        gravity_cam=gravity_cam,
        root_pos_cam=root_pos_cam,
        root_rot6d_cam=_rot6d(root_R_cam),
        joint_traj=jtraj,
    )


def generate_dataset(robot: RobotModel, n_pairs: int, length: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [generate_pair(robot, length, rng) for _ in range(n_pairs)]


def validate_pairs(robot: RobotModel, pairs: list[SimPair]) -> dict:
    """Gate 1: every pair must have joint configs within limits and reachable wrist positions
    (finite, in a sane workspace). Returns a report; `all_valid` is the gate."""
    n_within = 0
    n_reachable = 0
    total_frames = 0
    for p in pairs:
        total_frames += len(p.joint_traj)
        n_within += sum(robot.within_limits(q) for q in p.joint_traj)
        n_reachable += int(np.isfinite(p.wrist_pos_cam).all()
                           and np.all(np.linalg.norm(p.wrist_pos_cam, axis=1) < 5.0))
    return {
        "n_pairs": len(pairs),
        "frames": total_frames,
        "joint_within_limits_pct": 100.0 * n_within / max(1, total_frames),
        "trajs_reachable_pct": 100.0 * n_reachable / max(1, len(pairs)),
        "all_valid": n_within == total_frames and n_reachable == len(pairs),
    }
