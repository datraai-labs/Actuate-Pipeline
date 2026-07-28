"""L5 arm/wrist retargeting -- wrist trajectory -> robot joint trajectory (Master Spec §L5).

Pipeline: extract the wrist trajectory from a canonical episode -> the VN + flow-matching
estimator samples candidate robot base frames -> score each by IK-ing the trajectory and pick the
best -> emit the joint-space and EE-space trajectories as a `RobotAction`.

`train_estimator` generates sim data and trains (a Kaggle job for the full run). `run` is
inference (CPU, seconds). The real-capture validation (gate 3) is deferred until depth is
trustworthy -- `run` works on any wrist trajectory, but the trajectory it retargets is only as
good as the depth that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from actuate.config import ControlMode, Side
from actuate.retarget.arm.candidates import candidate_spread, select_best
from actuate.retarget.arm.estimator import RootFrameEstimator
from actuate.retarget.arm.robot import load_robot
from actuate.schema import SE3
from actuate.schema.episode import RobotAction


@dataclass
class ArmRetargetResult:
    embodiment: str
    frame_ids: list[int]
    joint_traj: np.ndarray            # (T, n)
    root_pos: np.ndarray
    root_R: np.ndarray
    convergence: float
    residual_mm: float
    manipulability: float
    candidate_spread_m: float
    action: RobotAction

    def summary(self) -> str:
        return (f"{self.embodiment}: {len(self.frame_ids)} frames | IK convergence "
                f"{100 * self.convergence:.0f}% | residual {self.residual_mm:.1f} mm | "
                f"manip {self.manipulability:.3f} | candidate spread "
                f"{self.candidate_spread_m:.2f} m")


def default_model_path(embodiment: str) -> Path:
    """User-local location used by both ``train-arm`` and the processing pipeline."""
    from actuate.config.auth import config_dir

    return config_dir() / "models" / f"{embodiment}_root_frame.pt"


def _wrist_rot6d(pose: SE3) -> np.ndarray:
    w, x, y, z = pose.quaternion_wxyz
    M = Rotation.from_quat([x, y, z, w]).as_matrix()
    return np.concatenate([M[:, 0], M[:, 1]])


def _extract_wrist(canonical, side: Side = Side.RIGHT):
    """Pull the (camera-frame) wrist trajectory + gravity from a canonical episode."""
    pos, rot6d, fids = [], [], []
    grav_dir = np.array([0.0, -1.0, 0.0])  # default: camera-down (no IMU gravity wired yet)
    for f in canonical.frames:
        h = f.hands.get(side) or (next(iter(f.hands.values())) if f.hands else None)
        if h is None or h.wrist_pose is None:
            continue
        pos.append(np.asarray(h.wrist_pose.position_m))
        rot6d.append(_wrist_rot6d(h.wrist_pose))
        fids.append(f.frame_idx)
        # if SLAM gave a camera pose, gravity in the camera frame is its rotation applied to
        # world-down; take it from the first frame that has one.
        if f.camera_pose is not None and len(fids) == 1:
            w, x, y, z = f.camera_pose.quaternion_wxyz
            grav_dir = Rotation.from_quat([x, y, z, w]).inv().apply([0, 0, -1.0])
    if len(pos) < 2:
        raise ValueError("episode has fewer than 2 frames with a wrist pose; nothing to retarget")
    g = grav_dir / (np.linalg.norm(grav_dir) + 1e-9)
    return np.array(pos), np.array(rot6d), g, fids


def train_estimator(
    embodiment: str,
    out_model: Path,
    *,
    n_pairs: int = 2000,
    length: int = 32,
    epochs: int = 2000,
    hidden: int = 64,
    device: str = "cpu",
    seed: int = 0,
) -> RootFrameEstimator:
    """Generate sim data for `embodiment` and train the root-frame estimator. Full run: Kaggle."""
    from actuate.retarget.arm.simdata import generate_dataset

    robot = load_robot(embodiment)
    data = generate_dataset(robot, n_pairs, length=length, seed=seed)
    est = RootFrameEstimator(hidden=hidden, device=device)
    est.train(data, epochs=epochs, seed=seed)
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    est.save(out_model)
    return est


def run(
    canonical,
    embodiment: str,
    model: Path | RootFrameEstimator,
    *,
    n_candidates: int = 16,
    cluster_k: int | None = None,
    control_mode: ControlMode = ControlMode.JOINT,
    side: Side = Side.RIGHT,
) -> ArmRetargetResult:
    """Retarget a canonical episode's wrist trajectory to `embodiment` joint space.

    All sampled roots are scored by default.  Clustering is available for large candidate
    sweeps, but at the normal 16 samples it can discard the one collision-free IK solution.
    """
    robot = load_robot(embodiment)
    est = model if isinstance(model, RootFrameEstimator) else RootFrameEstimator.load(Path(model))

    wrist_pos, wrist_rot6d, gravity, fids = _extract_wrist(canonical, side=side)
    cands = est.sample(wrist_pos, wrist_rot6d, gravity, n_samples=n_candidates)
    best = select_best(robot, cands, wrist_pos, wrist_rot6d, cluster_k=cluster_k)

    action = RobotAction(
        embodiment=embodiment,
        control_mode=control_mode,
        joint_traj=tuple(tuple(float(v) for v in q) for q in best.joint_traj),
        ee_traj=tuple(best.ee_traj),
    )
    return ArmRetargetResult(
        embodiment=embodiment, frame_ids=fids, joint_traj=best.joint_traj,
        root_pos=best.root_pos, root_R=best.root_R, convergence=best.convergence,
        residual_mm=best.residual_mm, manipulability=best.manipulability,
        candidate_spread_m=candidate_spread(cands), action=action,
    )


def attach_to_episode(episode, result: ArmRetargetResult):
    """Return a copy of the episode with result.action written into action_robot[embodiment]."""
    action_robot = dict(episode.action_robot)
    action_robot[result.embodiment] = result.action
    return episode.model_copy(update={"action_robot": action_robot})


__all__ = [
    "ArmRetargetResult",
    "RootFrameEstimator",
    "attach_to_episode",
    "default_model_path",
    "run",
    "train_estimator",
]
