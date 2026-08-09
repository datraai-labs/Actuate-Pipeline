"""L5 finger retargeting -- human fingertips -> dexterous-hand joints (Master Spec §L5).

GeoRT-style geometric (kinematic, contact-blind) retargeting. `train` needs no paired data: it
samples the robot hand's own C-space and fits a per-human fingertip calibration. `run` maps a
canonical episode's hand keypoints to per-frame robot finger joints.

**Honest status:** validated on sim + synthetic MANO. We have no physical dexterous hand, so
real-hand validation is DEFERRED. Contact-blind by construction (fingertip geometry only, no
grasp force). Allegro has no pinky -> the human pinky is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from actuate.config import Side
from actuate.retarget.finger.geort import Calibration, GeoRT, fit_calibration
from actuate.retarget.finger.hand import (
    ALLEGRO_MANO_TIPS,
    MANO_PINKY_TIP,
    HandModel,
    load_hand,
)


@dataclass
class FingerRetargetResult:
    embodiment: str
    frame_ids: list[int]
    finger_traj: np.ndarray          # (T, n_joints)
    within_limits_pct: float
    self_collision_frames: int
    mean_openness: float             # mean fingertip distance from the palm (m) -- open vs fist

    def summary(self) -> str:
        return (f"{self.embodiment}: {len(self.frame_ids)} frames | joints in-limits "
                f"{self.within_limits_pct:.0f}% | self-collision frames "
                f"{self.self_collision_frames} | mean openness {self.mean_openness:.3f} m")


def human_fingertips(keypoints_3d, tip_ids=ALLEGRO_MANO_TIPS, global_orient=None) -> np.ndarray:
    """MANO/MediaPipe 21 keypoints -> (n_tips, 3) fingertips in the hand's CANONICAL frame.

    Wrist-relative is not enough. WiLoR's `keypoints_3d` are root-relative but still carry the
    hand's `global_orient` -- its rotation in the camera frame -- whereas the calibration poses are
    generated at global_orient=0. Comparing the two directly compares a rotated hand against an
    unrotated reference: measured, that dropped the per-finger alignment with the canonical open
    pose to cosine 0.68, and de-rotating restores it to 0.95.

    So pass `global_orient` (axis-angle, from the MANO fit) whenever you have it. Without it the
    tips are merely wrist-relative -- correct only if the hand is already canonically oriented.

    Keypoint order is MediaPipe (thumb 1-4, index 5-8, ...), verified against WiLoR output: its
    thumb chain matches MANO's to 1 mm across all four joints. The pinky (kp 20) is dropped for
    Allegro -- it has no pinky.
    """
    kp = np.asarray(keypoints_3d, dtype=np.float64)
    rel = kp - kp[0]
    if global_orient is not None:
        from scipy.spatial.transform import Rotation

        rel = Rotation.from_rotvec(np.asarray(global_orient,
                                              dtype=np.float64).reshape(3)).inv().apply(rel)
    return np.array([rel[i] for i in tip_ids])


def train(
    hand: str | HandModel,
    out_model: Path,
    *,
    human_calibration: dict | None = None,
    n_robot_samples: int = 20000,
    epochs: int = 800,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 200,
) -> GeoRT:
    """Train GeoRT for `hand`. Unsupervised: the robot's own C-space + a human calibration.

    `human_calibration`: {pose_name: (n_tips, 3)} wrist-relative human fingertips for CANONICAL
    poses -- at minimum `{"open": ..., "fist": ...}`, i.e. what you get by asking a person to
    open their hand and make a fist. Pairing those against the robot's own open/fist configs is
    what pins the curl direction; a statistics-only alignment inverts it (see fit_calibration).
    If None, human input is assumed already in the robot's fingertip frame -- fine for sim
    tests, wrong for a real human.
    """
    hm = load_hand(hand) if isinstance(hand, str) else hand
    rng = np.random.default_rng(seed)
    qs, tips = hm.sample_cspace(n_robot_samples, rng)

    g = GeoRT(hm.n_tips, hm.n, hm.joint_limits, device=device)
    if human_calibration is not None:
        g.calibration = fit_calibration(human_calibration, hm.canonical_fingertips())
    g.train(tips, qs, epochs=epochs, seed=seed, log_every=log_every)
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    g.save(out_model)
    return g


def run(
    canonical,
    hand_model: str | HandModel,
    finger_model: Path | GeoRT,
    *,
    embodiment: str = "allegro",
    side: Side = Side.RIGHT,
    device: str = "cpu",
) -> FingerRetargetResult:
    """Retarget a canonical episode's hand keypoints to per-frame dexterous-hand joints."""
    hm = load_hand(hand_model) if isinstance(hand_model, str) else hand_model
    g = finger_model if isinstance(finger_model, GeoRT) else GeoRT.load(Path(finger_model),
                                                                       device=device)
    joints, fids = [], []
    for f in canonical.frames:
        h = f.hands.get(side) or (next(iter(f.hands.values())) if f.hands else None)
        if h is None or h.keypoints_3d is None:
            continue
        go = h.mano.global_orient if h.mano is not None else None
        q = g.predict(human_fingertips(h.keypoints_3d, global_orient=go))
        joints.append(q)
        fids.append(f.frame_idx)
    if not joints:
        raise ValueError("no frames with hand keypoints; nothing to retarget")

    J = np.array(joints)
    in_lim = float(np.mean([hm.within_limits(q) for q in J]) * 100)
    collisions = int(sum(hm.self_collision_count(q) > 0 for q in J))
    openness = float(np.mean([np.linalg.norm(hm.fk_fingertips(q), axis=1).mean() for q in J]))
    return FingerRetargetResult(
        embodiment=embodiment, frame_ids=fids, finger_traj=J,
        within_limits_pct=in_lim, self_collision_frames=collisions, mean_openness=openness,
    )


__all__ = [
    "ALLEGRO_MANO_TIPS",
    "MANO_PINKY_TIP",
    "Calibration",
    "FingerRetargetResult",
    "GeoRT",
    "HandModel",
    "fit_calibration",
    "human_fingertips",
    "load_hand",
    "run",
    "train",
]
