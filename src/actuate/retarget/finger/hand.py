"""Dexterous hand model for finger retargeting -- MuJoCo (Master Spec §L5).

Wraps a robot hand MJCF and exposes what geometric retargeting needs: fingertip forward
kinematics in the PALM frame, joint limits, random C-space sampling, and self-collision
detection (the "fingers don't interpenetrate" gate).

Allegro (the hand GeoRT validates on) is 16 DoF over FOUR fingers -- index (ff), middle (mf),
ring (rf), thumb (th). It has **no pinky**, so a 5-finger human hand loses its pinky in
correspondence. That is a real information loss, recorded rather than papered over.
"""

from __future__ import annotations

import numpy as np

#: Allegro finger order and the MANO/MediaPipe fingertip keypoint each maps to.
#: MANO 21-keypoint tips: thumb=4, index=8, middle=12, ring=16, pinky=20.
#: Allegro has no pinky -> keypoint 20 is DROPPED (see module docstring).
ALLEGRO_FINGERS = ("ff", "mf", "rf", "th")
ALLEGRO_TIP_BODIES = ("ff_tip", "mf_tip", "rf_tip", "th_tip")
#: human MANO tip keypoint per Allegro finger, same order as ALLEGRO_FINGERS
ALLEGRO_MANO_TIPS = (8, 12, 16, 4)   # index, middle, ring, thumb
MANO_PINKY_TIP = 20                  # dropped: Allegro has no pinky


class HandModel:
    """A MuJoCo dexterous hand: fingertip FK in the palm frame, limits, self-collision."""

    def __init__(self, mjcf_path: str, tip_bodies, palm_body: str = "palm") -> None:
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, palm_body)
        if self.palm_id < 0:
            raise ValueError(f"palm body {palm_body!r} not found")
        self.tip_ids = []
        for b in tip_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bid < 0:
                raise ValueError(f"tip body {b!r} not found")
            self.tip_ids.append(bid)

        self.joint_ids = [j for j in range(self.model.njnt)
                          if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.joint_ids]
        self.n = len(self.joint_ids)
        self.joint_limits = np.array([self.model.jnt_range[j] for j in self.joint_ids],
                                     dtype=np.float64)  # (n, 2)
        self.n_tips = len(self.tip_ids)

    def _set_q(self, q):
        for adr, v in zip(self.qpos_adr, np.asarray(q, dtype=np.float64)):
            self.data.qpos[adr] = v
        self._mj.mj_forward(self.model, self.data)

    def fk_fingertips(self, q) -> np.ndarray:
        """(n_tips, 3) fingertip positions expressed in the PALM frame."""
        self._set_q(q)
        palm_p = self.data.xpos[self.palm_id]
        palm_R = self.data.xmat[self.palm_id].reshape(3, 3)
        tips = np.array([self.data.xpos[t] for t in self.tip_ids])
        return (tips - palm_p) @ palm_R  # world -> palm frame

    def clamp(self, q) -> np.ndarray:
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def within_limits(self, q) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.joint_limits[:, 0] - 1e-9)
                    and np.all(q <= self.joint_limits[:, 1] + 1e-9))

    def random_config(self, rng, margin: float = 0.0) -> np.ndarray:
        lo = self.joint_limits[:, 0] + margin
        hi = self.joint_limits[:, 1] - margin
        return rng.uniform(lo, hi)

    def self_collision_count(self, q, penetration_tol: float = 2e-3) -> int:
        """Count genuinely INTERPENETRATING body pairs.

        MuJoCo already excludes parent/child pairs, so a hit here is two separate fingers (or a
        finger and the palm) overlapping. `penetration_tol` matters: a real fist has fingers
        TOUCHING (contact distance ~0), which is not interpenetration. Only overlap deeper than
        the tolerance counts, so a legitimate closed fist is not flagged as broken geometry.
        """
        self._set_q(q)
        self._mj.mj_forward(self.model, self.data)
        return sum(1 for i in range(self.data.ncon)
                   if self.data.contact[i].dist < -abs(penetration_tol))

    def sample_cspace(self, n_samples: int, rng, margin: float = 0.0):
        """Robot C-space samples: (configs (N, n), fingertips (N, n_tips, 3) in palm frame)."""
        qs = np.array([self.random_config(rng, margin) for _ in range(n_samples)])
        tips = np.array([self.fk_fingertips(q) for q in qs])
        return qs, tips

    def canonical_configs(self) -> dict[str, np.ndarray]:
        """The robot side of the calibration: an OPEN and a FIST config.

        Flexion joints run low->high as extended->curled on Allegro, so the lower limit is the
        open hand and the upper limit the fist. These pair with the human's canonical poses to
        pin the curl DIRECTION (see geort.fit_calibration).
        """
        return {"open": self.joint_limits[:, 0].copy(), "fist": self.joint_limits[:, 1].copy()}

    def canonical_fingertips(self) -> dict[str, np.ndarray]:
        return {k: self.fk_fingertips(q) for k, q in self.canonical_configs().items()}


def allegro_hand() -> HandModel:
    """Allegro right hand (16 DoF, 4 fingers) from robot_descriptions."""
    from robot_descriptions import allegro_hand_mj_description

    return HandModel(allegro_hand_mj_description.MJCF_PATH,
                     tip_bodies=ALLEGRO_TIP_BODIES, palm_body="palm")


_HAND_LOADERS = {"allegro": allegro_hand}


def load_hand(name: str) -> HandModel:
    if name not in _HAND_LOADERS:
        raise ValueError(f"unknown hand {name!r}; available: {sorted(_HAND_LOADERS)}")
    return _HAND_LOADERS[name]()
