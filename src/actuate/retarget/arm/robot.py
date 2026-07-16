"""Robot kinematics + IK for arm retargeting, on MuJoCo (Master Spec §L5).

### Why MuJoCo instead of Pinocchio

The spec suggests starting with Pinocchio. Pinocchio has **no Windows wheels** and fails to
build from source on this dev box (`pip install pin` -> build error), so it is not an option
here. MuJoCo installs cleanly on Windows *and* Linux/Kaggle, gives forward kinematics, the
geometric Jacobian (`mj_jac`), joint limits, and -- for gate 3 -- physics replay and collision,
all from ONE dependency that actually works. So this uses MuJoCo for kinematics + IK; the
interface (`fk`, `jacobian`, `ik`) is solver-agnostic, so a Pinocchio backend can slot in later
on Linux without touching the estimator or the retarget pipeline.

IK is damped least squares (Levenberg-Marquardt) on the 6-DoF end-effector error, clamped to
joint limits -- standard, deterministic, and enough for per-episode retargeting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from actuate.schema import SE3


@dataclass
class IKResult:
    q: np.ndarray            # joint config (arm DoF)
    converged: bool
    pos_err_mm: float        # final position error, millimetres
    rot_err_deg: float       # final orientation error, degrees
    iters: int


class RobotModel:
    """A MuJoCo robot exposing FK / Jacobian / IK over its ARM joints.

    `arm_joint_names` selects the revolute arm DoF (excludes the gripper). `ee_body` is the
    frame IK targets (the wrist/flange -- for Franka, "hand").
    """

    def __init__(self, mjcf_path: str, ee_body: str, arm_joint_names: list[str]) -> None:
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_body_id < 0:
            raise ValueError(f"ee body {ee_body!r} not found")

        self.arm_joint_ids = []
        self.arm_qpos_adr = []
        self.arm_dof_adr = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} not found")
            self.arm_joint_ids.append(jid)
            self.arm_qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.arm_dof_adr.append(int(self.model.jnt_dofadr[jid]))
        self.n = len(arm_joint_names)
        self.joint_limits = np.array(
            [self.model.jnt_range[j] for j in self.arm_joint_ids], dtype=np.float64
        )  # (n, 2)

    # -- FK -----------------------------------------------------------------------------
    def _set_q(self, q: np.ndarray) -> None:
        for adr, v in zip(self.arm_qpos_adr, q):
            self.data.qpos[adr] = v
        self._mj.mj_forward(self.model, self.data)

    def fk(self, q: np.ndarray) -> SE3:
        """End-effector pose (world frame) for arm config q."""
        self._set_q(np.asarray(q, dtype=np.float64))
        pos = self.data.xpos[self.ee_body_id].copy()
        quat = self.data.xquat[self.ee_body_id].copy()  # MuJoCo: (w, x, y, z)
        return SE3(position_m=tuple(float(x) for x in pos),
                   quaternion_wxyz=tuple(float(x) for x in quat))

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6xN geometric Jacobian (linear; angular) at the EE, over the arm DoF."""
        self._set_q(np.asarray(q, dtype=np.float64))
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mj.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        cols = self.arm_dof_adr
        return np.vstack([jacp[:, cols], jacr[:, cols]])  # (6, n)

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.joint_limits[:, 0] + margin)
                    and np.all(q <= self.joint_limits[:, 1] - margin))

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def random_config(self, rng: np.random.Generator, margin: float = 0.1) -> np.ndarray:
        lo = self.joint_limits[:, 0] + margin
        hi = self.joint_limits[:, 1] - margin
        return rng.uniform(lo, hi)

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa manipulability sqrt(det(J J^T)) -- 0 at a singularity."""
        J = self.jacobian(q)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    # -- IK (damped least squares) ------------------------------------------------------
    def _ik_once(self, target, q0, R_t, t_pos, pos_tol_m, rot_tol_rad, max_iters, damping, step):
        from scipy.spatial.transform import Rotation

        q = self.clamp(np.asarray(q0, dtype=np.float64))
        pos_err = rot_err = np.inf
        prev = np.inf
        it = 0
        for it in range(1, max_iters + 1):
            cur = self.fk(q)
            cw, cx, cy, cz = cur.quaternion_wxyz
            e_pos = t_pos - np.asarray(cur.position_m)
            e_rot = (R_t * Rotation.from_quat([cx, cy, cz, cw]).inv()).as_rotvec()
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            if pos_err < pos_tol_m and rot_err < rot_tol_rad:
                break
            total = pos_err + rot_err
            # adaptive damping: if we stalled/worsened, damp harder (more conservative step)
            damp = damping * (4.0 if total > prev else 1.0)
            prev = total
            err = np.concatenate([e_pos, e_rot])
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + (damp ** 2) * np.eye(6), err)
            q = self.clamp(q + step * dq)
        return q, pos_err, rot_err, it

    def ik(
        self, target: SE3, q0: np.ndarray | None = None, *,
        pos_tol_m: float = 1e-3, rot_tol_rad: float = 1e-2,
        max_iters: int = 120, damping: float = 0.05, step: float = 0.5,
        restarts: int = 8, rng: np.random.Generator | None = None,
    ) -> IKResult:
        """Solve arm config placing the EE at `target`. Damped least squares with restarts.

        Tries `q0` (warm start) first, then random restarts until one converges -- IK from a
        far cold start can stall at a joint limit, so restarts turn ~50% single-shot into
        near-100%. In the retarget pipeline, each trajectory frame warm-starts from the previous
        frame's solution, so a single shot almost always converges there.
        """
        from scipy.spatial.transform import Rotation

        rng = rng or np.random.default_rng()
        t_pos = np.asarray(target.position_m, dtype=np.float64)
        w, x, y, z = target.quaternion_wxyz
        R_t = Rotation.from_quat([x, y, z, w])

        best = None
        starts = ([q0] if q0 is not None else []) + [
            self.random_config(rng, margin=0.2) for _ in range(restarts)
        ]
        for q0i in starts:
            q, pe, re, it = self._ik_once(target, q0i, R_t, t_pos, pos_tol_m, rot_tol_rad,
                                          max_iters, damping, step)
            conv = pe < pos_tol_m and re < rot_tol_rad
            if best is None or (pe + re) < best[1] + best[2]:
                best = (q, pe, re, it, conv)
            if conv:
                break
        q, pe, re, it, conv = best
        return IKResult(q=q, converged=conv, pos_err_mm=pe * 1000,
                        rot_err_deg=np.degrees(re), iters=it)


#: Franka Emika Panda arm joints (7-DoF), from the MuJoCo Menagerie MJCF.
_FRANKA_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


#: EE body + arm joints per embodiment, keyed by registry name. Only kinematic wiring lives
#: here; everything else (DoF, hand, cameras) is in the embodiment registry.
_ARM_WIRING = {
    "franka_panda": ("hand", _FRANKA_ARM_JOINTS),
    "franka_dual": ("hand", _FRANKA_ARM_JOINTS),
}


def _resolve_mjcf(urdf_path: str) -> str:
    """Map an embodiment `urdf_path` to a MuJoCo MJCF file path.

    Supports "robot_descriptions:<module>" (resolves to that description's downloaded MJCF) and
    a plain filesystem path. The scheme keeps machine-specific cache paths out of the registry.
    """
    if urdf_path.startswith("robot_descriptions:"):
        import importlib

        mod = importlib.import_module(f"robot_descriptions.{urdf_path.split(':', 1)[1]}")
        return mod.MJCF_PATH
    return urdf_path


def load_robot(embodiment) -> RobotModel:
    """Build a RobotModel for an embodiment (name or EmbodimentSpec)."""
    from actuate.config.embodiments import get_embodiment

    spec = get_embodiment(embodiment) if isinstance(embodiment, str) else embodiment
    if spec.urdf_path is None:
        raise ValueError(f"embodiment {spec.name!r} has no kinematic model (urdf_path=None)")
    if spec.name not in _ARM_WIRING:
        raise ValueError(f"no arm wiring registered for {spec.name!r}")
    ee_body, arm_joints = _ARM_WIRING[spec.name]
    return RobotModel(_resolve_mjcf(spec.urdf_path), ee_body=ee_body, arm_joint_names=arm_joints)


def franka_panda() -> RobotModel:
    """Load Franka Panda from robot_descriptions (downloads the MJCF once, then caches)."""
    return load_robot("franka_panda")
