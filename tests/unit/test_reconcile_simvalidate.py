"""B.0 gates: reconcile must FAIL on an inconsistent trajectory; sim_validate must CATCH a
deliberate joint-limit violation. Both are red->green by construction -- each test feeds the
broken input and asserts the flag, so a checker that waves everything through fails the test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from actuate.retarget import reconcile, sim_validate


# ---------------------------------------------------------------- lightweight stand-ins
@dataclass
class _Arm:
    frame_ids: list
    joint_traj: np.ndarray
    convergence: float = 0.97


@dataclass
class _Fingers:
    frame_ids: list
    finger_traj: np.ndarray


class _Robot:
    """Limits [-1, 1] on every joint; a config counts as colliding iff any |q| > 2."""

    joint_limits = np.array([[-1.0, 1.0]] * 3)

    def within_limits(self, q):
        return bool(np.all(np.abs(np.asarray(q)) <= 1.0))

    def self_collision_count(self, q, penetration_tol=2e-3):
        return int(np.any(np.abs(np.asarray(q)) > 2.0))


def _smooth(n, dof, step=0.01):
    return np.cumsum(np.full((n, dof), step), axis=0)


# ---------------------------------------------------------------- reconcile
def test_reconcile_passes_a_coherent_pair():
    ids = list(range(20))
    r = reconcile.run(_Arm(ids, _smooth(20, 7)), _Fingers(ids, _smooth(20, 16)), "franka_panda")
    assert r.ok and r.arm_teleports == 0 and r.finger_teleports == 0
    assert r.frame_overlap == 1.0


def test_reconcile_fails_on_a_teleporting_trajectory():
    """BROKEN VARIANT (the required gate): shuffled frames = per-frame solver discontinuity."""
    ids = list(range(20))
    traj = _smooth(20, 7) * 30           # big range so shuffling makes real jumps
    rng = np.random.default_rng(0)
    shuffled = traj[rng.permutation(20)]
    r = reconcile.run(_Arm(ids, shuffled), _Fingers(ids, _smooth(20, 16)), "franka_panda")
    assert not r.ok
    assert r.arm_teleports > 0
    assert any("teleport" in reason for reason in r.reasons)


def test_reconcile_flags_poor_frame_overlap():
    a = _Arm(list(range(20)), _smooth(20, 7))
    f = _Fingers(list(range(15, 35)), _smooth(20, 16))    # only 5/20 shared
    r = reconcile.run(a, f, "franka_panda")
    assert not r.ok
    assert any("overlap" in reason for reason in r.reasons)


def test_reconcile_grasp_agreement_is_none_without_canonical():
    ids = list(range(10))
    r = reconcile.run(_Arm(ids, _smooth(10, 7)), _Fingers(ids, _smooth(10, 16)), "franka_panda")
    assert r.grasp_agreement is None     # skipped is reported, never silently passed


# ---------------------------------------------------------------- sim_validate
def test_sim_validate_passes_an_in_limit_trajectory():
    traj = np.linspace(-0.5, 0.5, 30).reshape(-1, 1) * np.ones(3)
    r = sim_validate.run(None, "test", traj, robot_model=_Robot(), ik_convergence=0.97)
    assert r.eligible
    assert r.joint_limit_violations == 0 and r.collision_count == 0


def test_sim_validate_catches_a_deliberate_limit_violation():
    """THE required gate: a trajectory pushed outside the limits must be caught."""
    traj = np.linspace(-0.5, 0.5, 30).reshape(-1, 1) * np.ones(3)
    broken = traj.copy()
    broken[10:13] = 1.5                  # outside [-1, 1]
    r = sim_validate.run(None, "test", broken, robot_model=_Robot())
    assert not r.eligible
    assert r.joint_limit_violations == 3
    assert any("limit" in reason for reason in r.reasons)


def test_sim_validate_catches_collision_frames():
    traj = np.zeros((10, 3))
    traj[4] = 2.5                        # _Robot treats |q|>2 as interpenetration... and >1 as
    r = sim_validate.run(None, "test", traj, robot_model=_Robot())
    assert not r.eligible
    assert r.collision_count == 1        # ...a limit violation too; both must be reported
    assert r.joint_limit_violations == 1


def test_sim_validate_flags_low_ik_convergence():
    traj = np.zeros((10, 3))
    r = sim_validate.run(None, "test", traj, robot_model=_Robot(), ik_convergence=0.84)
    assert not r.eligible
    assert any("IK convergence" in reason for reason in r.reasons)


def test_sim_validate_catches_a_temporal_teleport():
    arm = _Arm(
        frame_ids=[0, 1, 2],
        joint_traj=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.81, 0.0, 0.0]]),
    )
    r = sim_validate.run(None, "test", arm, robot_model=_Robot())
    assert not r.eligible
    assert r.temporal_discontinuities == 1
    assert any("trajectory jump" in reason for reason in r.reasons)


def test_sim_validate_scales_motion_by_sparse_source_frame_gaps():
    arm = _Arm(
        frame_ids=[0, 100, 200],
        joint_traj=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.81, 0.0, 0.0]]),
    )
    r = sim_validate.run(None, "test", arm, robot_model=_Robot())
    assert r.eligible
    assert r.temporal_discontinuities == 0


def test_sim_validate_finger_traj_requires_a_hand_model():
    with pytest.raises(ValueError, match="hand_model"):
        sim_validate.run(None, "test", np.zeros((5, 3)), robot_model=_Robot(),
                         finger_traj=np.zeros((5, 16)))


@pytest.mark.slow
def test_sim_validate_on_the_real_franka_model():
    """Same gates against the actual MuJoCo Franka, not the stand-in."""
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")
    from actuate.retarget.arm.robot import franka_panda

    fr = franka_panda()
    rng = np.random.default_rng(0)
    mid = fr.joint_limits.mean(axis=1)
    traj = mid + 0.05 * rng.standard_normal((10, len(mid)))
    ok = sim_validate.run(None, "franka_panda", traj, robot_model=fr, ik_convergence=0.95)
    assert ok.eligible, ok.summary()

    broken = traj.copy()
    broken[3] = fr.joint_limits[:, 1] + 0.5      # beyond the real upper limits
    bad = sim_validate.run(None, "franka_panda", broken, robot_model=fr)
    assert not bad.eligible
    assert bad.joint_limit_violations >= 1
