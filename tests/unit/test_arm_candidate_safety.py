from types import SimpleNamespace

import numpy as np

from actuate.retarget.arm.candidates import CandidateScore, score_candidate


def _score(*, convergence: float, collisions: int) -> CandidateScore:
    return CandidateScore(
        root_pos=np.zeros(3),
        root_R=np.eye(3),
        convergence=convergence,
        residual_mm=0.1,
        manipulability=1.0,
        joint_margin=0.5,
        smoothness=0.01,
        collision_frames=collisions,
        joint_traj=np.zeros((2, 2)),
        ee_traj=[],
    )


def test_collision_free_candidate_ranks_above_colliding_candidate():
    assert _score(convergence=0.9, collisions=0).rank_key() > _score(
        convergence=1.0, collisions=1
    ).rank_key()


class _SeededRobot:
    joint_limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    def ik(self, target, q0=None, restarts=2, rng=None):
        q = rng.uniform(-0.2, 0.2, 2)
        return SimpleNamespace(q=q, converged=True, pos_err_mm=0.1)

    def fk(self, q):
        return tuple(q)

    def manipulability(self, q):
        return 1.0

    def self_collision_count(self, q):
        return 0


def test_candidate_ik_restarts_are_deterministic():
    wrist_pos = np.array([[0.1, 0.0, 0.2], [0.2, 0.0, 0.2]])
    wrist_rot6d = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (2, 1))
    args = (_SeededRobot(), np.zeros(3), np.eye(3), wrist_pos, wrist_rot6d)
    a = score_candidate(*args, seed=7)
    b = score_candidate(*args, seed=7)
    np.testing.assert_allclose(a.joint_traj, b.joint_traj)
