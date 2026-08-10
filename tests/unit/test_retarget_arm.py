"""L5 arm retargeting: VN equivariance, sim-data validity, IK, and the run pipeline.

The load-bearing correctness test is `test_vn_layers_are_exactly_equivariant` -- the whole
"SE(3)-equivariant" claim rests on it, so it is checked to numerical zero. The robot-dependent
tests need MuJoCo + robot_descriptions (they download/cache the Franka model); they skip cleanly
where those aren't available. Full estimator convergence (>90%, gate 2) is a Kaggle training job,
not a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------------------
# Pure pieces (torch / numpy only)
# --------------------------------------------------------------------------------------


def test_vn_layers_are_exactly_equivariant():
    torch = pytest.importorskip("torch")
    from scipy.spatial.transform import Rotation

    from actuate.retarget.arm.estimator import RootFrameEstimator

    est = RootFrameEstimator(hidden=32)
    torch.manual_seed(0)
    B, H = 4, 32
    x = torch.randn(B, 3, 3)
    cond = torch.randn(B, H, 3)
    t = torch.rand(B)
    feats = torch.randn(B, 10, 4, 3)
    Rm = torch.tensor(Rotation.from_euler("xyz", [0.3, -0.5, 0.8]).as_matrix(), dtype=torch.float32)

    def rot(z):
        return torch.einsum("ij,...j->...i", Rm, z)

    with torch.no_grad():
        v = est.net.velocity(x, cond, t)
        v_rot = est.net.velocity(rot(x), rot(cond), t)
        enc = est.net.encode(feats)
        enc_rot = est.net.encode(rot(feats))

    # v(Rx, Rcond) == R v(x, cond)  and  enc(Rf) == R enc(f)
    assert (v_rot - rot(v)).abs().max().item() < 1e-4
    assert (enc_rot - rot(enc)).abs().max().item() < 1e-4


def test_rot6d_roundtrip_recovers_the_rotation():
    from scipy.spatial.transform import Rotation

    from actuate.retarget.arm.simdata import _rot6d, rot6d_to_matrix

    rng = np.random.default_rng(0)
    for _ in range(20):
        R = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
        r6 = np.concatenate([R[:, 0], R[:, 1]])
        Rr = rot6d_to_matrix(r6)
        assert np.allclose(Rr, R, atol=1e-5)
        # and _rot6d(Rotation) matches
        assert np.allclose(_rot6d(Rotation.from_matrix(R)), r6, atol=1e-6)


# --------------------------------------------------------------------------------------
# Robot-dependent (MuJoCo + robot_descriptions)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def franka():
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")
    from actuate.retarget.arm.robot import franka_panda

    try:
        return franka_panda()
    except Exception as exc:  # model download failed (offline CI)
        pytest.skip(f"could not load Franka model: {exc}")


def test_franka_has_7_arm_dof_with_limits(franka):
    assert franka.n == 7
    assert franka.joint_limits.shape == (7, 2)
    assert np.all(franka.joint_limits[:, 0] < franka.joint_limits[:, 1])


def test_ik_solves_reachable_and_fails_unreachable(franka):
    rng = np.random.default_rng(0)
    # reachable: FK a real config, IK back
    q_true = franka.random_config(rng, margin=0.2)
    target = franka.fk(q_true)
    res = franka.ik(target, rng=rng)
    assert res.converged and res.pos_err_mm < 2.0

    # broken variant: a target 10 m away is unreachable -> IK must NOT report convergence
    far = target.model_copy(update={"position_m": (10.0, 10.0, 10.0)})
    res_bad = franka.ik(far, restarts=3, rng=rng)
    assert not res_bad.converged


def test_gate1_sim_pairs_are_valid(franka):
    from actuate.retarget.arm import simdata

    pairs = simdata.generate_dataset(franka, n_pairs=15, length=16, seed=0)
    rep = simdata.validate_pairs(franka, pairs)
    assert rep["all_valid"]                          # GATE 1
    assert rep["joint_within_limits_pct"] == 100.0
    # gravity is a unit vector in the camera frame
    assert abs(np.linalg.norm(pairs[0].gravity_cam) - 1.0) < 1e-5


def test_run_pipeline_produces_a_joint_trajectory_and_candidate_spread(franka):
    pytest.importorskip("torch")
    from actuate.config import Provenance, RigType, Side
    from actuate.retarget.arm import RootFrameEstimator, run, simdata
    from actuate.schema import (
        SE3,
        CanonicalEpisode,
        CanonicalFrame,
        HandState,
        ImageRef,
    )

    # tiny estimator (mechanics only; convergence quality is a Kaggle concern)
    est = RootFrameEstimator(hidden=32)
    est.train(simdata.generate_dataset(franka, 120, length=10, seed=0), epochs=60, log_every=0)

    from scipy.spatial.transform import Rotation

    frames = []
    for i in range(10):
        p = (0.3 + 0.02 * i, -0.01 * i, 0.6)
        q = Rotation.from_euler("xyz", [0.02 * i, 0.1, 0.0]).as_quat()
        hs = HandState(wrist_pose=SE3(position_m=p, quaternion_wxyz=(q[3], q[0], q[1], q[2])))
        frames.append(CanonicalFrame(
            t=i / 30, rig=RigType.HEAD_MOUNTED, episode_id="syn", frame_idx=i,
            images={"head": ImageRef(uri="x", frame_index=i)}, hands={Side.RIGHT: hs},
            provenance={"hands": Provenance.VISION_PRIMARY}))
    ep = CanonicalEpisode(episode_id="syn", capture_id="a" * 64, source_content_hash="a" * 64,
                          rig=RigType.HEAD_MOUNTED, frames=tuple(frames), task="t")

    res = run(ep, "franka_panda", est, n_candidates=6, cluster_k=3)
    assert res.joint_traj.shape == (10, 7)
    assert len(res.action.joint_traj) == 10 and len(res.action.ee_traj) == 10
    assert 0.0 <= res.convergence <= 1.0
    assert res.candidate_spread_m > 0.0             # GATE 4: distinct hypotheses
