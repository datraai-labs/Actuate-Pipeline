"""Part D gates that don't need LeRobot: norm round-trip fails on a corrupted stat
(TRI LBM: non-negotiable), tier filter excludes both ways, dual-space alignment refuses
ambiguity, manifest reports diversity separately."""

from __future__ import annotations

import numpy as np
import pytest

from actuate.config import ConsentStatus, RigType, Side, Tier
from actuate.package.lerobot_export import (
    ExportRefused,
    _assert_export_consent,
    _degenerate_dimensions,
    _episode_tier,
    _robot_action_rows,
    _tier_filter,
)
from actuate.package.manifest import generate
from actuate.package.normalize import (
    PERCENTILES,
    compute_field_stats,
    denormalize_p01_p99,
    normalize_p01_p99,
    verify_round_trip,
)
from actuate.schema import CanonicalEpisode
from actuate.schema.episode import RobotAction
from actuate.schema.frame import SE3, CanonicalFrame, HandState

# ---------------------------------------------------------------- fixtures


def _episode(n=10, tier=None, task="move the pen", **kw):
    frames = tuple(
        CanonicalFrame(
            t=i / 30.0, rig=RigType.HEAD_MOUNTED, episode_id="ep", frame_idx=i,
            hands={Side.RIGHT: HandState(
                wrist_pose=SE3(position_m=(0.1 * i, 0.0, 0.5),
                               quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)))},
            provenance={"hands": "vision_primary"},
        )
        for i in range(n)
    )
    return CanonicalEpisode(episode_id="ep", capture_id="c" * 64, rig=RigType.HEAD_MOUNTED,
                            frames=frames, tier=tier, task=task, **kw)


# ---------------------------------------------------------------- normalize
def test_full_percentile_set_is_computed():
    rng = np.random.default_rng(0)
    s = compute_field_stats(rng.normal(size=(500, 3)))
    for p in PERCENTILES:
        assert getattr(s, f"p{p:02d}") is not None
    # p50 really is the median
    assert s.p50[0] == pytest.approx(0.0, abs=0.15)


def test_round_trip_green_on_honest_stats():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(400, 4))
    verify_round_trip(a, compute_field_stats(a))     # must not raise


def test_round_trip_red_on_a_corrupted_stat():
    """THE gate: a tampered p99 must FAIL loudly. The subtlety is that 'in range' is judged
    by the data's own percentiles -- a check trusting the shipped stat could never fail."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(400, 4))
    s = compute_field_stats(a)
    corrupted = s.model_copy(update={"p99": tuple(v * 0.2 for v in s.p99)})
    with pytest.raises(ValueError, match="round-trip failed"):
        verify_round_trip(a, corrupted)


def test_round_trip_red_on_a_degenerate_stat():
    """p01 == p99 (a collapsed span, e.g. stats computed over one frame) forces the
    degenerate-span fallback and cannot reproduce the data. Note swapped p01/p99 is NOT a
    catchable corruption -- it is a sign-flipped but perfectly invertible linear map."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(300, 2))
    s = compute_field_stats(a)
    degenerate = s.model_copy(update={"p01": s.p99})
    with pytest.raises(ValueError, match="round-trip failed"):
        verify_round_trip(a, degenerate)


def test_outliers_saturate_by_design():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(300, 2))
    s = compute_field_stats(a)
    big = np.array([[1e6, -1e6]])
    n = normalize_p01_p99(big, s)
    assert np.all(np.abs(n) <= 1.0)
    assert not np.allclose(denormalize_p01_p99(n, s), big)   # saturation, not round-trip


# ---------------------------------------------------------------- tier filter
def test_tier_filter_resolves_aliases():
    assert _tier_filter("all") is None
    assert _tier_filter("stage1") == {Tier.STAGE1_VOLUME}
    assert _tier_filter("stage2") == {Tier.STAGE2_ANCHOR}
    assert _tier_filter(Tier.STAGE2_ANCHOR) == {Tier.STAGE2_ANCHOR}
    with pytest.raises(ExportRefused, match="unknown tier"):
        _tier_filter("stage3")


def test_tier_filter_excludes_both_ways():
    """Gate: --tier stage1 excludes stage2 episodes AND vice versa. Synthetic fixtures --
    the real corpus is n=1, so this is unit-only and says so."""
    s1 = _episode(tier=Tier.STAGE1_VOLUME)
    s2 = _episode(tier=Tier.STAGE2_ANCHOR)
    f1, f2 = _tier_filter("stage1"), _tier_filter("stage2")
    assert _episode_tier(s1) in f1 and _episode_tier(s2) not in f1
    assert _episode_tier(s2) in f2 and _episode_tier(s1) not in f2


def test_unassigned_tier_is_volume_never_anchor():
    """stage2 is a CLAIM (matched viewpoint, verified alignment); it must never be a
    default an episode drifts into."""
    assert _episode_tier(_episode(tier=None)) is Tier.STAGE1_VOLUME


# ---------------------------------------------------------------- dual-space alignment
def _with_robot(ep, n_steps, n_joints=7):
    action = RobotAction(embodiment="franka_panda", control_mode="joint",
                         joint_traj=tuple(tuple(0.1 * i for _ in range(n_joints))
                                          for i in range(n_steps)))
    return ep.model_copy(update={
        "action_robot": {"franka_panda": action},
        "retarget_eligibility": {"franka_panda": True},
    })


def test_robot_rows_align_by_full_episode_length():
    ep = _with_robot(_episode(n=10), n_steps=10)
    keep = np.array([0, 2, 4])
    rows = _robot_action_rows(ep, "franka_panda", keep, n_total=10)
    assert rows.shape == (3, 7)
    assert rows[1][0] == pytest.approx(0.2)          # row 2 of the full trajectory


def test_robot_rows_accept_kept_length():
    ep = _with_robot(_episode(n=10), n_steps=3)
    rows = _robot_action_rows(ep, "franka_panda", np.array([0, 2, 4]), n_total=10)
    assert rows.shape == (3, 7)


def test_robot_rows_refuse_ambiguous_length():
    """A misaligned robot action is worse than none -- refuse, don't guess."""
    ep = _with_robot(_episode(n=10), n_steps=6)
    with pytest.raises(ExportRefused, match="frame accounting"):
        _robot_action_rows(ep, "franka_panda", np.array([0, 2, 4]), n_total=10)


def test_robot_rows_refuse_a_missing_embodiment():
    with pytest.raises(ExportRefused, match="no action_robot"):
        _robot_action_rows(_episode(), "franka_panda", np.array([0]), n_total=10)


def test_robot_rows_refuse_a_failed_physics_verdict():
    ep = _with_robot(_episode(n=10), n_steps=10).model_copy(
        update={"retarget_eligibility": {"franka_panda": False}}
    )
    with pytest.raises(ExportRefused, match="ineligible"):
        _robot_action_rows(ep, "franka_panda", np.array([0, 1]), n_total=10)


def test_constant_grasp_blocks_export():
    names = ["x", "grasp"]
    state = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    action = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ExportRefused, match="grasp.*zero variance"):
        _degenerate_dimensions("ep", state, action, np.array([0, 1, 2]), names)


def test_local_export_rechecks_consent_and_pii():
    with pytest.raises(ExportRefused, match="local export blocked"):
        _assert_export_consent(_episode())


# ---------------------------------------------------------------- manifest
def test_manifest_reports_diversity_axes_separately():
    from actuate.schema.episode import Diversity

    eps = [
        _episode().model_copy(update={
            "diversity": Diversity(scene_id="desk", demonstrator_id=f"d{i}")})
        for i in range(3)
    ]
    m = generate(eps)
    assert m.scene_count == 1                # 1 scene ...
    assert m.demonstrator_count == 3         # ... 3 demonstrators: NOT one number
    assert m.episode_count == 3


def test_manifest_is_honest_about_the_uncomfortable():
    ep = _episode(task=None, consent=ConsentStatus.PENDING)
    m = generate([ep])
    assert m.task_distribution == {"<untasked>": 1}
    assert m.tier_distribution == {"<unassigned>": 1}
    assert m.episodes_with_unknown_scene == 1
    assert m.mean_certificate["contact_consistency"] is None   # never measured stays None
