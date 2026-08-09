"""Part B gates: components computed from real sources; None = not measured, never zero;
quality never opens the consent gate (the gate 4 red->green)."""

from __future__ import annotations

import json

import pytest

from actuate.certify import score as certify_score
from actuate.certify.score import (
    CertificateComponents,
    composite_quality,
    contact_consistency,
    find_mistakes,
    perception_confidence,
    speed_bin,
    sync_integrity,
)
from actuate.config import ConsentStatus, RigType
from actuate.io.consent import ConsentViolation, check_episode_deliverable
from actuate.schema import CanonicalEpisode
from actuate.schema.frame import CanonicalFrame


def _frame(i, conf, fps=30.0, **kw):
    return CanonicalFrame(t=i / fps, rig=RigType.HEAD_MOUNTED, episode_id="ep",
                          frame_idx=i, confidence=conf, **kw)


def _episode(frames, **kw):
    return CanonicalEpisode(episode_id="ep", capture_id="c" * 64, rig=RigType.HEAD_MOUNTED,
                            frames=tuple(frames), **kw)


# ------------------------------------------------------------------ components
def test_sync_integrity_reads_the_l0_report(tmp_path):
    (tmp_path / "quality_certificate.json").write_text(json.dumps({
        "episodes": [{"components": {"sync_drift": {"max_drift_ms": 3.9447}}}]}))
    (tmp_path / "session_meta.json").write_text(json.dumps({"fps_nominal": 30.0}))
    s = sync_integrity(tmp_path)
    assert s == pytest.approx(1 - 3.9447 / (1000 / 30), abs=1e-6)


def test_sync_integrity_is_none_without_a_report(tmp_path):
    assert sync_integrity(None) is None
    assert sync_integrity(tmp_path) is None      # empty dir: not measured, not 0.0


def test_sync_integrity_reads_modern_frame_aligned_imu(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "session.h5", "w") as h5:
        imu = h5.create_group("imu")
        imu.create_dataset("timestamp_ns", data=[1_002_000_000, 1_035_000_000])
        imu.create_dataset("video_timestamp_ns", data=[1_000_000_000, 1_033_000_000])
    (tmp_path / "session_meta.json").write_text(json.dumps({"fps_nominal": 30.0}))

    assert sync_integrity(tmp_path) == pytest.approx(1 - 2.0 / (1000 / 30))


def test_contact_consistency_is_none_on_a_contactless_rig():
    """head_mounted measures nothing. None (not measured) -- 0.0 would claim 'measured,
    catastrophic' about a channel with no sensor."""
    ep = _episode([_frame(0, {"hands": 0.9})])
    assert contact_consistency(ep) is None


def test_perception_confidence_aggregates_frame_confidence():
    ep = _episode([_frame(0, {"hands_calibrated": 0.8, "depth_calibrated": 0.4}),
                   _frame(1, {"hands_calibrated": 0.6})])
    assert perception_confidence(ep) == pytest.approx((0.6 + 0.6) / 2)


def test_uncalibrated_model_scores_do_not_become_certificate_confidence():
    ep = _episode([_frame(0, {"hands": 0.99, "depth": 0.95, "grasp": 0.5})])
    assert perception_confidence(ep) is None


def test_speed_uses_timestamps_not_frame_count():
    """A 45-frame subsample of a 95 s episode is SLOW, not fast -- the raw-frame-count shim
    this replaces got exactly this wrong."""
    ep = _episode([_frame(i, {}, fps=45 / 95.0) for i in range(45)])   # t spans ~95 s
    assert speed_bin(ep) == 3
    fast = _episode([_frame(i, {}) for i in range(45)])                # 45 frames at 30 fps
    assert speed_bin(fast) == 1


def test_mistakes_flag_low_confidence_segments_and_not_good_ones():
    """BROKEN-VARIANT CHECK built in: confident frames must NOT be flagged, so a scorer
    that flags everything (or nothing) fails this test."""
    good = [_frame(i, {"hands_calibrated": 0.9}) for i in range(30)]   # 0-1 s: fine
    bad = [_frame(30 + i, {"hands_calibrated": 0.1}) for i in range(30)]
    flags = find_mistakes(_episode(good + bad))
    assert len(flags) == 1
    assert "low_confidence@1s-2s" in flags[0]
    assert find_mistakes(_episode(good)) == ()


# ------------------------------------------------------------------ composite
def test_composite_renormalises_over_measured_components():
    """An unmeasured channel must neither help nor hurt."""
    partial = CertificateComponents(perception_confidence=0.5, calibration_completeness=0.5)
    full_at_same_level = CertificateComponents(
        sync_integrity=0.5, calibration_completeness=0.5, perception_confidence=0.5,
        contact_consistency=0.5, ik_convergence_rate=0.5)
    q_partial, notes = composite_quality(partial)
    q_full, _ = composite_quality(full_at_same_level)
    assert q_partial == q_full == 3
    assert any("not measured" in n for n in notes)


def test_composite_floors_at_1_when_nothing_is_measured():
    q, notes = composite_quality(CertificateComponents())
    assert q == 1 and notes


# ------------------------------------------------------------------ the gate that matters
def test_gate4_consent_still_blocks_at_quality_5():
    """quality != consent. A perfect score with consent=pending must still refuse delivery.
    This is the red->green: remove the consent check and this leaks."""
    frames = [_frame(i, {"hands": 1.0}) for i in range(30)]
    ep = _episode(frames, consent=ConsentStatus.PENDING)
    report = certify_score(ep, session_dir=None, intrinsics_measured=True)
    boosted = report.episode.model_copy(update={
        "episode_meta": report.episode.episode_meta.model_copy(update={"quality": 5})})
    assert boosted.episode_meta.quality == 5
    with pytest.raises(ConsentViolation):
        check_episode_deliverable(boosted)


def test_score_attaches_l5_results_when_given():
    class _Sim:
        eligible = False
        ik_convergence_rate = 0.84
        reasons = ("IK convergence 84% < 90%",)

    class _Rec:
        ok = False
        reasons = ("3 arm teleport(s)",)

    ep = _episode([_frame(i, {"hands": 0.8}) for i in range(10)])
    r = certify_score(ep, "franka_panda", sim_result=_Sim(), reconcile_result=_Rec())
    assert r.retarget_eligibility == {"franka_panda": False}
    assert not r.strategy_alignment["franka_panda"].ok
    assert "teleport" in r.strategy_alignment["franka_panda"].reason
    assert r.components.ik_convergence_rate == pytest.approx(0.84)
    assert any("sim_validate" in m for m in r.mistakes)


def test_score_without_l5_reports_not_measured():
    ep = _episode([_frame(0, {"hands": 0.8})])
    r = certify_score(ep)
    assert r.components.ik_convergence_rate is None
    assert r.retarget_eligibility == {} and r.strategy_alignment == {}
