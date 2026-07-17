"""Part E gates: aligned+matching -> stage2_anchor; mismatched -> FLAGGED not silently
accepted; no claim -> stage1_volume. Plus the case the real corpus is actually in:
an aligned claim that CANNOT be verified must flag, not pass."""

from __future__ import annotations

import json

import pytest

from actuate.config import Tier
from actuate.config.embodiments import EmbodimentSpec, HandSpec
from actuate.ingest import run_ingest
from actuate.ingest.run import _check_alignment, _session_intrinsics


@pytest.fixture()
def session(tmp_path):
    (tmp_path / "compressed.mp4").write_bytes(b"\x00" * 2048)   # content, not codec, matters
    (tmp_path / "session_meta.json").write_text(json.dumps(
        {"frame_count": 60, "duration_seconds": 2.0, "fps_nominal": 30.0}))
    return tmp_path


def _spec(intrinsics):
    return EmbodimentSpec(
        name="calibrated_bot", description="test", arm_dof=7,
        hand=HandSpec(name="g", dof=1, n_fingers=2), cameras=("wrist",),
        camera_intrinsics=intrinsics)


def _register(monkeypatch, spec):
    import actuate.ingest.run as ingest_run
    monkeypatch.setattr(ingest_run, "get_embodiment", lambda name: spec)


# ---------------------------------------------------------------- gate 3: no claim
def test_no_aligned_claim_defaults_to_stage1(session):
    res = run_ingest("head_mounted", session)
    assert res.tier is Tier.STAGE1_VOLUME
    assert res.intrinsics_match is None and res.aligned_robot is None
    assert len(res.capture_id) == 64
    assert res.manifest_path.exists()


# ---------------------------------------------------------------- gate 1: verified match
def test_matching_intrinsics_earn_stage2_anchor(session, monkeypatch):
    (session / "camera_intrinsics.json").write_text(json.dumps({"fx": 610.0, "fy": 612.0}))
    _register(monkeypatch, _spec((600.0, 600.0)))          # within 5%
    res = run_ingest("head_mounted", session, aligned_robot="calibrated_bot")
    assert res.tier is Tier.STAGE2_ANCHOR
    assert res.intrinsics_match is True
    assert not res.flags


# ---------------------------------------------------------------- gate 2: mismatch FLAGS
def test_mismatched_intrinsics_flag_and_stay_stage1(session, monkeypatch):
    """THE gate: a mismatched claim is never silently accepted as aligned."""
    (session / "camera_intrinsics.json").write_text(json.dumps({"fx": 1104.0, "fy": 1104.0}))
    _register(monkeypatch, _spec((660.0, 660.0)))           # the real fx bug, as a fixture
    res = run_ingest("head_mounted", session, aligned_robot="calibrated_bot")
    assert res.tier is Tier.STAGE1_VOLUME
    assert res.intrinsics_match is False
    assert any("MISMATCH" in f for f in res.flags)


# ---------------------------------------------------------------- the real corpus's case
def test_unverifiable_claim_flags_never_passes(session):
    """No calibration on either side -> UNVERIFIABLE, flagged, stage1. This is the state
    of every current embodiment (camera_intrinsics=None), so it is the case that will
    actually fire in production first."""
    res = run_ingest("head_mounted", session, aligned_robot="franka_panda")
    assert res.tier is Tier.STAGE1_VOLUME
    assert res.intrinsics_match is None
    assert any("UNVERIFIABLE" in f for f in res.flags)


def test_unknown_robot_raises_not_flags(session):
    with pytest.raises(KeyError):
        run_ingest("head_mounted", session, aligned_robot="not_a_robot")


# ---------------------------------------------------------------- helpers
def test_session_intrinsics_reads_fx_fy(tmp_path):
    (tmp_path / "camera_intrinsics.json").write_text(json.dumps({"fx": 660.0, "fy": 662.0}))
    assert _session_intrinsics(tmp_path) == (660.0, 662.0)
    assert _session_intrinsics(tmp_path / "nowhere") is None


def test_check_alignment_tolerance_boundary(tmp_path):
    (tmp_path / "camera_intrinsics.json").write_text(json.dumps({"fx": 630.0, "fy": 630.0}))
    flags: list[str] = []
    import actuate.ingest.run as ingest_run
    orig = ingest_run.get_embodiment
    ingest_run.get_embodiment = lambda name: _spec((600.0, 600.0))
    try:
        tier, ok = _check_alignment(tmp_path, "x", flags)    # 5.0% -> exactly at rtol
        assert tier is Tier.STAGE2_ANCHOR and ok
        (tmp_path / "camera_intrinsics.json").write_text(
            json.dumps({"fx": 631.0, "fy": 631.0}))          # just past it
        tier, ok = _check_alignment(tmp_path, "x", flags)
        assert tier is Tier.STAGE1_VOLUME and ok is False
    finally:
        ingest_run.get_embodiment = orig
