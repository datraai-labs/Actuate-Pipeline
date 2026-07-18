"""Regression tests for the two customer-facing bugs on a FRESH source (no bundled v1
artifacts): the canonical stage must give an actionable perception-failed error instead of
a cryptic metric-depth crash, and RLDS export must explain the protobuf constraint rather
than blow up on a tfds import error."""

from __future__ import annotations

import pytest


def test_canonical_stage_surfaces_perception_failure_on_fresh_source(tmp_path):
    """Perception failed + no v1 hand_pose_3d.json to fall back on -> a clear RuntimeError
    that names the perceive reason, NOT a downstream CanonicalBuildError about metric depth
    (the confusing error the customer hit)."""
    from actuate.pipeline.run import _Ctx, _stage_canonical

    session = tmp_path / "sess"
    session.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    ctx = _Ctx(session=session, out=out, profile={"rig": "head_mounted"},
               reporter=lambda *a: None, confirm=lambda p: False,
               checkpoint={"perceive": {"status": "skipped",
                                        "note": "GPU stages failed: ImportError wandb"}})
    ctx.canonical_path = out / "canonical.json"

    with pytest.raises(RuntimeError) as exc:
        _stage_canonical(ctx, {})                     # empty perception dict
    msg = str(exc.value)
    assert "perception stage did not produce hands" in msg
    assert "ImportError wandb" in msg                 # the ACTUAL reason is surfaced
    assert "metric" not in msg.lower()                # not the cryptic depth error


def test_canonical_still_uses_v1_fallback_when_artifacts_exist(tmp_path, monkeypatch):
    """The v1-legacy path is only taken when the session actually carries v1 artifacts
    (the bundled demo) -- a fresh source never guesses it."""
    from actuate.pipeline import run as run_mod

    session = tmp_path / "sess"
    session.mkdir()
    (session / "hand_pose_3d.json").write_text("[]")   # pretend v1 artifacts present
    out = tmp_path / "out"
    out.mkdir()
    ctx = run_mod._Ctx(session=session, out=out, profile={"rig": "head_mounted"},
                       reporter=lambda *a: None, confirm=lambda p: False, checkpoint={})
    ctx.canonical_path = out / "canonical.json"

    called = {}

    def _fake_build_episode(sess, cid, task=None):
        called["v1"] = True
        raise RuntimeError("v1 build reached")         # we only assert the PATH was chosen

    import actuate.canonical

    monkeypatch.setattr(actuate.canonical, "build_episode", _fake_build_episode)
    with pytest.raises(RuntimeError, match="v1 build reached"):
        _stage = run_mod._stage_canonical
        _stage(ctx, {})
    assert called.get("v1") is True


def test_rlds_export_explains_protobuf_when_tfds_missing(monkeypatch, tmp_path):
    """When tensorflow-datasets can't import (protobuf<6 in [all]), RLDS export must raise a
    clear ExportRefused pointing to the separate-env fix -- not a raw ImportError."""
    import builtins

    from actuate.package.rlds_export import ExportRefused, export_rlds

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "tensorflow_datasets":
            raise ImportError("cannot import name 'runtime_version' from google.protobuf")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ExportRefused) as exc:
        export_rlds([], tmp_path)
    msg = str(exc.value)
    assert "protobuf" in msg and "separate env" in msg
    assert "LeRobot v3 export works here" in msg       # tells the user what DOES work
