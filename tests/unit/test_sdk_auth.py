"""Phase 6 Parts A+C: auth config (local/cloud, key masking) and the SDK plumbing
(source resolution, format aliases, ProcessingRun accessors) -- all GPU-free."""

from __future__ import annotations

import json

import pytest

from actuate.config import auth


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTUATE_HOME", str(tmp_path / ".actuate"))


# ---------------------------------------------------------------- auth (Part C)
def test_local_login_needs_no_key():
    cfg = auth.login(mode="local")
    assert cfg["mode"] == "local" and cfg["api_key"] is None
    assert auth.is_authenticated()


def test_cloud_login_stores_key_but_redaction_hides_it():
    cfg = auth.login(mode="cloud", api_key="ak_secret_value")
    assert cfg["api_key"] == "ak_secret_value"          # on disk
    assert auth.redacted(cfg)["api_key"] == "set (hidden)"   # never displayed
    assert "ak_secret_value" not in json.dumps(auth.redacted(cfg))


def test_cloud_login_requires_a_key():
    with pytest.raises(ValueError, match="needs an api_key"):
        auth.login(mode="cloud")


def test_require_cloud_is_a_clear_coming_soon():
    with pytest.raises(NotImplementedError, match="coming soon"):
        auth.require_cloud()


def test_set_default_persists_and_rejects_unknown_keys():
    auth.login(mode="local")
    auth.set_default("default_embodiment", "aloha_v2")
    assert auth.load_config()["default_embodiment"] == "aloha_v2"
    with pytest.raises(ValueError, match="cannot set"):
        auth.set_default("api_key", "x")                # not settable via config set


def test_defaults_appear_even_on_an_old_file(tmp_path, monkeypatch):
    p = auth.config_path()
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"mode": "local"}))          # older, sparse file
    cfg = auth.load_config()
    assert cfg["default_export_format"] == "lerobot_v3"  # merged from DEFAULT_CONFIG


# ---------------------------------------------------------------- SDK plumbing (Part A)
def test_login_helper_local_and_cloud():
    from actuate import sdk

    assert sdk.login()["mode"] == "local"
    assert sdk.login(api_key="ak_x")["mode"] == "cloud"


def test_resolve_local_video_stages_a_session(tmp_path):
    from actuate import sdk

    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00" * 32)
    session = sdk._resolve_source(str(vid), tmp_path / "work")
    assert session.is_dir()
    assert (session / "clip.mp4").exists()               # staged in


def test_resolve_existing_dir_passthrough(tmp_path):
    from actuate import sdk

    d = tmp_path / "sess"
    d.mkdir()
    assert sdk._resolve_source(str(d), tmp_path / "work") == d


def test_resolve_missing_source_errors(tmp_path):
    from actuate import sdk

    with pytest.raises(FileNotFoundError, match="source not found"):
        sdk._resolve_source(str(tmp_path / "nope.mp4"), tmp_path)


def test_resolve_remote_scheme_dispatches_to_sources(tmp_path, monkeypatch):
    """Remote schemes now route through actuate.sources (Part F). Stub the resolver so no
    network is touched -- we only assert the dispatch happens."""
    from actuate import sdk, sources

    monkeypatch.setattr(sources, "resolve", lambda src, wr, **k: tmp_path / "staged")
    assert sdk._resolve_source("hf://foo/bar", tmp_path) == tmp_path / "staged"


def test_resolve_unknown_scheme_errors(tmp_path):
    from actuate import sdk

    with pytest.raises(ValueError, match="unrecognised source scheme"):
        sdk._resolve_source("ftp://foo/bar", tmp_path)


def test_export_format_aliases():
    from actuate import sdk

    assert sdk._FORMATS["lerobot_v3"] == "lerobot"
    assert sdk._FORMATS["openx"] == "rlds"


def test_processing_run_reads_off_the_canonical(tmp_path):
    """ProcessingRun.summary/certificate read the canonical the pipeline wrote -- build a
    tiny real one and a fake PipelineResult, no GPU."""
    from actuate.config import RigType
    from actuate.pipeline.run import PipelineResult
    from actuate.schema import CanonicalEpisode
    from actuate.schema.episode import EpisodeMeta
    from actuate.schema.frame import CanonicalFrame
    from actuate.sdk import ProcessingRun

    frames = tuple(CanonicalFrame(t=i / 30.0, rig=RigType.HEAD_MOUNTED, episode_id="ep",
                                  frame_idx=i, confidence={"hands": 0.8},
                                  provenance={"hands": "vision_primary"}) for i in range(5))
    ep = CanonicalEpisode(episode_id="ep", capture_id="c" * 64, rig=RigType.HEAD_MOUNTED,
                          frames=frames, task="pick up cup",
                          episode_meta=EpisodeMeta(quality=3, speed=1))
    canon = tmp_path / "canonical.json"
    canon.write_text(ep.model_dump_json())
    result = PipelineResult(out=tmp_path, canonical_path=canon,
                            checkpoint={"canonical": {"status": "done"},
                                        "package": {"status": "done"}}, seconds=1.0)
    run = ProcessingRun(session=tmp_path, profile={"rig": "head_mounted", "video": "x.mp4",
                                                   "embodiment": "franka_panda"},
                        _result=result)
    assert run.status == "completed"
    assert run.quality == 3 and run.num_frames == 5 and run.num_episodes == 1
    s = run.summary()
    assert s["task"] == "pick up cup" and s["stages"]["canonical"] == "done"
    c = run.certificate()
    assert c["quality"] == 3 and c["consent"] == "pending" and c["deliverable"] is False
