from __future__ import annotations

import json


def test_run_manifest_structures_legacy_coverage_note_and_omissions(tmp_path):
    from actuate.config import RigType
    from actuate.pipeline.run import _Ctx, _write_run_manifest
    from actuate.schema import CanonicalEpisode, CanonicalFrame

    session = tmp_path / "session"
    out = tmp_path / "out"
    session.mkdir()
    out.mkdir()
    (session / "video.mp4").write_bytes(b"video-ref")
    (session / "session_meta.json").write_text(
        json.dumps({"frame_count": 90, "rig": "head_mounted"})
    )
    (out / "artifacts").mkdir()
    (out / "artifacts" / "privacy_report.json").write_text("{}")
    episode = CanonicalEpisode(
        episode_id="ep",
        capture_id="a" * 64,
        rig=RigType.HEAD_MOUNTED,
        task="test",
        frames=(CanonicalFrame(
            t=0.0,
            rig=RigType.HEAD_MOUNTED,
            episode_id="ep",
            frame_idx=15,
        ),),
    )
    canonical = out / "canonical.json"
    canonical.write_text(episode.model_dump_json())
    ctx = _Ctx(
        session=session,
        out=out,
        profile={
            "video": "video.mp4",
            "perception": {"max_frames": None},
        },
        reporter=lambda *_args: None,
        confirm=lambda _prompt: False,
        checkpoint={
            "perceive": {"status": "done", "note": "90/90 frames (full source)"},
            "retarget": {"status": "skipped", "note": "no trained model"},
        },
    )
    ctx.canonical_path = canonical
    _write_run_manifest(ctx)
    manifest = json.loads((out / "run_manifest.json").read_text())
    coverage = manifest["result"]["frame_coverage"]
    assert coverage["source_evaluated_fraction"] == 1.0
    assert coverage["canonical_retained_fraction"] == round(1 / 90, 6)
    assert coverage["sampling_mode"] == "full_source"
    assert manifest["artifacts"]["privacy_report"] == "artifacts/privacy_report.json"
    assert manifest["not_produced"]["robot_trajectory"] == "no trained model"


def test_lineage_is_machine_readable_and_profile_bound():
    from actuate.lineage import build_lineage

    first = build_lineage({"rig": "head_mounted", "max_frames": None})
    second = build_lineage({"rig": "head_mounted", "max_frames": 45})
    assert first["python_version"]
    assert first["profile_sha256"] != second["profile_sha256"]
    assert "git_sha" in first and "dependencies" in first


def test_lineage_uses_validated_build_sha_when_git_metadata_is_absent(monkeypatch):
    import subprocess

    from actuate import lineage

    def unavailable(*_args, **_kwargs):
        raise subprocess.SubprocessError("source archive has no .git directory")

    commit = "a1" * 20
    monkeypatch.setattr(lineage.subprocess, "run", unavailable)
    monkeypatch.setenv("ACTUATE_BUILD_GIT_SHA", commit)
    monkeypatch.setenv("ACTUATE_BUILD_GIT_DIRTY", "false")
    assert lineage._git_state() == (commit, False)

    monkeypatch.setenv("ACTUATE_BUILD_GIT_SHA", "not-a-commit")
    assert lineage._git_state() == (None, None)


def test_run_readme_ships_frozen_schema_and_correct_copy(tmp_path):
    from actuate.package.delivery_docs import write_run_readme
    from actuate.schema import SCHEMA_VERSION

    write_run_readme(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    schema = tmp_path / f"canonical_v{SCHEMA_VERSION}.schema.json"
    assert schema.is_file() and json.loads(schema.read_text())["title"]
    assert "may be copied" in readme
    assert "may re-encode" in readme


def test_cache_identity_uses_capture_hash_not_staging_path(tmp_path):
    from actuate.pipeline.cache import _session_identity

    digest = "a" * 64
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    payload = json.dumps({"content_hash": digest})
    (one / "capture_manifest.json").write_text(payload)
    (two / "capture_manifest.json").write_text(payload)
    assert _session_identity(one) == _session_identity(two) == digest
