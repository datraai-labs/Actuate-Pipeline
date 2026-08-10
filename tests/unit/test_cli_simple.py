"""Phase 6 Part B: the simplified CLI (login/config/status/report/process/export) via
Typer's CliRunner -- no GPU, no network. Process's heavy path is mocked; the offline
commands (status/config/report) run for real against a tiny prebuilt canonical."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from actuate.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTUATE_HOME", str(tmp_path / ".actuate"))


def _tiny_canonical(dir_):
    from actuate.config import ConsentStatus, RigType
    from actuate.schema import CanonicalEpisode
    from actuate.schema.episode import EpisodeMeta
    from actuate.schema.frame import CanonicalFrame

    frames = tuple(CanonicalFrame(t=i / 30.0, rig=RigType.HEAD_MOUNTED, episode_id="ep",
                                  frame_idx=i, confidence={"hands": 0.8},
                                  provenance={"hands": "vision_primary"}) for i in range(4))
    ep = CanonicalEpisode(episode_id="ep", capture_id="c" * 64, rig=RigType.HEAD_MOUNTED,
                          frames=frames, task="pick up cup",
                          consent=ConsentStatus.GRANTED,
                          episode_meta=EpisodeMeta(quality=2, speed=1))
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "canonical.json").write_text(ep.model_dump_json())
    return dir_


# ---------------------------------------------------------------- login / status / config
def test_login_local_then_status_shows_ready():
    assert runner.invoke(app, ["login", "--local"]).exit_code == 0
    r = runner.invoke(app, ["status"])
    assert r.exit_code == 0 and "authenticated: True" in r.stdout


def test_login_cloud_key_is_masked_in_config_show():
    runner.invoke(app, ["login", "--key", "ak_super_secret"])
    r = runner.invoke(app, ["config", "show"])
    assert "ak_super_secret" not in r.stdout
    assert "set (hidden)" in r.stdout


def test_config_set_and_show():
    runner.invoke(app, ["login", "--local"])
    assert runner.invoke(app, ["config", "set", "default_embodiment", "aloha_v2"]).exit_code == 0
    assert "aloha_v2" in runner.invoke(app, ["config", "show"]).stdout


def test_config_set_rejects_unknown_key():
    r = runner.invoke(app, ["config", "set", "api_key", "x"])
    assert r.exit_code == 1 and "cannot set" in r.stdout


# ---------------------------------------------------------------- report (real canonical)
def test_report_prints_certificate_and_consent(tmp_path):
    d = _tiny_canonical(tmp_path / "run")
    r = runner.invoke(app, ["report", str(d)])
    assert r.exit_code == 0
    assert "quality 2/5" in r.stdout
    assert "deliverable: False" in r.stdout            # consent granted but pii pending
    assert "consent=granted" in r.stdout


def test_report_errors_with_a_next_step(tmp_path):
    r = runner.invoke(app, ["report", str(tmp_path / "empty")])
    assert r.exit_code == 1
    assert "actuate process" in r.stdout               # tells the user what to DO


# ---------------------------------------------------------------- process (mocked heavy path)
def test_process_error_message_is_actionable(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("source not found: nope.mp4")

    import actuate

    monkeypatch.setattr(actuate, "process", _boom)
    r = runner.invoke(app, ["process", "nope.mp4", "--task", "x"])
    assert r.exit_code == 1 and "cannot process" in r.stdout


def test_process_only_grants_consent_when_operator_says_so(monkeypatch, tmp_path):
    seen = []

    class _Run:
        status = "completed"
        _result = SimpleNamespace(out=tmp_path / "out")

        def summary(self):
            return {
                "quality": None,
                "num_frames": 1,
                "task": "pick up cup",
                "canonical_path": str(tmp_path / "out" / "canonical.json"),
            }

    import actuate

    def _process(*args, **kwargs):
        seen.append(kwargs.get("consent"))
        return _Run()

    monkeypatch.setattr(actuate, "process", _process)
    without = runner.invoke(app, ["process", "clip.mp4", "--task", "pick up cup"])
    with_flag = runner.invoke(
        app,
        ["process", "clip.mp4", "--task", "pick up cup", "--consent-granted"],
    )
    assert without.exit_code == 0 and with_flag.exit_code == 0
    assert seen == [None, "granted"]


def test_export_errors_without_canonical(tmp_path):
    r = runner.invoke(app, ["export", str(tmp_path / "empty")])
    assert r.exit_code == 1 and "actuate process" in r.stdout


def test_deliver_lists_all_missing_options_at_once(tmp_path):
    r = runner.invoke(app, ["deliver", str(tmp_path / "dataset")])
    assert r.exit_code == 2
    assert "missing required options: --customer, --episodes" in r.stdout


# ---------------------------------------------------------------- power-user tree survives
def test_advanced_commands_still_registered():
    help_text = runner.invoke(app, ["--help"]).stdout
    for cmd in ("run", "certify", "retarget", "language", "deliver", "package"):
        assert cmd in help_text                        # not broken by the simplified layer
