import sqlite3
from datetime import UTC, datetime
from hashlib import sha256

import actuate_delivery.cli as cli_module
import actuate_delivery.run as run_module
from actuate_delivery.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def read_run(database_path):
    with sqlite3.connect(database_path) as database:
        row = database.execute(
            "SELECT run_id, source, created_at, last_resumed_at FROM run"
        ).fetchone()
        count = database.execute("SELECT COUNT(*) FROM run").fetchone()[0]
        version = database.execute("PRAGMA user_version").fetchone()[0]
    return row, count, version


def test_run_creates_and_resumes_one_ledger(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    source.mkdir()
    times = iter(
        [
            datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 18, 12, 1, tzinfo=UTC),
        ]
    )

    class Clock:
        @classmethod
        def now(cls, timezone):
            assert timezone is UTC
            return next(times)

    monkeypatch.setattr(run_module, "datetime", Clock)

    output = tmp_path / "delivery"
    first_result = runner.invoke(
        app, ["run", str(source), str(run_dir), "--output", str(output)],
        input="n\n",
    )
    assert first_result.exit_code == 0, first_result.output
    first, count, version = read_run(run_dir / "run.sqlite")

    same_source = source / ".." / source.name
    second_result = runner.invoke(
        app, ["run", str(same_source), str(run_dir), "--output", str(output)],
        input="n\n",
    )
    assert second_result.exit_code == 0, second_result.output
    second, second_count, second_version = read_run(run_dir / "run.sqlite")

    assert first[0] == second[0]
    assert first[1:3] == second[1:3]
    assert first[1] == source.resolve().as_uri()
    assert first[3] == "2026-08-18T12:00:00+00:00"
    assert second[3] == "2026-08-18T12:01:00+00:00"
    assert count == second_count == 1
    assert version == second_version == 6
    assert "unchanged: 0" in second_result.output
    assert "Stopped safely" in first_result.output
    assert "Operator name" not in first_result.output


def test_run_rejects_changed_source_without_mutating_ledger(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    run_dir = tmp_path / "run"
    source_a.mkdir()
    source_b.mkdir()
    output = tmp_path / "delivery"
    assert runner.invoke(
        app, ["run", str(source_a), str(run_dir), "--output", str(output)],
        input="n\n",
    ).exit_code == 0
    database_path = run_dir / "run.sqlite"
    before = read_run(database_path)
    before_hash = sha256(database_path.read_bytes()).hexdigest()
    before_mtime = database_path.stat().st_mtime_ns

    result = runner.invoke(
        app, ["run", str(source_b), str(run_dir), "--output", str(output)],
        input="",
    )
    after = read_run(database_path)

    assert result.exit_code != 0
    assert "Invalid value for SOURCE" in result.output
    assert source_a.name in result.output
    assert source_b.name in result.output
    assert after == before
    assert sha256(database_path.read_bytes()).hexdigest() == before_hash
    assert database_path.stat().st_mtime_ns == before_mtime


def test_team_option_was_replaced_without_starting_a_run(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    source.mkdir()

    result = runner.invoke(
        app, ["run", str(source), str(run_dir), "--team"]
    )

    assert result.exit_code != 0
    assert "No such option: --team" in result.output
    assert not run_dir.exists()


def test_output_option_creates_review_sheet_before_delivery(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    output = tmp_path / "delivery"
    source.mkdir()

    result = runner.invoke(
        app, ["run", str(source), str(run_dir), "--output", str(output)],
        input="y\ny\ny\ny\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "review path:" in result.output
    assert "Approve Run QC and continue?" in result.output
    assert "Operator name" not in result.output
    assert (run_dir / "review.csv").is_file()
    assert not output.exists()


def test_status_reports_persisted_file_progress_and_timing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "take.txt").write_text("raw")
    run_dir = tmp_path / "run"
    run_module.prepare_inventory(str(source), run_dir)

    result = runner.invoke(app, ["status", str(run_dir)])

    assert result.exit_code == 0
    assert "progress: 1 of 1" in result.output
    assert "take.txt: preserved" in result.output


def test_partial_telemetry_policy_is_required_persisted_and_invalidated(tmp_path):
    database_path = tmp_path / "run.sqlite"
    first, second = "a" * 64, "b" * 64
    with sqlite3.connect(database_path) as database:
        database.execute(
            """CREATE TABLE tel_artifact (
                   capture_id TEXT PRIMARY KEY, status TEXT, source_sha256 TEXT,
                   parquet_sha256 TEXT)"""
        )
        database.execute(
            "INSERT INTO tel_artifact VALUES (?, 'decoded', ?, ?)",
            (first, "1" * 64, "2" * 64),
        )
    included = [
        {"row": {"capture_id": first}},
        {"row": {"capture_id": second}},
    ]

    pending = run_module._telemetry_policy(database_path, included)
    assert pending == {
        "coverage": "partial", "included_episodes": 2,
        "episodes_with_telemetry": 1, "requires_choice": True, "choice": None,
    }
    chosen = run_module._telemetry_policy(database_path, included, "exclude_all")
    assert chosen["choice"] == "exclude_all"
    assert not chosen["requires_choice"]
    assert run_module._telemetry_policy(database_path, included) == chosen

    with sqlite3.connect(database_path) as database:
        database.execute(
            "UPDATE tel_artifact SET source_sha256=? WHERE capture_id=?",
            ("3" * 64, first),
        )
    stale = run_module._telemetry_policy(database_path, included)
    assert stale["requires_choice"]
    assert stale["choice"] is None


def test_cli_resolves_partial_telemetry_inside_the_delivery_flow(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module, "STAGES", (("delivery", "Build delivery"),))
    monkeypatch.setattr(cli_module, "workflow_state", lambda run_dir: [{
        "stage": "delivery", "status": "waiting", "approved": False, "summary": None,
    }])

    def policy(run_dir, choice=None):
        calls.append(choice)
        return {
            "coverage": "partial", "included_episodes": 3,
            "episodes_with_telemetry": 1, "requires_choice": choice is None,
            "choice": choice,
        }

    monkeypatch.setattr(cli_module, "telemetry_policy", policy)
    monkeypatch.setattr(cli_module, "run_stage", lambda *args: type("Result", (), {
        "summary": {"output": str(tmp_path / "delivery")},
    })())
    monkeypatch.setattr(cli_module, "_show", lambda *args: None)

    result = runner.invoke(app, [
        "run", str(tmp_path / "source"), str(tmp_path / "run"),
        "--output", str(tmp_path / "delivery"),
    ], input="include available\ny\n")

    assert result.exit_code == 0, result.output
    assert "Telemetry is available for 1 of 3 included episodes" in result.output
    assert calls == [None, "include_available"]
