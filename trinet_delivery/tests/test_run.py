import sqlite3
from datetime import UTC, datetime
from hashlib import sha256

import trinet_delivery.run as run_module
from trinet_delivery.cli import app
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

    first_result = runner.invoke(
        app, ["run", str(source), str(run_dir), "--ui"]
    )
    assert first_result.exit_code == 0, first_result.output
    first, count, version = read_run(run_dir / "run.sqlite")

    same_source = source / ".." / source.name
    second_result = runner.invoke(app, ["run", str(same_source), str(run_dir)])
    assert second_result.exit_code == 0, second_result.output
    second, second_count, second_version = read_run(run_dir / "run.sqlite")

    assert first[0] == second[0]
    assert first[1:3] == second[1:3]
    assert first[1] == source.resolve().as_uri()
    assert first[3] == "2026-08-18T12:00:00+00:00"
    assert second[3] == "2026-08-18T12:01:00+00:00"
    assert count == second_count == 1
    assert version == second_version == 11
    assert "preserved=0" in second_result.output
    assert "qc_created=0" in first_result.output
    assert "ui=not_implemented_until_hosted_phase" in first_result.output


def test_run_rejects_changed_source_without_mutating_ledger(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    run_dir = tmp_path / "run"
    source_a.mkdir()
    source_b.mkdir()
    assert runner.invoke(app, ["run", str(source_a), str(run_dir)]).exit_code == 0
    database_path = run_dir / "run.sqlite"
    before = read_run(database_path)
    before_hash = sha256(database_path.read_bytes()).hexdigest()
    before_mtime = database_path.stat().st_mtime_ns

    result = runner.invoke(app, ["run", str(source_b), str(run_dir)])
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
        app, ["run", str(source), str(run_dir), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "review_sheet=" in result.output
    assert "delivery_status=no_captures_included" in result.output
    assert (run_dir / "review.csv").is_file()
    assert not output.exists()
