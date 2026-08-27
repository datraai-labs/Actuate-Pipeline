import csv
import json
import sqlite3
from hashlib import sha256
from pathlib import Path

import actuate_delivery.run as run_module
import pytest
from actuate_delivery.qc import build_qc
from actuate_delivery.run import RunError, RunInputError, complete_local_delivery


def facts(capture_id, grouping="complete", partial=False):
    coverage = 2 if partial else 3
    return {
        "capture_id": capture_id, "capture_layout": "single_video",
        "grouping_status": grouping,
        "source": {"file_count": 1, "bytes": 10, "verified_members": 1,
                   "all_hashes_verified_in_current_run": True, "members": []},
        "imu": {"status": "decoded", "sample_count": 20},
        "streams": [{"camera_stream_id": "single",
                     "vts": {"status": "decoded", "frame_count": 3},
                     "video": {"status": "verified", "frame_count": 3}}],
        "telemetry": {"status": "absent"},
        "timing": {"status": "ready", "row_count": 3, "matched_rows": 3,
                   "coverage_rows": coverage, "stereo_pair_count": 0,
                   "stereo_unmatched_rows": 0,
                   "streams": [{"camera_stream_id": "single", "row_count": 3,
                                "matched_rows": 3, "coverage_rows": coverage,
                                "outside_imu_coverage_rows": 3 - coverage,
                                "missing_sof_rows": 0, "video_only_rows": 0,
                                "vts_only_rows": 0}]},
    }


def insert_capture(run_dir, capture_id, group, grouping="complete", partial=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "run.sqlite"
    with sqlite3.connect(database_path) as database:
        if database.execute("PRAGMA user_version").fetchone()[0] == 0:
            database.executescript("""
                CREATE TABLE capture_candidate (
                    parent_path TEXT, capture_key TEXT, capture_layout TEXT,
                    grouping_status TEXT, file_count INTEGER,
                    PRIMARY KEY (parent_path, capture_key));
                CREATE TABLE capture_snapshot (
                    parent_path TEXT, capture_key TEXT, capture_id TEXT, is_canonical INTEGER,
                    PRIMARY KEY (parent_path, capture_key));
                CREATE TABLE qc_artifact (
                    capture_id TEXT PRIMARY KEY, input_signature TEXT, status TEXT,
                    json_relative_path TEXT, json_sha256 TEXT, pass_count INTEGER,
                    fail_count INTEGER, unknown_count INTEGER,
                    not_applicable_count INTEGER, reason TEXT);
                CREATE TABLE tel_artifact (
                    capture_id TEXT PRIMARY KEY, status TEXT, source_sha256 TEXT,
                    parquet_sha256 TEXT);
                PRAGMA user_version = 11;
            """)
        database.execute(
            "INSERT INTO capture_candidate VALUES ('batch/device', ?, 'single_video', ?, 1)",
            (group, grouping),
        )
        database.execute(
            "INSERT INTO capture_snapshot VALUES ('batch/device', ?, ?, 1)",
            (group, capture_id),
        )
        relative = f"work/{capture_id}/qc_internal.json"
        artifact = build_qc(facts(capture_id, grouping, partial), run_dir / relative)
        database.execute(
            "INSERT INTO qc_artifact VALUES (?, '', 'ready', ?, ?, ?, ?, ?, ?, NULL)",
            (capture_id, relative, artifact.json_sha256, artifact.pass_count,
             artifact.fail_count, artifact.unknown_count, artifact.not_applicable_count),
        )


def review_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=run_module.REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fake_package(monkeypatch, captured):
    def project(episodes, output, calibration=None, telemetry_mode="automatic"):
        assert calibration is None
        assert telemetry_mode == "exclude_all"
        captured.extend(episodes)
        output.mkdir(parents=True)

    def package(run_dir, projection, output):
        assert run_dir.is_dir() and projection.is_dir()
        output.mkdir(parents=True)

    monkeypatch.setattr(run_module, "project_supplier", project)
    monkeypatch.setattr(run_module, "build_delivery", package)
    monkeypatch.setattr(run_module, "_vendor_visualizations", lambda *args: [])


def test_review_resume_appends_new_capture_and_builds_only_included(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    output = tmp_path / "delivery"
    insert_capture(run_dir, "a" * 64, "take1")

    first = complete_local_delivery(str(source), run_dir, output)
    assert (first.status, first.included, first.excluded, first.pending) == (
        "review_required", 0, 0, 1)
    rows = review_rows(first.review_path)
    rows[0].update(decision="include")
    write_rows(first.review_path, rows)
    insert_capture(run_dir, "b" * 64, "take2")

    second = complete_local_delivery(str(source), run_dir, output)
    assert (second.status, second.included, second.excluded, second.pending) == (
        "review_required", 1, 0, 1)
    rows = review_rows(second.review_path)
    assert [row["source_group"] for row in rows] == ["take1", "take2"]
    rows[1].update(decision="exclude")
    write_rows(second.review_path, rows)
    captured = []
    fake_package(monkeypatch, captured)

    final = complete_local_delivery(str(source), run_dir, output)
    assert (final.status, final.included, final.excluded, final.pending) == (
        "complete", 1, 1, 0)
    assert output.is_dir()
    assert len(captured) == 1
    assert captured[0]["internal_qc"]["capture_id"] == "a" * 64
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 15
        decisions = database.execute(
            "SELECT capture_id, status, decided_at FROM delivery_decision ORDER BY 1"
        ).fetchall()
        assert [(row[0], row[1]) for row in decisions] == [
            ("a" * 64, "include"), ("b" * 64, "exclude")]
        assert all(row[2] for row in decisions)


def test_episode_ids_are_allocated_once_and_new_captures_append(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    output = tmp_path / "delivery"
    first_capture = "a" * 64
    second_capture = "b" * 64
    insert_capture(run_dir, first_capture, "take-z")

    complete_local_delivery(str(source), run_dir, output)
    assert review_rows(run_dir / "review.csv")[0]["episode_id"] == "episode_000001"

    insert_capture(run_dir, second_capture, "take-a")
    complete_local_delivery(str(source), run_dir, output)
    rows = {row["capture_id"]: row["episode_id"]
            for row in review_rows(run_dir / "review.csv")}

    assert rows == {
        first_capture: "episode_000001",
        second_capture: "episode_000002",
    }
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        assert database.execute(
            "SELECT capture_id, episode_number FROM delivery_episode ORDER BY episode_number"
        ).fetchall() == [(first_capture, 1), (second_capture, 2)]


def test_schema_14_review_migrates_without_episode_reviewer_or_free_text(tmp_path):
    run_dir = tmp_path / "run"
    capture_id = "a" * 64
    insert_capture(run_dir, capture_id, "take", partial=True)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.executescript("""
            CREATE TABLE delivery_decision (
                capture_id TEXT PRIMARY KEY, qc_sha256 TEXT NOT NULL,
                status TEXT NOT NULL, limitations_json TEXT NOT NULL,
                decided_by TEXT NOT NULL, decided_at TEXT NOT NULL);
            CREATE TABLE delivery_episode (
                capture_id TEXT PRIMARY KEY, episode_number INTEGER NOT NULL UNIQUE);
            PRAGMA user_version = 14;
        """)
        qc_sha256 = database.execute(
            "SELECT json_sha256 FROM qc_artifact WHERE capture_id=?", (capture_id,)
        ).fetchone()[0]
        database.execute(
            "INSERT INTO delivery_decision VALUES (?, ?, 'include', ?, 'old reviewer', ?)",
            (capture_id, qc_sha256, '["operator text"]', "2026-08-26T00:00:00Z"),
        )

    _, entries = run_module._delivery_review(run_dir / "run.sqlite", run_dir)

    with sqlite3.connect(run_dir / "run.sqlite") as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 15
        columns = [row[1] for row in database.execute("PRAGMA table_info(delivery_decision)")]
        decision = database.execute(
            "SELECT limitations_json, decided_at FROM delivery_decision"
        ).fetchone()
    assert "decided_by" not in columns
    assert json.loads(decision[0]) == [
        "1 camera frame is outside IMU coverage (single: 1)."
    ]
    assert decision[1] == "2026-08-26T00:00:00Z"
    assert "decided_by" not in review_rows(run_dir / "review.csv")[0]
    assert entries[0]["decision"]["limitations"] == json.loads(decision[0])


def test_complete_episodes_receive_ids_before_incomplete_candidates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    incomplete = "a" * 64
    complete = "b" * 64
    insert_capture(run_dir, incomplete, "take-a", grouping="incomplete")
    insert_capture(run_dir, complete, "take-z")

    complete_local_delivery(str(source), run_dir, tmp_path / "delivery")
    rows = {row["capture_id"]: row["episode_id"]
            for row in review_rows(run_dir / "review.csv")}

    assert rows == {
        complete: "episode_000001",
        incomplete: "episode_000002",
    }


@pytest.mark.parametrize(("changes", "message"), [
    ({"capture_layout": "stereo_pair"}, "facts changed or became stale"),
    ({"qc_sha256": "0" * 64}, "facts changed or became stale"),
    ({"decision": "maybe"}, "Invalid review decision"),
    ({"limitations_json": "{"}, "Invalid limitations JSON"),
    ({"decision": "exclude", "limitations_json": '["not delivered"]'},
     "pipeline-controlled"),
])
def test_review_rejects_invalid_or_edited_rows(tmp_path, changes, message):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    output = tmp_path / "delivery"
    insert_capture(run_dir, "a" * 64, "take")
    complete_local_delivery(str(source), run_dir, output)
    rows = review_rows(run_dir / "review.csv")
    rows[0].update(changes)
    write_rows(run_dir / "review.csv", rows)

    with pytest.raises(RunError, match=message):
        complete_local_delivery(str(source), run_dir, output)


def test_changed_qc_replaces_stale_saved_decision_with_pending_review(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    output = tmp_path / "delivery"
    capture_id = "a" * 64
    insert_capture(run_dir, capture_id, "take")
    complete_local_delivery(str(source), run_dir, output)
    rows = review_rows(run_dir / "review.csv")
    rows[0]["decision"] = "include"
    write_rows(run_dir / "review.csv", rows)
    run_module._delivery_review(run_dir / "run.sqlite", run_dir)

    path = run_dir / f"work/{capture_id}/qc_internal.json"
    changed = build_qc(facts(capture_id, partial=True), path)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            """UPDATE qc_artifact SET json_sha256=?, pass_count=?, fail_count=?,
                      unknown_count=?, not_applicable_count=? WHERE capture_id=?""",
            (changed.json_sha256, changed.pass_count, changed.fail_count,
             changed.unknown_count, changed.not_applicable_count, capture_id),
        )

    result = complete_local_delivery(str(source), run_dir, output)

    assert result.status == "review_required"
    assert review_rows(run_dir / "review.csv")[0]["decision"] == ""
    assert not output.exists()


def test_review_rejects_duplicate_removed_and_blocking_rows(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    duplicate_run = tmp_path / "duplicate-run"
    insert_capture(duplicate_run, "a" * 64, "take")
    complete_local_delivery(str(source), duplicate_run, tmp_path / "duplicate-output")
    rows = review_rows(duplicate_run / "review.csv")
    write_rows(duplicate_run / "review.csv", rows + rows)
    with pytest.raises(RunError, match="duplicate capture"):
        complete_local_delivery(str(source), duplicate_run, tmp_path / "duplicate-output")

    removed_run = tmp_path / "removed-run"
    insert_capture(removed_run, "b" * 64, "take")
    complete_local_delivery(str(source), removed_run, tmp_path / "removed-output")
    with sqlite3.connect(removed_run / "run.sqlite") as database:
        database.execute("DELETE FROM capture_snapshot")
    with pytest.raises(RunError, match="no longer current"):
        complete_local_delivery(str(source), removed_run, tmp_path / "removed-output")

    blocked_run = tmp_path / "blocked-run"
    insert_capture(blocked_run, "c" * 64, "take", grouping="incomplete")
    complete_local_delivery(str(source), blocked_run, tmp_path / "blocked-output")
    rows = review_rows(blocked_run / "review.csv")
    assert "capture_structure" in rows[0]["blocking_checks"]
    rows[0].update(decision="include")
    write_rows(blocked_run / "review.csv", rows)
    with pytest.raises(RunError, match="Blocking capture cannot be included"):
        complete_local_delivery(str(source), blocked_run, tmp_path / "blocked-output")


def test_material_include_requires_controlled_limitation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    output = tmp_path / "delivery"
    insert_capture(run_dir, "a" * 64, "take", partial=True)
    complete_local_delivery(str(source), run_dir, output)
    rows = review_rows(run_dir / "review.csv")
    assert "camera_imu_coverage" in rows[0]["material_checks"]
    rows[0].update(decision="include", limitations_json='["operator text"]')
    write_rows(run_dir / "review.csv", rows)
    with pytest.raises(RunError, match="pipeline-controlled"):
        complete_local_delivery(str(source), run_dir, output)

    rows = review_rows(run_dir / "review.csv")
    rows[0]["limitations_json"] = (
        '["1 camera frame is outside IMU coverage (single: 1)."]')
    write_rows(run_dir / "review.csv", rows)
    fake_package(monkeypatch, [])
    assert complete_local_delivery(str(source), run_dir, output).status == "complete"


def test_delivery_output_cannot_overlap_source_or_already_exist(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    insert_capture(run_dir, "a" * 64, "take")

    with pytest.raises(RunInputError, match="outside the source"):
        complete_local_delivery(str(source), run_dir, source / "delivery")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RunInputError, match="already exists"):
        complete_local_delivery(str(source), run_dir, existing)


def test_delivery_review_requires_ready_qc_for_every_current_capture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    insert_capture(run_dir, "a" * 64, "take")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute("UPDATE qc_artifact SET status='failed' WHERE capture_id=?", ("a" * 64,))

    with pytest.raises(RunError, match="Every current capture must have ready QC"):
        complete_local_delivery(str(source), run_dir, tmp_path / "delivery")


def test_review_sheet_qc_hash_matches_current_artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    insert_capture(run_dir, "a" * 64, "take")
    result = complete_local_delivery(str(source), run_dir, tmp_path / "delivery")
    row = review_rows(result.review_path)[0]
    qc_path = run_dir / "work" / ("a" * 64) / "qc_internal.json"

    assert row["qc_sha256"] == sha256(qc_path.read_bytes()).hexdigest()
    qc_path.write_text(json.dumps({"changed": True}))
    with pytest.raises(RunError, match="Current QC artifact changed"):
        complete_local_delivery(str(source), run_dir, tmp_path / "delivery")


def test_review_write_failure_rolls_back_decision_and_can_retry(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    insert_capture(run_dir, "a" * 64, "take")
    complete_local_delivery(str(source), run_dir, tmp_path / "delivery")
    rows = review_rows(run_dir / "review.csv")
    rows[0].update(decision="exclude")
    write_rows(run_dir / "review.csv", rows)
    replace = Path.replace

    def fail_review_replace(path, target):
        if path.name == ".review.csv.staging":
            raise OSError("injected review publication failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_review_replace)
    with pytest.raises(RunError, match="injected review publication failure"):
        complete_local_delivery(str(source), run_dir, tmp_path / "delivery")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        assert database.execute("SELECT COUNT(*) FROM delivery_decision").fetchone()[0] == 0

    monkeypatch.setattr(Path, "replace", replace)
    result = complete_local_delivery(str(source), run_dir, tmp_path / "delivery")
    assert result.status == "no_captures_included"
