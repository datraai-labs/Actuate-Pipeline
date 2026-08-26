import json
import sqlite3
import zipfile

import pytest
from actuate_delivery.qc import build_qc
from actuate_delivery.run import complete_local_delivery
from actuate_delivery.web import (
    BatchCreate,
    Decision,
    WebRun,
    _bind_calibration,
    _calibration_label,
    _safe_relative,
    _source_signature,
    create_app,
)
from actuate_delivery.workflow import _record, approve_stage, run_stage
from fastapi import HTTPException


def test_new_browser_batch_is_calibration_neutral_by_default():
    assert BatchCreate(name="new batch").use_configured_calibration is False


def test_empty_browser_starts_with_neutral_dataset(tmp_path):
    app = create_app(tmp_path / "source", tmp_path / "run", tmp_path / "delivery")
    batches = next(route.endpoint for route in app.routes
                   if getattr(route, "path", None) == "/api/batches")

    assert batches() == [{
        "batch_id": "initial",
        "name": "New dataset",
        "status": "empty",
        "episodes": 0,
        "incomplete": 0,
        "delivery_ready": False,
        "source_files": 0,
        "source_bytes": 0,
    }]


def test_calibration_label_contains_facts_not_internal_ids():
    label = _calibration_label({
        "calibration_id": "calibration_000001",
        "rig_id": "rig_000001",
        "source": {"method": "Kalibr"},
        "transforms": {"baseline_m": 0.07048174243693624},
    })

    assert label == "Configured stereo calibration - Kalibr - 70.5 mm baseline"
    assert "rig_" not in label
    assert "calibration_" not in label


def ready_run(run_dir, partial=False):
    capture_id = "a" * 64
    coverage = 2 if partial else 3
    facts = {
        "capture_id": capture_id,
        "capture_layout": "single_video",
        "grouping_status": "complete",
        "source": {"file_count": 1, "bytes": 10, "verified_members": 1,
                   "all_hashes_verified_in_current_run": True, "members": []},
        "imu": {"status": "decoded", "sample_count": 20},
        "streams": [{"camera_stream_id": "single",
                     "vts": {"status": "decoded", "frame_count": 3},
                     "video": {"status": "verified", "frame_count": 3,
                               "codec": "hevc", "width": 1920, "height": 1080}}],
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
    run_dir.mkdir()
    artifact = build_qc(facts, run_dir / f"work/{capture_id}/qc_internal.json")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
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
            INSERT INTO capture_candidate VALUES ('batch', 'take', 'single_video', 'complete', 1);
            PRAGMA user_version = 11;
        """)
        database.execute(
            "INSERT INTO capture_snapshot VALUES ('batch', 'take', ?, 1)", (capture_id,))
        database.execute(
            "INSERT INTO qc_artifact VALUES (?, '', 'ready', ?, ?, ?, ?, ?, ?, NULL)",
            (capture_id, f"work/{capture_id}/qc_internal.json", artifact.json_sha256,
             artifact.pass_count, artifact.fail_count, artifact.unknown_count,
             artifact.not_applicable_count),
        )
    for stage in ("inventory", "sensors", "video", "timing", "qc"):
        _record(run_dir / "run.sqlite", stage, "complete", {})
        approve_stage(run_dir, stage, "owner")
    return capture_id


def test_web_decision_round_trip_and_stale_rejection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    web = WebRun(source, run_dir, tmp_path / "delivery")
    episode = web.episodes()[0][0]
    assert episode["episode_id"] == "episode_000001"
    decision = Decision(
        qc_sha256=episode["qc_sha256"], expected_revision="", status="include",
        limitations=[], decided_by="owner",
    )

    web.decide(capture_id, decision)
    saved = web.episodes()[0][0]["decision"]
    assert saved["status"] == "include"
    assert saved["decided_by"] == "owner"

    with pytest.raises(HTTPException, match="Decision changed") as error:
        web.decide(capture_id, decision)
    assert error.value.status_code == 409


def test_web_invalid_material_decision_restores_review(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir, partial=True)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    web = WebRun(source, run_dir, tmp_path / "delivery")
    episode = web.episodes()[0][0]

    with pytest.raises(HTTPException, match="require declared limitations"):
        web.decide(capture_id, Decision(
            qc_sha256=episode["qc_sha256"], expected_revision="", status="include",
            limitations=[], decided_by="owner",
        ))
    assert web.episodes()[0][0]["decision"] is None


@pytest.mark.parametrize("path", ["../take.imu", "/take.imu", "System Volume Information/x"])
def test_web_rejects_unsafe_upload_paths(path):
    with pytest.raises(HTTPException):
        _safe_relative(path)


def test_source_manifest_ignores_system_volume_information(tmp_path):
    source = tmp_path / "source"
    (source / "sample").mkdir(parents=True)
    (source / "sample/take.imu").write_bytes(b"imu")
    (source / "System Volume Information").mkdir()
    (source / "System Volume Information/generated.txt").write_text("ignore")

    files = WebRun(source, tmp_path / "run", tmp_path / "delivery").source_files()

    assert files == [{"relative_path": "sample/take.imu", "size": 3}]
    assert _source_signature(files) == [("sample/take.imu", 3)]


def test_job_state_survives_restart_and_reports_interruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "web_job.json").write_text(json.dumps({
        "status": "running", "error": None, "started_at": "2026-08-25T10:00:00Z",
        "finished_at": None, "elapsed_seconds": None,
    }))

    web = WebRun(source, run_dir, tmp_path / "delivery")

    assert web.job()["status"] == "failed"
    assert "interrupted" in web.job()["error"]
    assert json.loads((run_dir / "web_job.json").read_text())["status"] == "failed"


def test_delivery_summary_exposes_every_file_and_raw_derived_counts(tmp_path):
    output = tmp_path / "delivery"
    raw = output / "episodes/capture/raw/take.mp4"
    derived = output / "episodes/capture/derived/imu.parquet"
    raw.parent.mkdir(parents=True)
    derived.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    derived.write_bytes(b"{}")
    (output / "episodes.csv").write_text("episode_id\n")
    (output / ".DS_Store").write_bytes(b"finder metadata")

    summary = WebRun(tmp_path / "source", tmp_path / "run", output).delivery_summary()

    assert summary["file_count"] == 3
    assert summary["raw_files"] == 1
    assert summary["derived_files"] == 2
    assert [item["path"] for item in summary["files"]] == [
        "episodes/capture/derived/imu.parquet",
        "episodes/capture/raw/take.mp4",
        "episodes.csv",
    ]

    web = WebRun(tmp_path / "source", tmp_path / "run", output)
    web.build_archive()
    with zipfile.ZipFile(web.archive) as archive:
        assert all(not name.endswith(".DS_Store") for name in archive.namelist())


def test_failed_artifact_does_not_hide_reviewable_episode(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    ready_run(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            """CREATE TABLE video_artifact (
                   capture_id TEXT, camera_stream_id TEXT, status TEXT, error TEXT)"""
        )
        database.execute(
            "INSERT INTO video_artifact VALUES (?, 'single', 'failed', 'decode failed')",
            ("b" * 64,),
        )

    episodes, warning = WebRun(source, run_dir, tmp_path / "delivery").episodes()

    assert [episode["episode_id"] for episode in episodes] == ["episode_000001"]
    assert "Complete episodes remain reviewable" in warning


def test_built_delivery_is_not_reported_as_failed_batch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "delivery"
    output.mkdir()
    run_dir = tmp_path / "run"
    ready_run(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            """CREATE TABLE video_artifact (
                   capture_id TEXT, camera_stream_id TEXT, status TEXT, error TEXT)"""
        )
        database.execute(
            "INSERT INTO video_artifact VALUES (?, 'single', 'failed', 'decode failed')",
            ("b" * 64,),
        )

    assert WebRun(source, run_dir, output).job() == {
        "status": "complete", "error": None, "started_at": None,
        "finished_at": None, "elapsed_seconds": None, "stage": None,
    }


def test_failed_video_stage_is_not_displayed_as_complete(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    ready_run(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            """CREATE TABLE video_artifact (
                   capture_id TEXT, camera_stream_id TEXT, status TEXT, error TEXT)"""
        )
        database.execute(
            "INSERT INTO video_artifact VALUES (?, 'single', 'failed', 'decode failed')",
            ("a" * 64,),
        )
    _record(run_dir / "run.sqlite", "video", "failed", error="1 video failed")

    steps = WebRun(source, run_dir, tmp_path / "delivery").steps()

    video = next(step for step in steps if step["name"] == "Verify video")
    assert video == {
        "name": "Verify video", "status": "failed",
        "detail": "1 of 1 camera files processed; 1 failed",
        "stage": "video", "approved": False, "summary": None,
        "error": "1 video failed",
    }


def test_review_step_counts_only_complete_candidates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            "INSERT INTO capture_candidate VALUES ('batch', 'broken', 'single_video', 'incomplete', 1)"
        )
        database.execute(
            "INSERT INTO capture_snapshot VALUES ('batch', 'broken', ?, 1)", ("b" * 64,)
        )
        database.execute(
            """INSERT INTO qc_artifact
               SELECT ?, input_signature, status, json_relative_path, json_sha256,
                      pass_count, fail_count, unknown_count, not_applicable_count, reason
               FROM qc_artifact WHERE capture_id=?""",
            ("b" * 64, capture_id),
        )
        database.execute(
            """INSERT INTO delivery_decision
               (capture_id, qc_sha256, status, limitations_json, decided_by, decided_at)
               SELECT capture_id, json_sha256, 'include', '[]', 'owner',
                      '2026-08-26T00:00:00Z' FROM qc_artifact WHERE capture_id=?""",
            (capture_id,),
        )
    (run_dir / "review.csv").unlink()

    run_stage(str(source), run_dir, "review")
    step = next(item for item in WebRun(source, run_dir, tmp_path / "delivery").steps()
                if item["name"] == "Human review")

    assert step == {
        "name": "Human review", "status": "awaiting_approval",
        "detail": "1 of 1 decisions saved", "stage": "review", "approved": False,
        "summary": {"review_path": str(run_dir / "review.csv"), "episodes": 1,
                    "included": 1, "excluded": 0, "pending": 0},
        "error": None,
    }


def test_calibration_is_bound_to_included_episodes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    web = WebRun(source, run_dir, tmp_path / "delivery")
    episode = web.episodes()[0][0]
    web.decide(capture_id, Decision(
        qc_sha256=episode["qc_sha256"], expected_revision="", status="include",
        limitations=[], decided_by="owner",
    ))

    _bind_calibration(web, {
        "calibration_id": "calibration_000001", "rig_id": "rig_000001",
        "applies_to_episode_ids": ["old_episode"],
    })

    calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["applies_to_episode_ids"] == ["episode_000001"]
    assert calibration["calibration_id"] == "calibration_000001"


def test_progress_marks_absent_sensors_not_applicable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.executescript("""
            CREATE TABLE source_file (
                selected INTEGER, present INTEGER, source_sha256 TEXT, role TEXT);
            CREATE TABLE capture_candidate (
                parent_path TEXT, capture_key TEXT, capture_layout TEXT,
                grouping_status TEXT, file_count INTEGER,
                PRIMARY KEY (parent_path, capture_key));
            CREATE TABLE capture_snapshot (
                parent_path TEXT, capture_key TEXT, capture_id TEXT, is_canonical INTEGER,
                PRIMARY KEY (parent_path, capture_key));
            CREATE TABLE video_artifact (
                capture_id TEXT, camera_stream_id TEXT, status TEXT, error TEXT);
            INSERT INTO source_file VALUES (1, 1, 'hash', 'video');
            INSERT INTO capture_candidate VALUES (
                'batch', 'camera1', 'single_video', 'incomplete', 1);
            INSERT INTO capture_snapshot VALUES ('batch', 'camera1', 'aaaaaaaa', 1);
            PRAGMA user_version = 9;
        """)
    web = WebRun(source, run_dir, tmp_path / "delivery")
    web.job_status = "running"
    _record(run_dir / "run.sqlite", "sensors", "complete", {
        "imu_decoded": 0, "imu_reused": 0, "imu_failed": 0,
        "vts_decoded": 0, "vts_reused": 0, "vts_failed": 0,
        "tel_decoded": 0, "tel_reused": 0, "tel_failed": 0,
    })

    steps = web.steps()

    assert steps[1] == {
        "name": "Decode sensors", "status": "awaiting_approval",
        "detail": "No IMU, VTS, or telemetry sidecars were recognized",
        "stage": "sensors", "approved": False,
        "summary": {"imu_decoded": 0, "imu_reused": 0, "imu_failed": 0,
                    "vts_decoded": 0, "vts_reused": 0, "vts_failed": 0,
                    "tel_decoded": 0, "tel_reused": 0, "tel_failed": 0},
        "error": None,
    }
    assert steps[2] == {
        "name": "Verify video", "status": "waiting",
        "detail": "0 of 1 camera files processed; 0 failed",
        "stage": "video", "approved": False, "summary": None, "error": None,
    }
    assert web.episodes() == ([], None)
