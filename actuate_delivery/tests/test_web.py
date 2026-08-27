import json
import sqlite3
import zipfile
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from actuate_delivery.qc import build_qc
from actuate_delivery.run import complete_local_delivery, processing_progress
from actuate_delivery.web import (
    Approval,
    BatchCreate,
    Decision,
    SelectedSourceFile,
    TelemetryChoice,
    WebRun,
    _bind_calibration,
    _calibration_label,
    _safe_relative,
    _source_signature,
    create_app,
)
from actuate_delivery.workflow import _record, approve_stage, run_stage, workflow_state
from fastapi import HTTPException


def test_new_browser_batch_is_calibration_neutral_by_default():
    batch = BatchCreate(name="new batch", files=[
        SelectedSourceFile(relative_path="day/take.mp4", size=3, selected=True),
    ])

    assert batch.use_configured_calibration is False


def test_browser_batch_persists_selection_and_rejects_unselected_upload(tmp_path):
    run_dir = tmp_path / "run"
    app = create_app(tmp_path / "source", run_dir, tmp_path / "delivery")
    create = next(route.endpoint for route in app.routes
                  if getattr(route, "path", None) == "/api/batches"
                  and "POST" in route.methods)
    start = next(route.endpoint for route in app.routes
                 if getattr(route, "path", None) == "/api/upload/start/{relative_path:path}")
    finish = next(route.endpoint for route in app.routes
                  if getattr(route, "path", None) == "/api/upload/complete/{relative_path:path}")
    files = [
        SelectedSourceFile(relative_path="Device01/take1.mp4", size=3, selected=True),
        SelectedSourceFile(relative_path="Device02/take2.mp4", size=4, selected=False),
    ]

    batch = create(BatchCreate(name="mixed devices", files=files))
    metadata = json.loads(next((run_dir.parent / "batches").glob("*/batch.json")).read_text())

    assert metadata["selection"] == [
        {"relative_path": "Device01/take1.mp4", "size": 3, "selected": True},
        {"relative_path": "Device02/take2.mp4", "size": 4, "selected": False},
    ]
    assert start("Device01/take1.mp4", batch["batch_id"])["offset"] == 0
    with pytest.raises(HTTPException, match="size differs") as size_error:
        finish("Device01/take1.mp4", 4, batch["batch_id"])
    assert size_error.value.status_code == 400
    with pytest.raises(HTTPException, match="not selected") as error:
        start("Device02/take2.mp4", batch["batch_id"])
    assert error.value.status_code == 409


def test_browser_batch_rejects_duplicate_selection_paths(tmp_path):
    app = create_app(tmp_path / "source", tmp_path / "run", tmp_path / "delivery")
    create = next(route.endpoint for route in app.routes
                  if getattr(route, "path", None) == "/api/batches"
                  and "POST" in route.methods)
    files = [
        SelectedSourceFile(relative_path="Device01/take.mp4", size=3, selected=True),
        SelectedSourceFile(relative_path="Device01/take.mp4", size=4, selected=False),
    ]

    with pytest.raises(HTTPException, match="duplicate relative path") as error:
        create(BatchCreate(name="duplicate", files=files))
    assert error.value.status_code == 400


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


def test_stage_approval_route_accepts_nameless_confirmation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.txt").write_text("retained auxiliary file")
    run_dir = tmp_path / "run"
    run_stage(str(source), run_dir, "inventory")
    app = create_app(source, run_dir, tmp_path / "delivery")
    approve = next(route.endpoint for route in app.routes
                   if getattr(route, "path", None) == "/api/stages/{stage}/approve")

    assert approve("inventory", Approval()) == {
        "status": "approved", "stage": "inventory",
    }
    inventory = workflow_state(run_dir)[0]
    assert inventory["approved"] is True
    assert inventory["approved_by"] == ""


def test_review_ui_has_one_optional_batch_reviewer_and_no_editable_limitations():
    html = (Path(__file__).parents[1] / "review/index.html").read_text()

    assert 'id="checkpoint-reviewer"' not in html
    assert 'id="reviewer"' not in html
    assert 'id="limitations"' not in html
    assert "Supplier-visible limitations" not in html
    assert "Reviewer name - optional" in html
    assert html.count('autocomplete="name"') == 1
    assert "Camera and sensor timeline" in html
    assert "Timestamp coverage does not certify physical synchronization" in html
    assert "Jump to issue" in html
    assert "Before IMU" in html and "After IMU" in html and "Closest IMU" in html


def test_review_ui_previews_nested_selection_before_upload():
    html = (Path(__file__).parents[1] / "review/index.html").read_text()

    assert '<dialog class="selection-dialog"' in html
    assert 'id="selected-files"' in html
    assert 'data-folder=' in html
    assert 'data-file=' in html
    assert "selectionFiles.filter(item=>item.selected)" in html
    assert "const files=selectionFiles.map" in html
    assert "$('#selection').showModal();selectionChanged()" in html
    assert "folderRows(child,branch,folder,depth+1,rows)" in html


def test_review_ui_restores_latest_batch_and_can_finish_missing_zip():
    html = (Path(__file__).parents[1] / "review/index.html").read_text()

    assert "latest=[...batches].reverse().find" in html
    assert "delivery.exists?'Finish delivery ZIP':'Build customer delivery'" in html
    assert "The customer folder is complete, but its ZIP is not ready" in html
    assert "$('#download').download=`actuate-${activeBatch}.zip`" in html
    assert 'class="item-progress"' in html
    assert "step.progress.completed" in html
    assert "step.progress.current.label" in html
    assert 'data-telemetry="exclude_all"' in html
    assert 'data-telemetry="include_available"' in html
    assert "needsTelemetry" in html


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
            CREATE TABLE tel_artifact (
                capture_id TEXT PRIMARY KEY, status TEXT, source_sha256 TEXT,
                parquet_sha256 TEXT, error TEXT);
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


def evidence_run(run_dir):
    capture_id = "e" * 64
    work = run_dir / f"work/{capture_id}"
    cache = run_dir / "cache/blobs"
    work.mkdir(parents=True)
    cache.mkdir(parents=True)
    imu_path = work / "imu.parquet"
    timing_path = work / "frame_timing.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"accel_x_mps2": 1.0, "accel_y_mps2": 2.0, "accel_z_mps2": 3.0,
         "gyro_x_rad_s": 0.1, "gyro_y_rad_s": 0.2, "gyro_z_rad_s": 0.3},
        {"accel_x_mps2": 2.0, "accel_y_mps2": 3.0, "accel_z_mps2": 4.0,
         "gyro_x_rad_s": 0.2, "gyro_y_rad_s": 0.3, "gyro_z_rad_s": 0.4},
    ]), imu_path)
    base = {
        "vts_frame_number": 1, "venc_seq": 10, "sof_timestamp_ns": 1_000,
        "vts_match_status": "matched", "before_imu_index": 0,
        "before_imu_timestamp_ns": 900, "before_delta_ns": -100,
        "after_imu_index": 1, "after_imu_timestamp_ns": 1_100, "after_delta_ns": 100,
        "closest_imu_index": 0, "closest_imu_timestamp_ns": 900, "closest_delta_ns": -100,
        "within_imu_coverage": True, "mapping_status": "mapped",
        "stereo_peer_stream_id": "right", "stereo_peer_video_frame_index": 0,
        "stereo_pair_status": "matched",
    }
    pq.write_table(pa.Table.from_pylist([
        base | {"camera_stream_id": "left", "video_frame_index": 0, "mp4_pts_ns": 0,
                "within_imu_coverage": False, "stereo_peer_stream_id": None,
                "stereo_peer_video_frame_index": None, "stereo_pair_status": "unmatched"},
        base | {"camera_stream_id": "left", "video_frame_index": 1,
                "mp4_pts_ns": 33_000_000, "vts_frame_number": 2, "venc_seq": 11,
                "closest_imu_index": 1},
        base | {"camera_stream_id": "right", "video_frame_index": 0, "mp4_pts_ns": 0,
                "stereo_peer_stream_id": "left"},
        base | {"camera_stream_id": "right", "video_frame_index": 1,
                "mp4_pts_ns": 33_000_000, "vts_frame_number": 2, "venc_seq": 11,
                "closest_imu_index": 1, "stereo_peer_stream_id": "left",
                "stereo_peer_video_frame_index": 1},
    ]), timing_path)
    video = cache / "video"
    video.write_bytes(b"0123456789")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.executescript("""
            CREATE TABLE capture_snapshot (
                parent_path TEXT, capture_key TEXT, capture_id TEXT, is_canonical INTEGER);
            CREATE TABLE capture_member (
                parent_path TEXT, capture_key TEXT, source_item_id TEXT,
                role TEXT, camera_stream_id TEXT);
            CREATE TABLE source_file (
                source_item_id TEXT, cache_relative_path TEXT);
            CREATE TABLE imu_artifact (
                capture_id TEXT, status TEXT, parquet_relative_path TEXT, parquet_sha256 TEXT);
            CREATE TABLE timing_artifact (
                capture_id TEXT, status TEXT, parquet_relative_path TEXT, parquet_sha256 TEXT);
            PRAGMA user_version = 10;
        """)
        database.execute("INSERT INTO capture_snapshot VALUES ('day','take',?,1)", (capture_id,))
        for stream in ("left", "right"):
            database.execute("INSERT INTO capture_member VALUES ('day','take',?,'video',?)",
                             (stream, stream))
            database.execute("INSERT INTO source_file VALUES (?, 'cache/blobs/video')", (stream,))
        database.execute("INSERT INTO imu_artifact VALUES (?, 'decoded', ?, ?)", (
            capture_id, str(imu_path.relative_to(run_dir)), sha256(imu_path.read_bytes()).hexdigest()))
        database.execute("INSERT INTO timing_artifact VALUES (?, 'ready', ?, ?)", (
            capture_id, str(timing_path.relative_to(run_dir)),
            sha256(timing_path.read_bytes()).hexdigest()))
    return capture_id, timing_path


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
    )

    web.decide(capture_id, decision)
    saved = web.episodes()[0][0]["decision"]
    assert saved["status"] == "include"
    assert "decided_by" not in saved

    with pytest.raises(HTTPException, match="Decision changed") as error:
        web.decide(capture_id, decision)
    assert error.value.status_code == 409


def test_web_material_decision_records_controlled_limitation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir, partial=True)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    web = WebRun(source, run_dir, tmp_path / "delivery")
    episode = web.episodes()[0][0]

    web.decide(capture_id, Decision(
        qc_sha256=episode["qc_sha256"], expected_revision="", status="include",
    ))

    assert web.episodes()[0][0]["decision"]["limitations"] == [
        "1 camera frame is outside IMU coverage (single: 1)."
    ]


def test_episode_evidence_contains_graphs_issue_ranges_and_exact_joins(tmp_path):
    run_dir = tmp_path / "run"
    capture_id, _ = evidence_run(run_dir)

    evidence = WebRun(tmp_path / "source", run_dir, tmp_path / "output").evidence(capture_id)

    assert evidence["streams"] == ["left", "right"]
    assert evidence["basis"]["physical_sync_certified"] is False
    assert len(evidence["frames"]["camera_stream_id"]) == 4
    assert evidence["imu_plot"][0]["accel_x_mps2"] == 1.0
    assert [(item["kind"], item["stream"], item["position"], item["frame_count"])
            for item in evidence["issues"]] == [
        ("camera_imu_coverage", "left", "start", 1),
        ("stereo_pairing", "left", "start", 1),
    ]
    assert evidence["issues"][0]["evidence"]["before_delta_ns"] == -100


def test_episode_evidence_rejects_changed_parquet(tmp_path):
    run_dir = tmp_path / "run"
    capture_id, timing_path = evidence_run(run_dir)
    timing_path.write_bytes(b"changed")

    with pytest.raises(HTTPException, match="Verified artifact changed") as error:
        WebRun(tmp_path / "source", run_dir, tmp_path / "output").evidence(capture_id)
    assert error.value.status_code == 409


def test_camera_stream_route_resolves_verified_cached_video(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id, _ = evidence_run(run_dir)
    app = create_app(source, run_dir, tmp_path / "output")
    video = next(route.endpoint for route in app.routes
                 if getattr(route, "path", None) ==
                 "/api/episodes/{capture_id}/video/{stream}")

    response = video(capture_id, "left")

    assert Path(response.path).read_bytes() == b"0123456789"
    with pytest.raises(HTTPException, match="not available"):
        WebRun(source, run_dir, tmp_path / "output").video_path(capture_id, "single")


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
    (source / "take.txt").write_text("raw")
    run_dir = tmp_path / "run"
    run_stage(str(source), run_dir, "inventory")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            "UPDATE processing_item SET status='running' WHERE stage='inventory'"
        )
    (run_dir / "web_job.json").write_text(json.dumps({
        "status": "running", "error": None, "started_at": "2026-08-25T10:00:00Z",
        "finished_at": None, "elapsed_seconds": None, "stage": "inventory",
    }))

    web = WebRun(source, run_dir, tmp_path / "delivery")

    assert web.job()["status"] == "failed"
    assert "interrupted" in web.job()["error"]
    assert json.loads((run_dir / "web_job.json").read_text())["status"] == "failed"
    item = processing_progress(run_dir / "run.sqlite", "inventory")["items"][0]
    assert item["status"] == "interrupted"
    assert "service restarted" in item["error"]


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
    archive = web.build_archive()
    assert web.delivery_summary()["archive_sha256"] == archive["sha256"]
    with zipfile.ZipFile(web.archive) as archive:
        assert all(not name.endswith(".DS_Store") for name in archive.namelist())


def test_changed_archive_is_not_offered_for_download(tmp_path):
    output = tmp_path / "delivery"
    output.mkdir()
    (output / "README.md").write_text("dataset\n")
    run = WebRun(tmp_path / "source", tmp_path / "run", output)
    run.build_archive()

    with run.archive.open("ab") as archive:
        archive.write(b"changed")

    assert run.delivery_summary()["downloadable"] is False


def test_delivery_route_recovers_folder_when_zip_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    web = WebRun(source, run_dir, tmp_path / "unused")
    episode = web.episodes()[0][0]
    web.decide(capture_id, Decision(
        qc_sha256=episode["qc_sha256"], expected_revision="", status="include",
    ))
    approve_stage(run_dir, "review", "")
    output = tmp_path / "delivery"
    output.mkdir()
    (output / "README.md").write_text("dataset\n")

    class InlineThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr("actuate_delivery.web.Thread", InlineThread)
    app = create_app(source, run_dir, output)
    build = next(route.endpoint for route in app.routes
                 if getattr(route, "path", None) == "/api/delivery"
                 and "POST" in route.methods)
    state = next(route.endpoint for route in app.routes
                 if getattr(route, "path", None) == "/api/state")

    assert build() == {"status": "running", "stage": "archive"}
    assert state()["delivery"]["downloadable"] is True
    assert state()["job"]["status"] == "delivered"
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        assert archive.testzip() is None


def test_partial_telemetry_blocks_build_until_browser_choice(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    web = WebRun(source, run_dir, tmp_path / "delivery")
    episode = web.episodes()[0][0]
    web.decide(capture_id, Decision(
        qc_sha256=episode["qc_sha256"], expected_revision="", status="include",
    ))
    approve_stage(run_dir, "review", "")
    selected = {"choice": None}

    def policy(path, choice=None):
        assert path == run_dir
        if choice is not None:
            selected["choice"] = choice
        return {
            "coverage": "partial", "included_episodes": 2,
            "episodes_with_telemetry": 1,
            "requires_choice": selected["choice"] is None,
            "choice": selected["choice"],
        }

    monkeypatch.setattr("actuate_delivery.web.telemetry_policy", policy)
    app = create_app(source, run_dir, tmp_path / "delivery")
    state = next(route.endpoint for route in app.routes
                 if getattr(route, "path", None) == "/api/state")
    build = next(route.endpoint for route in app.routes
                 if getattr(route, "path", None) == "/api/delivery"
                 and "POST" in route.methods)
    choose = next(route.endpoint for route in app.routes
                  if getattr(route, "path", None) == "/api/delivery/telemetry")

    assert state()["telemetry_policy"]["requires_choice"]
    with pytest.raises(HTTPException, match="partial telemetry"):
        build()
    result = choose(TelemetryChoice(choice="exclude_all"))
    assert result["choice"] == "exclude_all"
    assert not state()["telemetry_policy"]["requires_choice"]


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
        "error": "1 video failed", "elapsed_seconds": None,
        "progress": {"completed": 0, "total": 0, "current": None, "items": []},
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
               (capture_id, qc_sha256, status, limitations_json, decided_at)
               SELECT capture_id, json_sha256, 'include', '[]',
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
        "error": None, "elapsed_seconds": 0.0,
        "progress": {"completed": 0, "total": 0, "current": None, "items": []},
    }


def test_web_does_not_project_limitations_for_incomplete_candidates(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"
    capture_id = ready_run(run_dir)
    complete_local_delivery(str(source), run_dir, tmp_path / "unused")
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            "INSERT INTO capture_candidate VALUES ('batch', 'broken', NULL, 'incomplete', 1)"
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
    (run_dir / "review.csv").unlink()
    calls = 0

    def limitations(_):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr("actuate_delivery.web.controlled_limitations", limitations)
    episodes, error = WebRun(source, run_dir, tmp_path / "delivery").episodes()

    assert error is None
    assert calls == 1
    assert [episode["grouping_status"] for episode in episodes] == ["complete", "incomplete"]
    assert episodes[1]["controlled_limitations"] == []


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
        "error": None, "elapsed_seconds": None,
        "progress": {"completed": 0, "total": 0, "current": None, "items": []},
    }
    assert steps[2] == {
        "name": "Verify video", "status": "waiting",
        "detail": "0 of 1 camera files processed; 0 failed",
        "stage": "video", "approved": False, "summary": None, "error": None,
        "elapsed_seconds": None,
        "progress": {"completed": 0, "total": 0, "current": None, "items": []},
    }
    assert web.episodes() == ([], None)
