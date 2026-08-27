import json
from hashlib import sha256

import actuate_delivery.qc as qc_module
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from actuate_delivery.qc import QcError, build_qc, timing_stream_facts


def facts(layout="single_video"):
    stream_ids = ("single",) if layout == "single_video" else ("left", "right")
    streams = [{
        "camera_stream_id": stream_id,
        "vts": {"status": "decoded", "frame_count": 3},
        "video": {"status": "verified", "frame_count": 3},
    } for stream_id in stream_ids]
    return {
        "capture_id": "a" * 64,
        "capture_layout": layout,
        "grouping_status": "complete",
        "source": {"file_count": 4, "bytes": 100, "verified_members": 4,
                   "all_hashes_verified_in_current_run": True, "members": []},
        "imu": {"status": "decoded", "sample_count": 20},
        "streams": streams,
        "telemetry": {"status": "absent"},
        "timing": {"status": "ready", "row_count": 3 * len(streams),
                   "matched_rows": 3 * len(streams), "coverage_rows": 3 * len(streams),
                   "stereo_pair_count": 3 if layout == "stereo_pair" else 0,
                   "stereo_unmatched_rows": 0,
                   "streams": [{"camera_stream_id": stream_id, "row_count": 3,
                                "matched_rows": 3, "coverage_rows": 3,
                                "outside_imu_coverage_rows": 0, "missing_sof_rows": 0,
                                "video_only_rows": 0, "vts_only_rows": 0}
                               for stream_id in stream_ids]},
    }


def results(path):
    document = json.loads(path.read_text())
    return {check["check_id"]: check for check in document["checks"]}, document["summary"]


def test_valid_mono_has_no_score_and_telemetry_is_not_applicable(tmp_path):
    output = tmp_path / "qc.json"
    artifact = build_qc(facts(), output)
    checks, summary = results(output)

    assert checks["camera_imu_coverage"]["result"] == "pass"
    assert checks["stereo_sequence_pairing"]["result"] == "not_applicable"
    assert checks["telemetry"]["result"] == "not_applicable"
    assert "score" not in output.read_text()
    assert artifact.not_applicable_count == summary["not_applicable"] == 2


def test_mismatched_frames_partial_coverage_and_unmatched_stereo_fail(tmp_path):
    input_facts = facts("stereo_pair")
    input_facts["streams"][0]["video"]["frame_count"] = 4
    input_facts["timing"].update(row_count=7, coverage_rows=5, stereo_unmatched_rows=1)
    input_facts["timing"]["streams"][0].update(
        coverage_rows=2, outside_imu_coverage_rows=1)
    output = tmp_path / "qc.json"
    artifact = build_qc(input_facts, output)
    checks, _ = results(output)

    assert checks["video_vts_frame_count:left"]["result"] == "fail"
    assert checks["camera_imu_coverage"] == {
        "check_id": "camera_imu_coverage", "result": "fail",
        "evidence": {"matched_camera_rows": 6, "rows_within_imu_coverage": 5,
                     "rows_outside_imu_coverage": 1},
    }
    assert checks["stereo_sequence_pairing"]["result"] == "fail"
    assert checks["camera_imu_coverage:left"]["result"] == "fail"
    assert checks["camera_imu_coverage:right"]["result"] == "pass"
    assert artifact.fail_count == 4


def test_missing_and_failed_upstream_evidence_stays_distinct(tmp_path):
    input_facts = facts()
    input_facts["imu"] = {"status": "failed", "error": "bad native bytes"}
    input_facts["streams"][0]["vts"] = {"status": "missing"}
    input_facts["timing"] = {"status": "unavailable", "reason": "VTS unavailable"}
    output = tmp_path / "qc.json"
    build_qc(input_facts, output)
    checks, _ = results(output)

    assert checks["imu_decode"]["result"] == "fail"
    assert checks["vts_decode:single"]["result"] == "unknown"
    assert checks["video_vts_frame_count:single"]["result"] == "unknown"
    assert checks["timing_artifact"]["result"] == "unknown"


def test_zero_matched_rows_make_coverage_unknown(tmp_path):
    input_facts = facts()
    input_facts["timing"].update(matched_rows=0, coverage_rows=0)
    output = tmp_path / "qc.json"
    build_qc(input_facts, output)

    assert results(output)[0]["camera_imu_coverage"]["result"] == "unknown"


def test_failed_staging_reverification_publishes_nothing(tmp_path, monkeypatch):
    output = tmp_path / "qc.json"
    monkeypatch.setattr(qc_module.json, "loads", lambda value: {})

    with pytest.raises(QcError, match="does not match"):
        build_qc(facts(), output)

    assert not output.exists()
    assert not (tmp_path / ".qc.json.staging").exists()


def test_timing_stream_facts_reverify_and_count_each_stream(tmp_path):
    path = tmp_path / "timing.parquet"
    pq.write_table(pa.table({
        "camera_stream_id": ["left", "left", "right"],
        "video_frame_index": [0, 1, 0],
        "mp4_pts_ns": [0, 33_000_000, 0],
        "vts_match_status": ["matched", "video_only", "matched"],
        "mapping_status": ["outside_imu_coverage", "no_vts", "mapped"],
        "stereo_pair_status": ["unmatched", "unmatched", "matched"],
    }), path)
    source_hash = sha256(path.read_bytes()).hexdigest()

    assert timing_stream_facts(path, source_hash) == [
        {"camera_stream_id": "left", "row_count": 2, "matched_rows": 1,
         "coverage_rows": 0, "outside_imu_coverage_rows": 1, "missing_sof_rows": 0,
         "video_only_rows": 1, "vts_only_rows": 0,
         "outside_imu_coverage_ranges": [{"position": "start", "frame_count": 1,
                                           "start_frame": 0, "end_frame": 0,
                                           "start_time_s": 0.0, "end_time_s": 0.0}],
         "video_vts_mismatch_ranges": [{"position": "end", "frame_count": 1,
                                         "start_frame": 1, "end_frame": 1,
                                         "start_time_s": 0.033, "end_time_s": 0.033}],
         "stereo_unmatched_ranges": [{"position": "start", "frame_count": 2,
                                       "start_frame": 0, "end_frame": 1,
                                       "start_time_s": 0.0, "end_time_s": 0.033}]},
        {"camera_stream_id": "right", "row_count": 1, "matched_rows": 1,
         "coverage_rows": 1, "outside_imu_coverage_rows": 0, "missing_sof_rows": 0,
         "video_only_rows": 0, "vts_only_rows": 0,
         "outside_imu_coverage_ranges": [], "video_vts_mismatch_ranges": [],
         "stereo_unmatched_ranges": []},
    ]
    with pytest.raises(QcError, match="SHA-256"):
        timing_stream_facts(path, "0" * 64)
