import csv
import json
import sqlite3
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import trinet_delivery.package as package_module
from trinet_delivery.package import PackageError, build_delivery, project_supplier
from trinet_delivery.qc import build_qc
from trinet_delivery.video import VideoArtifact, VideoError


def internal_qc(tmp_path, layout="single_video", partial=False, incomplete=False):
    stream_ids = ("single",) if layout == "single_video" else ("left", "right")
    members = [{"relative_path": "take.imu", "role": "imu", "camera_stream_id": None,
                "size_bytes": 40, "source_sha256": "1" * 64,
                "cache_relative_path": "cache/blobs/" + "1" * 64}]
    streams = []
    timing_streams = []
    for index, stream_id in enumerate(stream_ids):
        members.extend((
            {"relative_path": f"take_{stream_id}.mp4", "role": "video",
             "camera_stream_id": stream_id, "size_bytes": 100 + index,
             "source_sha256": str(2 + index * 2) * 64,
             "cache_relative_path": "cache/blobs/" + str(2 + index * 2) * 64},
            {"relative_path": f"take_{stream_id}.vts", "role": "vts",
             "camera_stream_id": stream_id, "size_bytes": 20,
             "source_sha256": str(3 + index * 2) * 64,
             "cache_relative_path": "cache/blobs/" + str(3 + index * 2) * 64},
        ))
        coverage = 2 if partial and index == 0 else 3
        streams.append({
            "camera_stream_id": stream_id,
            "vts": {"status": "decoded", "frame_count": 3,
                    "first_timestamp_ns": 100, "last_timestamp_ns": 300},
            "video": {"status": "verified", "frame_count": 3, "codec": "hevc",
                      "width": 1920, "height": 1080, "average_frame_rate": "30/1",
                      "duration_ns": (2 + index) * 1_000_000_000,
                      "audio_stream_count": 1,
                      "probe": {"audio_streams": [{"codec_name": "aac"}]}},
        })
        timing_streams.append({"camera_stream_id": stream_id, "row_count": 3,
                               "matched_rows": 3, "coverage_rows": coverage,
                               "outside_imu_coverage_rows": 3 - coverage,
                               "missing_sof_rows": 0, "video_only_rows": 0,
                               "vts_only_rows": 0})
    facts = {
        "capture_id": "a" * 64, "capture_layout": layout,
        "grouping_status": "incomplete" if incomplete else "complete",
        "source": {"file_count": len(members), "bytes": sum(m["size_bytes"] for m in members),
                   "verified_members": len(members),
                   "all_hashes_verified_in_current_run": True, "members": members},
        "imu": {"status": "decoded", "sample_count": 20, "source_sha256": "1" * 64,
                "native": {"version": 5, "declared_sample_rate_hz": 400,
                           "measured_sample_rate_hz": 399.5, "accel_full_scale_code": 2,
                           "gyro_full_scale_code": 3, "header_start_time_ns": 50,
                           "video_start_time_ns": 0, "flags": 4,
                           "device_id_hex": "ab" * 16, "ios_clock_offset_ns": 0,
                           "reserved_header_hex": "ff" * 28,
                           "first_sample_timestamp_ns": 75,
                           "last_sample_timestamp_ns": 500}},
        "streams": streams, "telemetry": {"status": "absent"},
        "timing": {"status": "ready", "row_count": 3 * len(streams),
                   "matched_rows": 3 * len(streams),
                   "coverage_rows": 3 * len(streams) - int(partial),
                   "stereo_pair_count": 3 if layout == "stereo_pair" else 0,
                   "stereo_unmatched_rows": int(partial), "streams": timing_streams},
    }
    path = tmp_path / f"{layout}-{partial}-{incomplete}.json"
    build_qc(facts, path)
    return json.loads(path.read_text())


def episode(internal, limitations=(), vendor_visualizations=()):
    return {"internal_qc": internal, "source_relative_directory": "batch/device",
            "source_group": "take", "vendor_calibration_references": [],
            "vendor_visualizations": list(vendor_visualizations),
            "decision": {"status": "include", "decided_by": "owner",
                         "decided_at": "2026-08-23T12:00:00Z",
                         "limitations": list(limitations)}}


def read_manifest(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def delivery_input(tmp_path, monkeypatch, visualization=False):
    internal = internal_qc(tmp_path)
    contents = {
        "take.imu": b"native imu bytes",
        "take_single.mp4": b"video bytes",
        "take_single.vts": b"native vts bytes",
    }
    run_dir = tmp_path / "run"
    for member in internal["facts"]["source"]["members"]:
        content = contents[member["relative_path"]]
        source_hash = sha256(content).hexdigest()
        member.update(size_bytes=len(content), source_sha256=source_hash,
                      cache_relative_path=f"cache/blobs/{source_hash}")
        cache = run_dir / member["cache_relative_path"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(content)
    internal["facts"]["source"]["bytes"] = sum(map(len, contents.values()))
    internal["facts"]["imu"]["source_sha256"] = next(
        member["source_sha256"] for member in internal["facts"]["source"]["members"]
        if member["role"] == "imu")
    vendor_visualizations = []
    if visualization:
        content = b"vendor visualization bytes"
        source_hash = sha256(content).hexdigest()
        vendor_visualizations.append({
            "relative_path": "batch/device/visualization.mp4",
            "size_bytes": len(content), "source_sha256": source_hash,
        })
        cache = run_dir / "cache/blobs" / source_hash
        cache.write_bytes(content)
    projection = tmp_path / "projection"
    project_supplier((episode(internal, vendor_visualizations=vendor_visualizations),), projection)

    capture_id = internal["capture_id"]
    work = run_dir / "work" / capture_id
    work.mkdir(parents=True)
    timestamps = [75 + round(index * 425 / 19) for index in range(20)]
    imu_table = pa.table({"timestamp_ns": timestamps}).replace_schema_metadata({
        b"source_sha256": internal["facts"]["imu"]["source_sha256"].encode()})
    imu_path = work / "imu.parquet"
    pq.write_table(imu_table, imu_path)
    timing_path = work / "frame_timing.parquet"
    pq.write_table(pa.table({
        "camera_stream_id": ["single"] * 3,
        "vts_match_status": ["matched"] * 3,
        "mapping_status": ["mapped"] * 3,
    }), timing_path)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.executescript("""
            CREATE TABLE capture_snapshot (
                parent_path TEXT, capture_key TEXT, capture_id TEXT, is_canonical INTEGER);
            CREATE TABLE capture_member (
                parent_path TEXT, capture_key TEXT, source_item_id TEXT, camera_stream_id TEXT);
            CREATE TABLE source_file (
                source_item_id TEXT, relative_path TEXT, parent_path TEXT, role TEXT,
                size_bytes INTEGER, present INTEGER, selected INTEGER, source_sha256 TEXT);
            CREATE TABLE imu_artifact (
                capture_id TEXT, source_sha256 TEXT, status TEXT,
                parquet_relative_path TEXT, parquet_sha256 TEXT, sample_count INTEGER, error TEXT);
            CREATE TABLE timing_artifact (
                capture_id TEXT, input_signature TEXT, status TEXT,
                parquet_relative_path TEXT, parquet_sha256 TEXT, row_count INTEGER,
                matched_rows INTEGER, coverage_rows INTEGER, stereo_pair_count INTEGER,
                stereo_unmatched_rows INTEGER, reason TEXT);
            CREATE TABLE tel_artifact (
                capture_id TEXT, source_sha256 TEXT, status TEXT,
                parquet_relative_path TEXT, parquet_sha256 TEXT, record_count INTEGER, error TEXT);
            PRAGMA user_version = 11;
        """)
        database.execute("INSERT INTO capture_snapshot VALUES ('batch/device', 'take', ?, 1)",
                         (capture_id,))
        for index, member in enumerate(internal["facts"]["source"]["members"]):
            source_item_id = str(index)
            database.execute("INSERT INTO capture_member VALUES ('batch/device', 'take', ?, ?)",
                             (source_item_id, member["camera_stream_id"]))
            database.execute("INSERT INTO source_file VALUES (?, ?, 'batch/device', ?, ?, 1, 1, ?)", (
                source_item_id, member["relative_path"], member["role"],
                member["size_bytes"], member["source_sha256"]))
        for index, member in enumerate(vendor_visualizations, start=100):
            database.execute("INSERT INTO source_file VALUES (?, ?, 'batch/device', 'auxiliary', ?, 1, 1, ?)", (
                str(index), member["relative_path"], member["size_bytes"],
                member["source_sha256"]))
        database.execute("INSERT INTO imu_artifact VALUES (?, ?, 'decoded', ?, ?, 20, NULL)", (
            capture_id, internal["facts"]["imu"]["source_sha256"],
            str(imu_path.relative_to(run_dir)), sha256(imu_path.read_bytes()).hexdigest()))
        database.execute(
            "INSERT INTO timing_artifact VALUES (?, '', 'ready', ?, ?, 3, 3, 3, 0, 0, NULL)", (
                capture_id, str(timing_path.relative_to(run_dir)),
                sha256(timing_path.read_bytes()).hexdigest()))

    def verified_video(source, output, expected_sha256):
        assert sha256(source.read_bytes()).hexdigest() == expected_sha256
        output.write_bytes(b"validated frame index")
        return VideoArtifact("0" * 64, 3, "hevc", 1920, 1080, "30/1",
                             2_000_000_000, 1, "{}")

    monkeypatch.setattr(package_module, "verify_video", verified_video)
    return run_dir, projection, tmp_path / "delivery"


def test_mono_projection_is_episode_first_and_supplier_allowlisted(tmp_path):
    output = tmp_path / "projection"
    included = episode(internal_qc(tmp_path))
    artifact = project_supplier((included,), output)
    repeated = project_supplier((included,), tmp_path / "repeated")
    row = read_manifest(output / "manifest.csv")[0]
    meta = json.loads((output / f"episodes/{row['capture_id']}/meta.json").read_text())
    qc = json.loads((output / row["qc_path"]).read_text())

    assert artifact.capture_count == 1
    assert repeated == artifact
    assert row["capture_layout"] == "single_video"
    assert row["camera_stream_count"] == "1"
    assert row["qc_result"] == "pass"
    assert len(meta["camera_streams"]) == 1
    assert meta["shared_imu"]["declared_sample_rate_hz"] == 400
    assert "reserved_header_hex" not in meta["shared_imu"]
    assert qc["result"] == "pass"
    supplier_text = json.dumps(qc)
    assert "reserved_header" not in supplier_text
    assert "ffmpeg" not in supplier_text
    assert "score" not in supplier_text


def test_stereo_has_one_manifest_row_stream_meta_and_declared_limitation(tmp_path):
    internal = internal_qc(tmp_path, "stereo_pair", partial=True)
    output = tmp_path / "projection"
    project_supplier((episode(internal, ("One left frame lies outside IMU coverage.",)),), output)
    row = read_manifest(output / "manifest.csv")[0]
    meta = json.loads((output / f"episodes/{row['capture_id']}/meta.json").read_text())
    qc = json.loads((output / row["qc_path"]).read_text())

    assert len(read_manifest(output / "manifest.csv")) == 1
    assert row["camera_stream_count"] == "2"
    assert row["camera_duration_min_s"] == "2.0"
    assert row["camera_duration_max_s"] == "3.0"
    assert row["timing_coverage_pct_min"] == "66.666667"
    assert len(meta["camera_streams"]) == 2
    assert qc["result"] == "pass_with_declared_limitation"
    assert qc["stereo"]["association_basis"] == "unique_equal_venc_seq"
    assert "stereo streams are not added together" in (output / "README.md").read_text()


def test_projection_lists_present_vendor_visualization_as_raw(tmp_path):
    internal = internal_qc(tmp_path, "stereo_pair")
    visualization = {"relative_path": "batch/device/visualization.mp4",
                     "size_bytes": 50, "source_sha256": "9" * 64}
    output = tmp_path / "projection"
    project_supplier((episode(internal, vendor_visualizations=(visualization,)),), output)
    row = read_manifest(output / "manifest.csv")[0]
    meta = json.loads((output / f"episodes/{row['capture_id']}/meta.json").read_text())
    qc = json.loads((output / row["qc_path"]).read_text())

    member = next(item for item in meta["raw_members"]
                  if item["role"] == "vendor_visualization")
    assert member == {"role": "vendor_visualization", "camera_stream_id": None,
                      "path": "raw/visualization.mp4", "byte_count": 50,
                      "sha256": "9" * 64}
    assert row["raw_file_count"] == "6"
    assert qc["checks"][0]["evidence"]["raw_file_count"] == 6


def test_vendor_visualization_association_is_unambiguous(tmp_path):
    with sqlite3.connect(tmp_path / "association.sqlite") as database:
        database.row_factory = sqlite3.Row
        database.executescript("""
            CREATE TABLE source_file (
                relative_path TEXT, parent_path TEXT, role TEXT, size_bytes INTEGER,
                source_sha256 TEXT, present INTEGER, selected INTEGER);
            CREATE TABLE capture_snapshot (
                parent_path TEXT, capture_key TEXT, capture_id TEXT, is_canonical INTEGER);
            INSERT INTO source_file VALUES
                ('batch/device/take_stereo_depth_imu.mp4', 'batch/device', 'auxiliary', 10, 'aaaaaaaa', 1, 1),
                ('batch/device/visualization.mp4', 'batch/device', 'auxiliary', 20, 'bbbbbbbb', 1, 1);
            INSERT INTO capture_snapshot VALUES
                ('batch/device', 'take', 'capture-a', 1),
                ('batch/device', 'other', 'capture-b', 0);
        """)
        associated = package_module._vendor_visualizations(database, "batch/device", "take")
        assert [item["relative_path"] for item in associated] == [
            "batch/device/take_stereo_depth_imu.mp4"]
        database.execute("DELETE FROM capture_snapshot WHERE capture_key='other'")
        associated = package_module._vendor_visualizations(database, "batch/device", "take")
        assert [item["relative_path"] for item in associated] == [
            "batch/device/take_stereo_depth_imu.mp4", "batch/device/visualization.mp4"]


def test_decision_blockers_duplicates_and_undeclared_limitations_refuse_projection(tmp_path):
    clean = internal_qc(tmp_path)
    missing_decision = episode(clean)
    missing_decision.pop("decision")
    with pytest.raises(PackageError):
        project_supplier((missing_decision,), tmp_path / "missing")
    with pytest.raises(PackageError, match="duplicate capture ID"):
        project_supplier((episode(clean), episode(clean)), tmp_path / "duplicate")
    with pytest.raises(PackageError, match="capture_structure"):
        project_supplier((episode(internal_qc(tmp_path, incomplete=True)),), tmp_path / "blocked")
    with pytest.raises(PackageError, match="declared limitations"):
        project_supplier((episode(internal_qc(tmp_path, "stereo_pair", partial=True)),),
                         tmp_path / "undeclared")


def test_staged_json_mismatch_publishes_nothing(tmp_path, monkeypatch):
    output = tmp_path / "projection"
    included = episode(internal_qc(tmp_path))
    monkeypatch.setattr(package_module.json, "loads", lambda value: {})

    with pytest.raises(PackageError, match="changed during write"):
        project_supplier((included,), output)

    assert not output.exists()


def test_final_delivery_materializes_and_reopens_every_required_file(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    artifact = build_delivery(run_dir, projection, output)
    row = read_manifest(output / "manifest.csv")[0]
    episode_path = output / "episodes" / row["capture_id"]

    assert artifact.capture_count == 1
    assert artifact.file_count == 9
    assert sorted(path.name for path in (episode_path / "raw").iterdir()) == [
        "take.imu", "take_single.mp4", "take_single.vts"]
    assert (episode_path / "derived/imu.parquet").is_file()
    assert (episode_path / "derived/frame_timing.parquet").is_file()
    assert json.loads((episode_path / "derived/qc.json").read_text())["result"] == "pass"
    assert not any(path.is_symlink() for path in output.rglob("*"))


def test_final_delivery_accepts_decision_schema_12(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute("PRAGMA user_version = 12")

    assert build_delivery(run_dir, projection, output).capture_count == 1


def test_vendor_visualization_is_delivered_and_decode_failure_blocks_output(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch, visualization=True)
    artifact = build_delivery(run_dir, projection, output)
    row = read_manifest(output / "manifest.csv")[0]
    episode_path = output / "episodes" / row["capture_id"]

    assert artifact.file_count == 10
    assert row["raw_file_count"] == "4"
    assert (episode_path / "raw/visualization.mp4").read_bytes() == b"vendor visualization bytes"

    run_dir, projection, output = delivery_input(tmp_path / "decode", monkeypatch,
                                                  visualization=True)
    verified_video = package_module.verify_video

    def fail_visualization(source, index, expected_sha256):
        if source.name == "visualization.mp4":
            raise VideoError("broken visualization")
        return verified_video(source, index, expected_sha256)

    monkeypatch.setattr(package_module, "verify_video", fail_visualization)
    with pytest.raises(PackageError, match="broken visualization"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_changed_vendor_visualization_or_omission_blocks_output(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch, visualization=True)
    visual = next(path for path in (run_dir / "cache/blobs").iterdir()
                  if path.read_bytes() == b"vendor visualization bytes")
    visual.write_bytes(b"x" * visual.stat().st_size)
    with pytest.raises(PackageError, match="SHA-256"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()

    run_dir, projection, output = delivery_input(tmp_path / "omitted", monkeypatch,
                                                  visualization=True)
    meta_path = next(projection.rglob("meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["raw_members"] = [member for member in meta["raw_members"]
                           if member["role"] != "vendor_visualization"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PackageError, match="run ledger"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_corrupt_raw_or_derived_input_exposes_no_delivery(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    raw = next(path for path in (run_dir / "cache/blobs").iterdir()
               if path.read_bytes() == b"native imu bytes")
    raw.write_bytes(b"x" * raw.stat().st_size)
    with pytest.raises(PackageError, match="SHA-256"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()

    run_dir, projection, output = delivery_input(tmp_path / "derived", monkeypatch)
    imu = next((run_dir / "work").rglob("imu.parquet"))
    original_hash = sha256(imu.read_bytes()).hexdigest()
    pq.write_table(pq.read_table(imu), imu, compression="gzip")
    assert sha256(imu.read_bytes()).hexdigest() != original_hash
    with pytest.raises(PackageError, match="SHA-256"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_changed_projection_or_parquet_facts_expose_no_delivery(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    meta_path = next(projection.rglob("meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["capture_id"] = "b" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PackageError, match="cross-reference"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()

    run_dir, projection, output = delivery_input(tmp_path / "members", monkeypatch)
    meta_path = next(projection.rglob("meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["raw_members"][0]["role"] = "video"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PackageError, match="run ledger"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()

    run_dir, projection, output = delivery_input(tmp_path / "parquet", monkeypatch)
    imu = next((run_dir / "work").rglob("imu.parquet"))
    table = pq.read_table(imu)
    pq.write_table(table.set_column(0, "timestamp_ns", pa.array(range(20))), imu)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute("UPDATE imu_artifact SET parquet_sha256=?",
                         (sha256(imu.read_bytes()).hexdigest(),))
    with pytest.raises(PackageError, match="IMU facts"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_preferred_video_fact_mismatch_exposes_no_delivery(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)

    def wrong_video(source, index, expected_sha256):
        index.write_bytes(b"wrong frame index")
        return VideoArtifact("0" * 64, 4, "hevc", 1920, 1080, "30/1",
                             2_000_000_000, 1, "{}")

    monkeypatch.setattr(package_module, "verify_video", wrong_video)
    with pytest.raises(PackageError, match="video facts"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_unexpected_staged_file_exposes_no_delivery(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    materialize = package_module._materialize

    def materialize_with_extra(source, target, expected_hash, expected_size=None):
        materialize(source, target, expected_hash, expected_size)
        (target.parent / "unexpected").write_bytes(b"unexpected")

    monkeypatch.setattr(package_module, "_materialize", materialize_with_extra)
    with pytest.raises(PackageError, match="file inventory"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()
