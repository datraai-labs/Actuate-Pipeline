import csv
import json
import sqlite3
from hashlib import sha256

import actuate_delivery.package as package_module
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from actuate_delivery.package import PackageError, build_delivery, project_supplier
from actuate_delivery.qc import build_qc
from actuate_delivery.video import VideoArtifact, VideoError


def internal_qc(tmp_path, layout="single_video", partial=False, incomplete=False):
    stream_ids = ("single",) if layout == "single_video" else ("left", "right")
    members = [
        {
            "relative_path": "take.imu",
            "role": "imu",
            "camera_stream_id": None,
            "size_bytes": 40,
            "source_sha256": "1" * 64,
            "cache_relative_path": "cache/blobs/" + "1" * 64,
        }
    ]
    streams = []
    timing_streams = []
    for index, stream_id in enumerate(stream_ids):
        members.extend(
            (
                {
                    "relative_path": f"take_{stream_id}.mp4",
                    "role": "video",
                    "camera_stream_id": stream_id,
                    "size_bytes": 100 + index,
                    "source_sha256": str(2 + index * 2) * 64,
                    "cache_relative_path": "cache/blobs/" + str(2 + index * 2) * 64,
                },
                {
                    "relative_path": f"take_{stream_id}.vts",
                    "role": "vts",
                    "camera_stream_id": stream_id,
                    "size_bytes": 20,
                    "source_sha256": str(3 + index * 2) * 64,
                    "cache_relative_path": "cache/blobs/" + str(3 + index * 2) * 64,
                },
            )
        )
        coverage = 2 if partial and index == 0 else 3
        streams.append(
            {
                "camera_stream_id": stream_id,
                "vts": {
                    "status": "decoded",
                    "frame_count": 3,
                    "first_timestamp_ns": 100,
                    "last_timestamp_ns": 300,
                },
                "video": {
                    "status": "verified",
                    "frame_count": 3,
                    "codec": "hevc",
                    "width": 1920,
                    "height": 1080,
                    "average_frame_rate": "30/1",
                    "duration_ns": (2 + index) * 1_000_000_000,
                    "audio_stream_count": 1,
                    "probe": {"audio_streams": [{"codec_name": "aac"}]},
                },
            }
        )
        timing_streams.append(
            {
                "camera_stream_id": stream_id,
                "row_count": 3,
                "matched_rows": 3,
                "coverage_rows": coverage,
                "outside_imu_coverage_rows": 3 - coverage,
                "missing_sof_rows": 0,
                "video_only_rows": 0,
                "vts_only_rows": 0,
                "outside_imu_coverage_ranges": (
                    [
                        {
                            "position": "start",
                            "frame_count": 1,
                            "start_frame": 0,
                            "end_frame": 0,
                            "start_time_s": 0.0,
                            "end_time_s": 0.0,
                        }
                    ]
                    if partial and index == 0
                    else []
                ),
                "video_vts_mismatch_ranges": [],
                "stereo_unmatched_ranges": (
                    [
                        {
                            "position": "start",
                            "frame_count": 1,
                            "start_frame": 0,
                            "end_frame": 0,
                            "start_time_s": 0.0,
                            "end_time_s": 0.0,
                        }
                    ]
                    if partial and index == 0
                    else []
                ),
            }
        )
    facts = {
        "capture_id": "a" * 64,
        "capture_layout": layout,
        "grouping_status": "incomplete" if incomplete else "complete",
        "source": {
            "file_count": len(members),
            "bytes": sum(m["size_bytes"] for m in members),
            "verified_members": len(members),
            "all_hashes_verified_in_current_run": True,
            "members": members,
        },
        "imu": {
            "status": "decoded",
            "sample_count": 20,
            "source_sha256": "1" * 64,
            "native": {
                "version": 5,
                "declared_sample_rate_hz": 400,
                "measured_sample_rate_hz": 399.5,
                "accel_full_scale_code": 2,
                "gyro_full_scale_code": 3,
                "header_start_time_ns": 50,
                "video_start_time_ns": 0,
                "flags": 4,
                "device_id_hex": "ab" * 16,
                "ios_clock_offset_ns": 0,
                "reserved_header_hex": "ff" * 28,
                "first_sample_timestamp_ns": 75,
                "last_sample_timestamp_ns": 500,
            },
        },
        "streams": streams,
        "telemetry": {"status": "absent"},
        "timing": {
            "status": "ready",
            "row_count": 3 * len(streams),
            "matched_rows": 3 * len(streams),
            "coverage_rows": 3 * len(streams) - int(partial),
            "stereo_pair_count": 3 if layout == "stereo_pair" else 0,
            "stereo_unmatched_rows": int(partial),
            "streams": timing_streams,
        },
    }
    path = tmp_path / f"{layout}-{partial}-{incomplete}.json"
    build_qc(facts, path)
    return json.loads(path.read_text())


def episode(internal, limitations=(), vendor_visualizations=(), episode_id="episode_000001"):
    return {
        "episode_id": episode_id,
        "internal_qc": internal,
        "source_relative_directory": "batch/device",
        "source_group": "take",
        "vendor_visualizations": list(vendor_visualizations),
        "decision": {
            "status": "include",
            "decided_at": "2026-08-23T12:00:00Z",
            "limitations": list(limitations),
        },
    }


def add_telemetry(internal, value="6"):
    source_hash = value * 64
    member = {
        "relative_path": "take.tel",
        "role": "telemetry",
        "camera_stream_id": None,
        "size_bytes": 12,
        "source_sha256": source_hash,
        "cache_relative_path": f"cache/blobs/{source_hash}",
    }
    internal["facts"]["source"]["members"].append(member)
    internal["facts"]["source"]["file_count"] += 1
    internal["facts"]["source"]["bytes"] += 12
    internal["facts"]["source"]["verified_members"] += 1
    internal["facts"]["telemetry"] = {
        "status": "decoded",
        "record_count": 4,
        "source_sha256": source_hash,
    }
    return internal


def calibration(episode_ids=("episode_000001",)):
    return {
        "schema_version": "actuate_delivery.calibration.v1",
        "calibration_id": "calibration_000001",
        "rig_id": "rig_000001",
        "applies_to_episode_ids": list(episode_ids),
        "applied_by_pipeline": False,
        "cameras": [{"stream_id": "left"}, {"stream_id": "right"}],
        "transforms": {"baseline_m": 0.07},
        "imu": {"update_rate_hz": 399.2},
    }


def read_manifest(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def delivery_input(tmp_path, monkeypatch, visualization=False, dataset_calibration=None):
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
        member.update(
            size_bytes=len(content),
            source_sha256=source_hash,
            cache_relative_path=f"cache/blobs/{source_hash}",
        )
        cache = run_dir / member["cache_relative_path"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(content)
    internal["facts"]["source"]["bytes"] = sum(map(len, contents.values()))
    internal["facts"]["imu"]["source_sha256"] = next(
        member["source_sha256"]
        for member in internal["facts"]["source"]["members"]
        if member["role"] == "imu"
    )
    vendor_visualizations = []
    if visualization:
        content = b"vendor visualization bytes"
        source_hash = sha256(content).hexdigest()
        vendor_visualizations.append(
            {
                "relative_path": "batch/device/visualization.mp4",
                "size_bytes": len(content),
                "source_sha256": source_hash,
            }
        )
        cache = run_dir / "cache/blobs" / source_hash
        cache.write_bytes(content)
    projection = tmp_path / "projection"
    project_supplier(
        (episode(internal, vendor_visualizations=vendor_visualizations),),
        projection,
        dataset_calibration,
    )

    capture_id = internal["capture_id"]
    work = run_dir / "work" / capture_id
    work.mkdir(parents=True)
    timestamps = [75 + round(index * 425 / 19) for index in range(20)]
    imu_table = pa.table({"timestamp_ns": timestamps}).replace_schema_metadata(
        {b"source_sha256": internal["facts"]["imu"]["source_sha256"].encode()}
    )
    imu_path = work / "imu.parquet"
    pq.write_table(imu_table, imu_path)
    timing_path = work / "frame_timing.parquet"
    pq.write_table(
        pa.table(
            {
                "camera_stream_id": ["single"] * 3,
                "video_frame_index": [0, 1, 2],
                "mp4_pts_ns": [0, 1_000_000_000, 2_000_000_000],
                "vts_match_status": ["matched"] * 3,
                "mapping_status": ["mapped"] * 3,
                "stereo_pair_status": ["not_applicable"] * 3,
            }
        ),
        timing_path,
    )
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
            CREATE TABLE delivery_episode (
                capture_id TEXT PRIMARY KEY, episode_number INTEGER NOT NULL UNIQUE);
            CREATE TABLE delivery_decision (
                capture_id TEXT PRIMARY KEY, status TEXT NOT NULL);
            PRAGMA user_version = 15;
        """)
        database.execute(
            "INSERT INTO capture_snapshot VALUES ('batch/device', 'take', ?, 1)", (capture_id,)
        )
        database.execute("INSERT INTO delivery_episode VALUES (?, 1)", (capture_id,))
        database.execute("INSERT INTO delivery_decision VALUES (?, 'include')", (capture_id,))
        for index, member in enumerate(internal["facts"]["source"]["members"]):
            source_item_id = str(index)
            database.execute(
                "INSERT INTO capture_member VALUES ('batch/device', 'take', ?, ?)",
                (source_item_id, member["camera_stream_id"]),
            )
            database.execute(
                "INSERT INTO source_file VALUES (?, ?, 'batch/device', ?, ?, 1, 1, ?)",
                (
                    source_item_id,
                    member["relative_path"],
                    member["role"],
                    member["size_bytes"],
                    member["source_sha256"],
                ),
            )
        for index, member in enumerate(vendor_visualizations, start=100):
            database.execute(
                "INSERT INTO source_file VALUES (?, ?, 'batch/device', 'auxiliary', ?, 1, 1, ?)",
                (
                    str(index),
                    member["relative_path"],
                    member["size_bytes"],
                    member["source_sha256"],
                ),
            )
        database.execute(
            "INSERT INTO imu_artifact VALUES (?, ?, 'decoded', ?, ?, 20, NULL)",
            (
                capture_id,
                internal["facts"]["imu"]["source_sha256"],
                str(imu_path.relative_to(run_dir)),
                sha256(imu_path.read_bytes()).hexdigest(),
            ),
        )
        database.execute(
            "INSERT INTO timing_artifact VALUES (?, '', 'ready', ?, ?, 3, 3, 3, 0, 0, NULL)",
            (
                capture_id,
                str(timing_path.relative_to(run_dir)),
                sha256(timing_path.read_bytes()).hexdigest(),
            ),
        )

    def verified_video(source, output, expected_sha256):
        assert sha256(source.read_bytes()).hexdigest() == expected_sha256
        output.write_bytes(b"validated frame index")
        return VideoArtifact("0" * 64, 3, "hevc", 1920, 1080, "30/1", 2_000_000_000, 1, "{}")

    monkeypatch.setattr(package_module, "verify_video", verified_video)
    return run_dir, projection, tmp_path / "delivery"


def test_mono_projection_is_episode_first_and_supplier_allowlisted(tmp_path):
    output = tmp_path / "projection"
    included = episode(internal_qc(tmp_path))
    artifact = project_supplier((included,), output)
    repeated = project_supplier((included,), tmp_path / "repeated")
    row = read_manifest(output / "episodes.csv")[0]
    meta = json.loads((output / f"episodes/{row['episode_id']}/meta.json").read_text())
    qc = json.loads((output / f"episodes/{row['episode_id']}/derived/qc.json").read_text())

    assert artifact.capture_count == 1
    assert repeated == artifact
    assert row["episode_id"] == "episode_000001"
    assert meta["episode_id"] == "episode_000001"
    assert "capture_id" not in meta
    assert "capture_id" not in row
    assert row["capture_layout"] == "single_video"
    assert row["camera_stream_count"] == "1"
    assert "qc_result" not in row
    assert "limitation_count" not in row
    assert len(meta["camera_streams"]) == 1
    assert meta["shared_imu"]["declared_sample_rate_hz"] == 400
    assert "reserved_header_hex" not in meta["shared_imu"]
    assert qc["schema_version"] == "actuate_delivery.qc_facts.v1"
    assert qc["episode_id"] == "episode_000001"
    assert "result" not in qc
    assert "previews" not in meta
    assert "rig_id" not in meta
    assert "calibration_id" not in meta
    supplier_text = json.dumps(qc)
    assert "reserved_header" not in supplier_text
    assert "ffmpeg" not in supplier_text
    assert "score" not in supplier_text


def test_stereo_has_one_manifest_row_stream_meta_and_declared_limitation(tmp_path):
    internal = internal_qc(tmp_path, "stereo_pair", partial=True)
    output = tmp_path / "projection"
    project_supplier((episode(internal, ("One left frame lies outside IMU coverage.",)),), output)
    row = read_manifest(output / "episodes.csv")[0]
    meta = json.loads((output / f"episodes/{row['episode_id']}/meta.json").read_text())
    qc = json.loads((output / f"episodes/{row['episode_id']}/derived/qc.json").read_text())

    assert len(read_manifest(output / "episodes.csv")) == 1
    assert row["camera_stream_count"] == "2"
    assert row["camera_duration_min_s"] == "2.0"
    assert row["camera_duration_max_s"] == "3.0"
    assert row["timing_coverage_pct_min"] == "66.666667"
    assert len(meta["camera_streams"]) == 2
    assert "result" not in qc
    assert (
        qc["camera_streams"][0]["timing"]["outside_imu_coverage_ranges"][0]["position"] == "start"
    )
    assert qc["stereo"]["association_basis"] == "unique_equal_venc_seq"
    assert "stereo streams are not added together" in (output / "README.md").read_text()


def test_projection_lists_present_visualization_as_neutral_preview(tmp_path):
    internal = internal_qc(tmp_path, "stereo_pair")
    visualization = {
        "relative_path": "batch/device/visualization.mp4",
        "size_bytes": 50,
        "source_sha256": "9" * 64,
    }
    output = tmp_path / "projection"
    project_supplier((episode(internal, vendor_visualizations=(visualization,)),), output)
    row = read_manifest(output / "episodes.csv")[0]
    meta = json.loads((output / f"episodes/{row['episode_id']}/meta.json").read_text())

    assert meta["previews"] == [
        {"path": "previews/capture_preview.mp4", "byte_count": 50, "sha256": "9" * 64}
    ]
    assert all(member["role"] != "vendor_visualization" for member in meta["raw_members"])
    assert row["raw_file_count"] == "5"
    assert row["preview_count"] == "1"


def test_partial_telemetry_requires_policy_and_filters_raw_and_derived_together(tmp_path):
    first = add_telemetry(internal_qc(tmp_path))
    second = internal_qc(tmp_path)
    second["capture_id"] = second["facts"]["capture_id"] = "b" * 64
    episodes = (episode(first), episode(second, episode_id="episode_000002"))

    with pytest.raises(PackageError, match="Partial telemetry"):
        project_supplier(episodes, tmp_path / "automatic")

    included = tmp_path / "included"
    project_supplier(episodes, included, telemetry_mode="include_available")
    first_meta = json.loads((included / "episodes/episode_000001/meta.json").read_text())
    second_meta = json.loads((included / "episodes/episode_000002/meta.json").read_text())
    first_qc = json.loads((included / "episodes/episode_000001/derived/qc.json").read_text())
    assert first_meta["telemetry"] == {"record_count": 4}
    assert any(member["role"] == "telemetry" for member in first_meta["raw_members"])
    assert first_qc["telemetry"] == {"record_count": 4}
    assert "telemetry" not in second_meta
    assert "1 of 2 episodes" in (included / "README.md").read_text()

    excluded = tmp_path / "excluded"
    project_supplier(episodes, excluded, telemetry_mode="exclude_all")
    assert all("telemetry" not in json.loads(path.read_text()) for path in excluded.rglob("*.json"))
    assert all(
        member["role"] != "telemetry"
        for path in excluded.rglob("meta.json")
        for member in json.loads(path.read_text())["raw_members"]
    )


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
            "batch/device/take_stereo_depth_imu.mp4"
        ]
        database.execute("DELETE FROM capture_snapshot WHERE capture_key='other'")
        associated = package_module._vendor_visualizations(database, "batch/device", "take")
        assert [item["relative_path"] for item in associated] == [
            "batch/device/take_stereo_depth_imu.mp4",
            "batch/device/visualization.mp4",
        ]


def test_decision_blockers_and_duplicates_refuse_projection(tmp_path):
    clean = internal_qc(tmp_path)
    missing_decision = episode(clean)
    missing_decision.pop("decision")
    with pytest.raises(PackageError):
        project_supplier((missing_decision,), tmp_path / "missing")
    with pytest.raises(PackageError, match="duplicate capture ID"):
        project_supplier((episode(clean), episode(clean)), tmp_path / "duplicate")
    with pytest.raises(PackageError, match="capture_structure"):
        project_supplier((episode(internal_qc(tmp_path, incomplete=True)),), tmp_path / "blocked")
    project_supplier(
        (episode(internal_qc(tmp_path, "stereo_pair", partial=True)),), tmp_path / "reviewed-facts"
    )


def test_projection_rejects_duplicate_or_invalid_public_episode_ids(tmp_path):
    first = internal_qc(tmp_path)
    second = json.loads(json.dumps(first))
    second["capture_id"] = "b" * 64
    second["facts"]["capture_id"] = "b" * 64

    with pytest.raises(PackageError, match="duplicate episode ID"):
        project_supplier((episode(first), episode(second)), tmp_path / "duplicate-public-id")
    with pytest.raises(PackageError, match="invalid episode ID"):
        project_supplier((episode(first, episode_id="take-a"),), tmp_path / "invalid-public-id")


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
    row = read_manifest(output / "episodes.csv")[0]
    episode_path = output / "episodes" / row["episode_id"]

    assert artifact.capture_count == 1
    assert artifact.file_count == 9
    assert sorted(path.name for path in (episode_path / "raw").iterdir()) == [
        "take.imu",
        "take_single.mp4",
        "take_single.vts",
    ]
    assert (episode_path / "derived/imu.parquet").is_file()
    assert (episode_path / "derived/frame_timing.parquet").is_file()
    assert (episode_path / "derived/qc.json").is_file()
    assert not any(path.is_symlink() for path in output.rglob("*"))


def test_final_delivery_requires_episode_identity_schema(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    assert build_delivery(run_dir, projection, output).capture_count == 1


def test_final_delivery_includes_bound_calibration_and_customer_qc(tmp_path, monkeypatch):
    expected = calibration()
    run_dir, projection, output = delivery_input(
        tmp_path, monkeypatch, dataset_calibration=expected
    )

    artifact = build_delivery(run_dir, projection, output)
    row = read_manifest(output / "episodes.csv")[0]
    meta = json.loads((output / "episodes/episode_000001/meta.json").read_text())

    assert artifact.file_count == 10
    assert json.loads((output / "calibration/calibration_000001.json").read_text()) == expected
    assert "rig_id" not in row
    assert "calibration_id" not in row
    assert meta["rig_id"] == "rig_000001"
    assert meta["calibration_id"] == "calibration_000001"
    customer_text = "\n".join(
        path.read_text()
        for path in output.rglob("*")
        if path.is_file() and path.suffix in (".csv", ".json", ".md")
    )
    assert "pass_with_declared_limitation" not in customer_text
    assert "quality_notes" not in customer_text
    assert "qc_result" not in customer_text


def test_calibration_requires_exact_episode_binding(tmp_path):
    included = episode(internal_qc(tmp_path))
    wrong = calibration(("episode_000002",))

    with pytest.raises(PackageError, match="episode binding"):
        project_supplier((included,), tmp_path / "projection", wrong)


def test_calibration_rejects_superseded_product_schema(tmp_path):
    included = episode(internal_qc(tmp_path))
    old = calibration()
    old["schema_version"] = "trinet_delivery.calibration.v1"

    with pytest.raises(PackageError, match="schema is unsupported"):
        project_supplier((included,), tmp_path / "projection", old)


def test_vendor_visualization_is_delivered_and_decode_failure_blocks_output(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch, visualization=True)
    artifact = build_delivery(run_dir, projection, output)
    row = read_manifest(output / "episodes.csv")[0]
    episode_path = output / "episodes" / row["episode_id"]

    assert artifact.file_count == 10
    assert row["raw_file_count"] == "3"
    assert row["preview_count"] == "1"
    assert (
        episode_path / "previews/capture_preview.mp4"
    ).read_bytes() == b"vendor visualization bytes"

    run_dir, projection, output = delivery_input(
        tmp_path / "decode", monkeypatch, visualization=True
    )
    verified_video = package_module.verify_video

    def fail_visualization(source, index, expected_sha256):
        if source.name == "capture_preview.mp4":
            raise VideoError("broken visualization")
        return verified_video(source, index, expected_sha256)

    monkeypatch.setattr(package_module, "verify_video", fail_visualization)
    with pytest.raises(PackageError, match="broken visualization"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_changed_vendor_visualization_or_omission_blocks_output(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch, visualization=True)
    visual = next(
        path
        for path in (run_dir / "cache/blobs").iterdir()
        if path.read_bytes() == b"vendor visualization bytes"
    )
    visual.write_bytes(b"x" * visual.stat().st_size)
    with pytest.raises(PackageError, match="SHA-256"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()

    run_dir, projection, output = delivery_input(
        tmp_path / "omitted", monkeypatch, visualization=True
    )
    meta_path = next(projection.rglob("meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["previews"] = []
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PackageError, match="run ledger"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_corrupt_raw_or_derived_input_exposes_no_delivery(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)
    raw = next(
        path
        for path in (run_dir / "cache/blobs").iterdir()
        if path.read_bytes() == b"native imu bytes"
    )
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
    qc_path = next(projection.rglob("qc.json"))
    qc = json.loads(qc_path.read_text())
    qc["episode_id"] = "episode_000002"
    qc_path.write_text(json.dumps(qc))
    with pytest.raises(PackageError, match="cross-reference"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()

    run_dir, projection, output = delivery_input(tmp_path / "episode-id", monkeypatch)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute("UPDATE delivery_episode SET episode_number=2")
    with pytest.raises(PackageError, match="episode ID is not in the run ledger"):
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
        database.execute(
            "UPDATE imu_artifact SET parquet_sha256=?", (sha256(imu.read_bytes()).hexdigest(),)
        )
    with pytest.raises(PackageError, match="IMU facts"):
        build_delivery(run_dir, projection, output)
    assert not output.exists()


def test_preferred_video_fact_mismatch_exposes_no_delivery(tmp_path, monkeypatch):
    run_dir, projection, output = delivery_input(tmp_path, monkeypatch)

    def wrong_video(source, index, expected_sha256):
        index.write_bytes(b"wrong frame index")
        return VideoArtifact("0" * 64, 4, "hevc", 1920, 1080, "30/1", 2_000_000_000, 1, "{}")

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
