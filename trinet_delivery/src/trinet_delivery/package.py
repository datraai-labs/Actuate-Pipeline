import csv
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trinet_delivery.qc import QcError, _supplier_qc, timing_stream_facts
from trinet_delivery.video import VideoError, verify_video


class PackageError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectionArtifact:
    capture_count: int
    manifest_sha256: str


@dataclass(frozen=True)
class DeliveryArtifact:
    capture_count: int
    file_count: int
    byte_count: int
    manifest_sha256: str


MANIFEST_FIELDS = (
    "capture_id", "source_relative_directory", "source_group", "capture_layout",
    "camera_stream_count", "raw_file_count", "raw_bytes", "device_id", "imu_sha256",
    "tel_sha256", "imu_samples", "imu_rate_hz", "camera_duration_min_s",
    "camera_duration_max_s", "timing_coverage_pct_min", "qc_result",
    "limitation_count", "qc_path",
)

META_IMU_FIELDS = (
    "version", "declared_sample_rate_hz", "measured_sample_rate_hz",
    "accel_full_scale_code", "gyro_full_scale_code", "header_start_time_ns",
    "video_start_time_ns", "flags", "device_id_hex", "ios_clock_offset_ns",
    "first_sample_timestamp_ns", "last_sample_timestamp_ns",
)


def _percentage(part, whole):
    return None if not whole else round(part * 100 / whole, 6)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _materialize(source: Path, target: Path, expected_hash: str, expected_size=None) -> None:
    if not source.is_file() or source.is_symlink():
        raise PackageError(f"Package input is not a regular file: {source}")
    if expected_size is not None and source.stat().st_size != expected_size:
        raise PackageError(f"Package input size changed: {source}")
    if _file_hash(source) != expected_hash:
        raise PackageError(f"Package input SHA-256 changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_mode & 0o222:
        shutil.copyfile(source, target)
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    if target.is_symlink() or _file_hash(target) != expected_hash:
        raise PackageError(f"Materialized file failed verification: {target}")


def _vendor_visualizations(database, parent_path, capture_key):
    rows = [dict(row) for row in database.execute(
        """SELECT relative_path, size_bytes, source_sha256
           FROM source_file
           WHERE parent_path=? AND role='auxiliary' AND present=1 AND selected=1
           ORDER BY relative_path""", (parent_path,))]
    capture_count = database.execute(
        """SELECT count(*) FROM capture_snapshot
           WHERE parent_path=?""", (parent_path,)).fetchone()[0]
    associated = []
    for row in rows:
        name = Path(row["relative_path"]).name
        if name == f"{capture_key}_stereo_depth_imu.mp4" or (
                capture_count == 1 and name == "visualization.mp4"):
            associated.append(row)
    return associated


def project_supplier(episodes: tuple[dict, ...], output: Path) -> ProjectionArtifact:
    if not episodes:
        raise PackageError("Supplier projection requires at least one included capture")
    if output.exists() or output.is_symlink():
        raise PackageError(f"Supplier projection output already exists: {output}")
    capture_ids = [episode["internal_qc"]["capture_id"] for episode in episodes]
    if len(capture_ids) != len(set(capture_ids)):
        raise PackageError("Supplier projection contains a duplicate capture ID")

    rows = []
    documents = {}
    episode_durations = []
    for episode in sorted(episodes, key=lambda item: item["internal_qc"]["capture_id"]):
        internal = episode["internal_qc"]
        facts = internal["facts"]
        try:
            supplier_qc = _supplier_qc(internal, episode["decision"])
        except (KeyError, StopIteration, QcError) as error:
            raise PackageError(str(error)) from error
        capture_id = internal["capture_id"]
        raw_members = []
        for member in facts["source"]["members"]:
            raw_members.append({
                "role": member["role"], "camera_stream_id": member["camera_stream_id"],
                "path": f"raw/{Path(member['relative_path']).name}",
                "byte_count": member["size_bytes"], "sha256": member["source_sha256"],
            })
        for member in episode["vendor_visualizations"]:
            raw_members.append({
                "role": "vendor_visualization", "camera_stream_id": None,
                "path": f"raw/{Path(member['relative_path']).name}",
                "byte_count": member["size_bytes"], "sha256": member["source_sha256"],
            })
        raw_members.sort(key=lambda member: member["path"])
        if len({member["path"] for member in raw_members}) != len(raw_members):
            raise PackageError(f"Capture has duplicate raw destination names: {capture_id}")

        source_integrity = next(check for check in supplier_qc["checks"]
                                if check["check"] == "source_integrity")
        source_integrity["evidence"] = {
            "raw_file_count": len(raw_members),
            "raw_bytes": sum(member["byte_count"] for member in raw_members),
        }

        camera_streams = []
        for stream in facts["streams"]:
            stream_id = stream["camera_stream_id"]
            video_members = [member for member in raw_members
                             if member["role"] == "video"
                             and member["camera_stream_id"] == stream_id]
            vts_members = [member for member in raw_members
                           if member["role"] == "vts"
                           and member["camera_stream_id"] == stream_id]
            if len(video_members) != 1 or len(vts_members) != 1:
                raise PackageError(f"Stream is not supplier-complete: {capture_id}/{stream_id}")
            timing = next(item for item in facts["timing"]["streams"]
                          if item["camera_stream_id"] == stream_id)
            video, vts = stream["video"], stream["vts"]
            camera_streams.append({
                "camera_stream_id": stream_id,
                "raw_video_path": video_members[0]["path"],
                "preferred_video_path": video_members[0]["path"],
                "video_sha256": video_members[0]["sha256"],
                "raw_vts_path": vts_members[0]["path"], "vts_sha256": vts_members[0]["sha256"],
                "codec": video["codec"], "width": video["width"], "height": video["height"],
                "fps": video["average_frame_rate"], "duration_s": round(video["duration_ns"] / 1e9, 9),
                "decoded_frames": video["frame_count"], "vts_frames": vts["frame_count"],
                "native_sof_start_ns": vts["first_timestamp_ns"],
                "native_sof_end_ns": vts["last_timestamp_ns"],
                "timing_matched_frames": timing["matched_rows"],
                "timing_covered_frames": timing["coverage_rows"],
                "timing_coverage_pct": _percentage(timing["coverage_rows"], timing["matched_rows"]),
                "audio": {"stream_count": video["audio_stream_count"],
                          "streams": video["probe"]["audio_streams"]},
            })
        native = facts["imu"]["native"]
        supplier_native = {key: native[key] for key in META_IMU_FIELDS}
        imu_member = next(member for member in raw_members if member["role"] == "imu")
        telemetry = facts["telemetry"]
        meta = {
            "schema_version": "trinet_delivery.meta.v1", "capture_id": capture_id,
            "capture_layout": facts["capture_layout"],
            "source": {"relative_directory": episode["source_relative_directory"],
                       "group": episode["source_group"]},
            "raw_members": raw_members, "camera_streams": camera_streams,
            "shared_imu": {"raw_path": imu_member["path"], "sha256": imu_member["sha256"],
                           "sample_count": facts["imu"]["sample_count"], **supplier_native},
            "telemetry": {"present": telemetry["status"] == "decoded",
                          "record_count": telemetry.get("record_count")},
            "vendor_calibration_references": episode["vendor_calibration_references"],
        }
        durations = [stream["duration_s"] for stream in camera_streams]
        coverage = [stream["timing_coverage_pct"] for stream in camera_streams]
        episode_durations.append(max(durations))
        rows.append({
            "capture_id": capture_id,
            "source_relative_directory": episode["source_relative_directory"],
            "source_group": episode["source_group"], "capture_layout": facts["capture_layout"],
            "camera_stream_count": len(camera_streams), "raw_file_count": len(raw_members),
            "raw_bytes": sum(member["byte_count"] for member in raw_members),
            "device_id": "" if set(native["device_id_hex"]) == {"0"} else native["device_id_hex"],
            "imu_sha256": imu_member["sha256"],
            "tel_sha256": telemetry.get("source_sha256") or "",
            "imu_samples": facts["imu"]["sample_count"],
            "imu_rate_hz": native["measured_sample_rate_hz"],
            "camera_duration_min_s": min(durations), "camera_duration_max_s": max(durations),
            "timing_coverage_pct_min": min(coverage), "qc_result": supplier_qc["result"],
            "limitation_count": len(supplier_qc["limitations"]),
            "qc_path": f"episodes/{capture_id}/derived/qc.json",
        })
        documents[capture_id] = (meta, supplier_qc)

    layouts = sorted({row["capture_layout"] for row in rows})
    limitations = sum(row["limitation_count"] for row in rows)
    readme = (
        "# Trinet verified dataset\n\n"
        f"Delivered episodes: {len(rows)}\n\n"
        f"Capture layouts: {', '.join(layouts)}\n\n"
        f"Raw bytes: {sum(row['raw_bytes'] for row in rows)}\n\n"
        f"Total camera-duration seconds: {round(sum(episode_durations), 9)} "
        "(maximum stream duration per episode; stereo streams are not added together)\n\n"
        f"Declared limitations: {limitations}\n\n"
        "Raw files are byte-identical evidence. IMU remains at native rate. "
        "Timing uses hardware SOF with Before, After, and Closest IMU references; "
        "it is not physical synchronization certification.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary)
        manifest = staging / "manifest.csv"
        with manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (staging / "README.md").write_text(readme)
        for capture_id, (meta, supplier_qc) in documents.items():
            episode = staging / "episodes" / capture_id
            (episode / "derived").mkdir(parents=True)
            (episode / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            (episode / "derived/qc.json").write_text(
                json.dumps(supplier_qc, indent=2, sort_keys=True) + "\n")
        with manifest.open(newline="") as file:
            expected_rows = [{key: "" if value is None else str(value)
                              for key, value in row.items()} for row in rows]
            if list(csv.DictReader(file)) != expected_rows:
                raise PackageError("Supplier manifest does not match the projected rows")
        for capture_id, expected in documents.items():
            episode = staging / "episodes" / capture_id
            if json.loads((episode / "meta.json").read_text()) != expected[0]:
                raise PackageError(f"Supplier meta JSON changed during write: {capture_id}")
            if json.loads((episode / "derived/qc.json").read_text()) != expected[1]:
                raise PackageError(f"Supplier QC JSON changed during write: {capture_id}")
        staging.replace(output)
    return ProjectionArtifact(len(rows), sha256((output / "manifest.csv").read_bytes()).hexdigest())


def build_delivery(run_dir: Path, projection: Path, output: Path) -> DeliveryArtifact:
    if output.exists() or output.is_symlink():
        raise PackageError(f"Delivery output already exists: {output}")
    if not projection.is_dir() or projection.is_symlink():
        raise PackageError(f"Supplier projection is not a directory: {projection}")
    try:
        with (projection / "manifest.csv").open(newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise PackageError("Supplier manifest schema changed before packaging")
            rows = list(reader)
        if not rows:
            raise PackageError("Supplier projection has no episodes")
        capture_ids = [row["capture_id"] for row in rows]
        if len(capture_ids) != len(set(capture_ids)):
            raise PackageError("Supplier projection contains duplicate capture IDs")
        projected_files = {"README.md", "manifest.csv"}
        for row in rows:
            capture_id = row["capture_id"]
            projected_files.update((f"episodes/{capture_id}/meta.json", row["qc_path"]))
        actual_projection = {
            path.relative_to(projection).as_posix()
            for path in projection.rglob("*") if path.is_file()
        }
        if actual_projection != projected_files or any(path.is_symlink() for path in projection.rglob("*")):
            raise PackageError("Supplier projection file inventory changed before packaging")

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
            staging = Path(temporary)
            (staging / "README.md").write_bytes((projection / "README.md").read_bytes())
            (staging / "manifest.csv").write_bytes((projection / "manifest.csv").read_bytes())
            expected_files = {"README.md", "manifest.csv"}
            with sqlite3.connect(run_dir / "run.sqlite") as database:
                database.row_factory = sqlite3.Row
                if database.execute("PRAGMA user_version").fetchone()[0] not in (11, 12):
                    raise PackageError("Delivery requires a schema-11 or schema-12 run ledger")
                for row in rows:
                    capture_id = row["capture_id"]
                    source_episode = projection / "episodes" / capture_id
                    meta = json.loads((source_episode / "meta.json").read_text())
                    qc = json.loads((source_episode / "derived/qc.json").read_text())
                    if meta["schema_version"] != "trinet_delivery.meta.v1":
                        raise PackageError(f"Unsupported supplier meta schema: {capture_id}")
                    if meta["capture_id"] != capture_id or qc["capture_id"] != capture_id:
                        raise PackageError(f"Supplier capture cross-reference changed: {capture_id}")
                    if row["qc_path"] != f"episodes/{capture_id}/derived/qc.json":
                        raise PackageError(f"Supplier QC path changed: {capture_id}")
                    if qc["human_decision"]["status"] != "include" or qc["transformations"]:
                        raise PackageError(f"Delivery requires included untransformed input: {capture_id}")
                    if row["qc_result"] != qc["result"] or int(row["limitation_count"]) != len(qc["limitations"]):
                        raise PackageError(f"Supplier QC summary changed: {capture_id}")
                    if (row["source_relative_directory"] != meta["source"]["relative_directory"]
                            or row["source_group"] != meta["source"]["group"]
                            or row["capture_layout"] != meta["capture_layout"]):
                        raise PackageError(f"Supplier source metadata changed: {capture_id}")

                    target_episode = staging / "episodes" / capture_id
                    (target_episode / "derived").mkdir(parents=True)
                    (target_episode / "meta.json").write_bytes(
                        (source_episode / "meta.json").read_bytes())
                    (target_episode / "derived/qc.json").write_bytes(
                        (source_episode / "derived/qc.json").read_bytes())
                    expected_files.update((f"episodes/{capture_id}/meta.json", row["qc_path"]))

                    members = meta["raw_members"]
                    source_members = [dict(member) for member in database.execute(
                        """SELECT source_file.role AS role, camera_stream_id, relative_path,
                                  size_bytes, source_sha256
                           FROM capture_snapshot JOIN capture_member USING (parent_path, capture_key)
                           JOIN source_file USING (source_item_id)
                           WHERE capture_id=? AND is_canonical=1 ORDER BY relative_path""",
                        (capture_id,))]
                    expected_members = [{
                        "role": member["role"], "camera_stream_id": member["camera_stream_id"],
                        "path": f"raw/{Path(member['relative_path']).name}",
                        "byte_count": member["size_bytes"], "sha256": member["source_sha256"],
                    } for member in source_members]
                    source_capture = database.execute(
                        """SELECT parent_path, capture_key FROM capture_snapshot
                           WHERE capture_id=? AND is_canonical=1""", (capture_id,)).fetchone()
                    if not source_capture:
                        raise PackageError(f"Canonical capture is missing from the run ledger: {capture_id}")
                    expected_members.extend({
                        "role": "vendor_visualization", "camera_stream_id": None,
                        "path": f"raw/{Path(member['relative_path']).name}",
                        "byte_count": member["size_bytes"], "sha256": member["source_sha256"],
                    } for member in _vendor_visualizations(
                        database, source_capture["parent_path"], source_capture["capture_key"]))
                    expected_members.sort(key=lambda member: member["path"])
                    if members != expected_members:
                        raise PackageError(f"Supplier raw members changed from the run ledger: {capture_id}")
                    if (int(row["raw_file_count"]) != len(members)
                            or int(row["raw_bytes"]) != sum(member["byte_count"] for member in members)):
                        raise PackageError(f"Supplier raw summary changed: {capture_id}")
                    member_paths = [member["path"] for member in members]
                    if len(member_paths) != len(set(member_paths)):
                        raise PackageError(f"Supplier raw paths are duplicated: {capture_id}")
                    visualization_index = 0
                    for member in members:
                        relative = Path(member["path"])
                        if relative.parts != ("raw", relative.name):
                            raise PackageError(f"Unsafe supplier raw path: {member['path']}")
                        target = target_episode / relative
                        _materialize(run_dir / "cache/blobs" / member["sha256"], target,
                                     member["sha256"], member["byte_count"])
                        expected_files.add(f"episodes/{capture_id}/{relative.as_posix()}")
                        if member["role"] == "vendor_visualization":
                            index = staging / f".validate-{capture_id}-vendor-{visualization_index}.parquet"
                            verify_video(target, index, member["sha256"])
                            index.unlink()
                            visualization_index += 1
                    source_integrity = next(check for check in qc["checks"]
                                            if check["check"] == "source_integrity")
                    if source_integrity["evidence"] != {
                            "raw_file_count": len(members),
                            "raw_bytes": sum(member["byte_count"] for member in members)}:
                        raise PackageError(f"Supplier source integrity summary changed: {capture_id}")

                    imu = database.execute(
                        "SELECT * FROM imu_artifact WHERE capture_id=?", (capture_id,)).fetchone()
                    timing = database.execute(
                        "SELECT * FROM timing_artifact WHERE capture_id=?", (capture_id,)).fetchone()
                    telemetry = database.execute(
                        "SELECT * FROM tel_artifact WHERE capture_id=?", (capture_id,)).fetchone()
                    if not imu or imu["status"] != "decoded" or not timing or timing["status"] != "ready":
                        raise PackageError(f"Required derived artifacts are not ready: {capture_id}")
                    derived = ((imu, "imu.parquet"), (timing, "frame_timing.parquet"))
                    for artifact, name in derived:
                        relative = Path(artifact["parquet_relative_path"])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise PackageError(f"Unsafe derived artifact path: {capture_id}/{name}")
                        _materialize(run_dir / relative, target_episode / "derived" / name,
                                     artifact["parquet_sha256"])
                        expected_files.add(f"episodes/{capture_id}/derived/{name}")
                    if meta["telemetry"]["present"]:
                        if not telemetry or telemetry["status"] != "decoded":
                            raise PackageError(f"Supplier telemetry is not ready: {capture_id}")
                        relative = Path(telemetry["parquet_relative_path"])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise PackageError(f"Unsafe telemetry artifact path: {capture_id}")
                        _materialize(run_dir / relative, target_episode / "derived/telemetry.parquet",
                                     telemetry["parquet_sha256"])
                        expected_files.add(f"episodes/{capture_id}/derived/telemetry.parquet")

                    imu_path = target_episode / "derived/imu.parquet"
                    imu_table = pq.read_table(imu_path, columns=["timestamp_ns"])
                    timestamps = imu_table.column("timestamp_ns")
                    shared_imu = meta["shared_imu"]
                    imu_metadata = pq.read_schema(imu_path).metadata
                    if (not len(timestamps) or len(timestamps) != imu["sample_count"]
                            or len(timestamps) != shared_imu["sample_count"]
                            or timestamps[0].as_py() != shared_imu["first_sample_timestamp_ns"]
                            or timestamps[-1].as_py() != shared_imu["last_sample_timestamp_ns"]
                            or imu_metadata[b"source_sha256"].decode() != shared_imu["sha256"]
                            or row["imu_sha256"] != shared_imu["sha256"]):
                        raise PackageError(f"Supplier IMU facts do not match Parquet: {capture_id}")

                    timing_path = target_episode / "derived/frame_timing.parquet"
                    stream_facts = {item["camera_stream_id"]: item for item in timing_stream_facts(
                        timing_path, timing["parquet_sha256"])}
                    if sum(item["row_count"] for item in stream_facts.values()) != timing["row_count"]:
                        raise PackageError(f"Supplier timing row count changed: {capture_id}")
                    if int(row["camera_stream_count"]) != len(meta["camera_streams"]):
                        raise PackageError(f"Supplier camera stream count changed: {capture_id}")
                    for stream in meta["camera_streams"]:
                        stream_id = stream["camera_stream_id"]
                        facts = stream_facts.get(stream_id)
                        if (facts is None or facts["matched_rows"] != stream["timing_matched_frames"]
                                or facts["coverage_rows"] != stream["timing_covered_frames"]):
                            raise PackageError(f"Supplier timing facts changed: {capture_id}/{stream_id}")
                        if (stream["raw_video_path"] not in member_paths
                                or stream["preferred_video_path"] != stream["raw_video_path"]):
                            raise PackageError(f"Supplier preferred video path changed: {capture_id}/{stream_id}")
                        video_path = target_episode / stream["preferred_video_path"]
                        validation_index = staging / f".validate-{capture_id}-{stream_id}.parquet"
                        video = verify_video(video_path, validation_index, stream["video_sha256"])
                        validation_index.unlink()
                        if (video.frame_count != stream["decoded_frames"] or video.codec != stream["codec"]
                                or video.width != stream["width"] or video.height != stream["height"]
                                or video.average_frame_rate != stream["fps"]
                                or round(video.duration_ns / 1e9, 9) != stream["duration_s"]
                                or video.audio_stream_count != stream["audio"]["stream_count"]):
                            raise PackageError(f"Supplier video facts changed: {capture_id}/{stream_id}")
                    if meta["telemetry"]["present"]:
                        tel_rows = pq.read_metadata(
                            target_episode / "derived/telemetry.parquet").num_rows
                        if (tel_rows != telemetry["record_count"]
                                or tel_rows != meta["telemetry"]["record_count"]
                                or row["tel_sha256"] != telemetry["source_sha256"]):
                            raise PackageError(f"Supplier telemetry facts changed: {capture_id}")
                    elif row["tel_sha256"]:
                        raise PackageError(f"Supplier manifest has unexpected telemetry: {capture_id}")

            actual_files = {
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*") if path.is_file()
            }
            if actual_files != expected_files or any(path.is_symlink() for path in staging.rglob("*")):
                raise PackageError("Final delivery file inventory does not match its projection")
            with (staging / "manifest.csv").open(newline="") as file:
                if list(csv.DictReader(file)) != rows:
                    raise PackageError("Final delivery manifest changed during packaging")
            staging.replace(output)
    except PackageError:
        raise
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error,
            json.JSONDecodeError, pa.ArrowException, VideoError) as error:
        raise PackageError(f"Delivery validation failed: {error}") from error
    files = [path for path in output.rglob("*") if path.is_file()]
    return DeliveryArtifact(len(rows), len(files), sum(path.stat().st_size for path in files),
                            _file_hash(output / "manifest.csv"))
