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

from actuate_delivery.qc import QcError, _supplier_qc, timing_stream_facts
from actuate_delivery.video import VideoError, verify_video


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


EPISODE_FIELDS = (
    "episode_id",
    "capture_layout",
    "camera_stream_count",
    "raw_file_count",
    "raw_bytes",
    "preview_count",
    "preview_bytes",
    "native_device_id",
    "imu_sha256",
    "tel_sha256",
    "imu_samples",
    "imu_rate_hz",
    "camera_duration_min_s",
    "camera_duration_max_s",
    "timing_coverage_pct_min",
    "camera_frames_without_imu_mapping",
    "stereo_paired_frames",
    "stereo_unmatched_frames",
)

META_IMU_FIELDS = (
    "version",
    "declared_sample_rate_hz",
    "measured_sample_rate_hz",
    "accel_full_scale_code",
    "gyro_full_scale_code",
    "header_start_time_ns",
    "video_start_time_ns",
    "flags",
    "device_id_hex",
    "ios_clock_offset_ns",
    "first_sample_timestamp_ns",
    "last_sample_timestamp_ns",
)

CUSTOMER_QC_FIELDS = {
    "schema_version",
    "episode_id",
    "integrity",
    "imu",
    "camera_streams",
    "timing_basis",
    "stereo",
    "telemetry",
}


def _percentage(part, whole):
    return None if not whole else round(part * 100 / whole, 6)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_customer_qc(document):
    required = CUSTOMER_QC_FIELDS - {"stereo", "telemetry"}
    if set(document) - CUSTOMER_QC_FIELDS or not required <= set(document):
        raise PackageError("Customer QC contains unsupported fields")
    if document["schema_version"] != "actuate_delivery.qc_facts.v1":
        raise PackageError("Customer QC schema is unsupported")
    forbidden = {
        "capture_id",
        "result",
        "status",
        "limitations",
        "human_decision",
        "reviewer",
        "transformations",
        "tool_versions",
        "source_path",
    }

    def inspect(value):
        if isinstance(value, dict):
            if forbidden & set(value):
                raise PackageError("Customer QC contains internal verdict or provenance fields")
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(document)


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
    rows = [
        dict(row)
        for row in database.execute(
            """SELECT relative_path, size_bytes, source_sha256
           FROM source_file
           WHERE parent_path=? AND role='auxiliary' AND present=1 AND selected=1
           ORDER BY relative_path""",
            (parent_path,),
        )
    ]
    capture_count = database.execute(
        """SELECT count(*) FROM capture_snapshot
           WHERE parent_path=?""",
        (parent_path,),
    ).fetchone()[0]
    associated = []
    for row in rows:
        name = Path(row["relative_path"]).name
        if name == f"{capture_key}_stereo_depth_imu.mp4" or (
            capture_count == 1 and name == "visualization.mp4"
        ):
            associated.append(row)
    return associated


def _calibration(calibration, episode_ids):
    if calibration is None:
        return None
    required = {
        "schema_version",
        "calibration_id",
        "rig_id",
        "applies_to_episode_ids",
        "applied_by_pipeline",
        "cameras",
        "transforms",
        "imu",
    }
    if not required <= set(calibration):
        raise PackageError("Dataset calibration is missing required fields")
    if calibration["schema_version"] != "actuate_delivery.calibration.v1":
        raise PackageError("Dataset calibration schema is unsupported")
    if calibration["applied_by_pipeline"] is not False:
        raise PackageError("Dataset calibration must not claim pipeline application")
    if Path(calibration["calibration_id"]).name != calibration["calibration_id"]:
        raise PackageError("Dataset calibration ID is unsafe")
    if sorted(calibration["applies_to_episode_ids"]) != sorted(episode_ids):
        raise PackageError("Dataset calibration episode binding does not match the delivery")
    return calibration


def project_supplier(
    episodes: tuple[dict, ...],
    output: Path,
    calibration: dict | None = None,
    telemetry_mode: str = "automatic",
) -> ProjectionArtifact:
    if not episodes:
        raise PackageError("Supplier projection requires at least one included capture")
    if output.exists() or output.is_symlink():
        raise PackageError(f"Supplier projection output already exists: {output}")
    capture_ids = [episode["internal_qc"]["capture_id"] for episode in episodes]
    if len(capture_ids) != len(set(capture_ids)):
        raise PackageError("Supplier projection contains a duplicate capture ID")
    episode_ids = [episode["episode_id"] for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise PackageError("Supplier projection contains a duplicate episode ID")
    if any(
        len(value) != 14
        or not value.startswith("episode_")
        or not value[8:].isdigit()
        or value == "episode_000000"
        for value in episode_ids
    ):
        raise PackageError("Supplier projection contains an invalid episode ID")
    calibration = _calibration(calibration, episode_ids)
    calibration_id = calibration["calibration_id"] if calibration else None
    rig_id = calibration["rig_id"] if calibration else None
    telemetry_count = sum(
        episode["internal_qc"]["facts"]["telemetry"]["status"] == "decoded" for episode in episodes
    )
    if telemetry_mode == "automatic":
        if telemetry_count not in (0, len(episodes)):
            raise PackageError("Partial telemetry requires an explicit dataset policy")
        telemetry_mode = "include_available" if telemetry_count else "exclude_all"
    if telemetry_mode not in ("include_available", "exclude_all"):
        raise PackageError(f"Unsupported telemetry mode: {telemetry_mode}")

    rows = []
    documents = {}
    episode_durations = []
    for episode in sorted(episodes, key=lambda item: item["episode_id"]):
        internal = episode["internal_qc"]
        facts = internal["facts"]
        include_telemetry = (
            telemetry_mode == "include_available" and facts["telemetry"]["status"] == "decoded"
        )
        try:
            supplier_qc = _supplier_qc(internal, episode["decision"])
        except (KeyError, StopIteration, QcError) as error:
            raise PackageError(str(error)) from error
        episode_id = episode["episode_id"]
        capture_id = internal["capture_id"]
        raw_members = []
        for member in facts["source"]["members"]:
            if member["role"] == "telemetry" and not include_telemetry:
                continue
            projected = {
                "role": member["role"],
                "path": f"raw/{Path(member['relative_path']).name}",
                "byte_count": member["size_bytes"],
                "sha256": member["source_sha256"],
            }
            if member["camera_stream_id"] is not None:
                projected["camera_stream_id"] = member["camera_stream_id"]
            raw_members.append(projected)
        raw_members.sort(key=lambda member: member["path"])
        if len({member["path"] for member in raw_members}) != len(raw_members):
            raise PackageError(f"Capture has duplicate raw destination names: {capture_id}")

        previews = []
        visualizations = episode["vendor_visualizations"]
        for index, member in enumerate(visualizations, 1):
            name = (
                "capture_preview.mp4"
                if len(visualizations) == 1
                else f"capture_preview_{index}.mp4"
            )
            previews.append(
                {
                    "path": f"previews/{name}",
                    "byte_count": member["size_bytes"],
                    "sha256": member["source_sha256"],
                }
            )

        supplier_qc["episode_id"] = episode_id
        supplier_qc["integrity"] = {
            "raw_file_count": len(raw_members),
            "raw_bytes": sum(member["byte_count"] for member in raw_members),
            "raw_files": raw_members,
        }
        if not include_telemetry:
            supplier_qc.pop("telemetry", None)

        camera_streams = []
        for stream in facts["streams"]:
            stream_id = stream["camera_stream_id"]
            video_members = [
                member
                for member in raw_members
                if member["role"] == "video" and member.get("camera_stream_id") == stream_id
            ]
            vts_members = [
                member
                for member in raw_members
                if member["role"] == "vts" and member.get("camera_stream_id") == stream_id
            ]
            if len(video_members) != 1 or len(vts_members) != 1:
                raise PackageError(f"Stream is not supplier-complete: {capture_id}/{stream_id}")
            timing = next(
                item for item in facts["timing"]["streams"] if item["camera_stream_id"] == stream_id
            )
            video, vts = stream["video"], stream["vts"]
            camera_streams.append(
                {
                    "camera_stream_id": stream_id,
                    "raw_video_path": video_members[0]["path"],
                    "preferred_video_path": video_members[0]["path"],
                    "video_sha256": video_members[0]["sha256"],
                    "raw_vts_path": vts_members[0]["path"],
                    "vts_sha256": vts_members[0]["sha256"],
                    "codec": video["codec"],
                    "width": video["width"],
                    "height": video["height"],
                    "fps": video["average_frame_rate"],
                    "duration_s": round(video["duration_ns"] / 1e9, 9),
                    "decoded_frames": video["frame_count"],
                    "vts_frames": vts["frame_count"],
                    "native_sof_start_ns": vts["first_timestamp_ns"],
                    "native_sof_end_ns": vts["last_timestamp_ns"],
                    "timing_matched_frames": timing["matched_rows"],
                    "timing_covered_frames": timing["coverage_rows"],
                    "timing_coverage_pct": _percentage(
                        timing["coverage_rows"], timing["matched_rows"]
                    ),
                    "audio": {
                        "stream_count": video["audio_stream_count"],
                        "streams": video["probe"]["audio_streams"],
                    },
                }
            )
        native = facts["imu"]["native"]
        supplier_native = {key: native[key] for key in META_IMU_FIELDS}
        imu_member = next(member for member in raw_members if member["role"] == "imu")
        telemetry = facts["telemetry"]
        meta = {
            "schema_version": "actuate_delivery.meta.v3",
            "episode_id": episode_id,
            "capture_layout": facts["capture_layout"],
            "raw_members": raw_members,
            "camera_streams": camera_streams,
            "shared_imu": {
                "raw_path": imu_member["path"],
                "sha256": imu_member["sha256"],
                "sample_count": facts["imu"]["sample_count"],
                **supplier_native,
            },
        }
        if calibration:
            meta.update(rig_id=rig_id, calibration_id=calibration_id)
        if previews:
            meta["previews"] = previews
        if include_telemetry:
            meta["telemetry"] = {"record_count": telemetry["record_count"]}
        if facts["capture_layout"] == "stereo_pair":
            meta["stereo"] = {
                "association_basis": "unique_equal_venc_seq",
                "paired_frames": facts["timing"]["stereo_pair_count"],
                "unmatched_frames": facts["timing"]["stereo_unmatched_rows"],
            }
        durations = [stream["duration_s"] for stream in camera_streams]
        coverage = [stream["timing_coverage_pct"] for stream in camera_streams]
        episode_durations.append(max(durations))
        rows.append(
            {
                "episode_id": episode_id,
                "capture_layout": facts["capture_layout"],
                "camera_stream_count": len(camera_streams),
                "raw_file_count": len(raw_members),
                "raw_bytes": sum(member["byte_count"] for member in raw_members),
                "preview_count": len(previews),
                "preview_bytes": sum(preview["byte_count"] for preview in previews),
                "native_device_id": ""
                if set(native["device_id_hex"]) == {"0"}
                else native["device_id_hex"],
                "imu_sha256": imu_member["sha256"],
                "tel_sha256": telemetry.get("source_sha256") if include_telemetry else "",
                "imu_samples": facts["imu"]["sample_count"],
                "imu_rate_hz": native["measured_sample_rate_hz"],
                "camera_duration_min_s": min(durations),
                "camera_duration_max_s": max(durations),
                "timing_coverage_pct_min": min(coverage),
                "camera_frames_without_imu_mapping": sum(
                    stream["timing_matched_frames"] - stream["timing_covered_frames"]
                    for stream in camera_streams
                ),
                "stereo_paired_frames": facts["timing"]["stereo_pair_count"],
                "stereo_unmatched_frames": facts["timing"]["stereo_unmatched_rows"],
            }
        )
        documents[episode_id] = (meta, supplier_qc)

    layouts = sorted({row["capture_layout"] for row in rows})
    layout_text = {"single_video": "monocular", "stereo_pair": "stereo"}
    preview_section = (
        "\n## Previews\n\nPreview videos are supplied informational views. Numeric depth or derived "
        "orientation shown in a preview is not a dataset signal.\n"
        if any(row["preview_count"] for row in rows)
        else ""
    )
    calibration_section = (
        f"\n## Calibration\n\nAll episodes reference `{calibration_id}` for `{rig_id}`. The supplied "
        "parameters are provided for downstream use and were not applied to raw files or "
        "timing tables.\n"
        if calibration
        else ""
    )
    delivered_telemetry = sum(bool(row["tel_sha256"]) for row in rows)
    telemetry_section = (
        "Native telemetry is included for every episode.\n\n"
        if delivered_telemetry == len(rows)
        else f"Native telemetry is included for {delivered_telemetry} of {len(rows)} episodes. "
        "Episodes without decoded telemetry contain no telemetry files.\n\n"
        if delivered_telemetry
        else ""
    )
    video_validation = (
        "Every camera video and every preview video was fully decoded"
        if any(row["preview_count"] for row in rows)
        else "Every camera video was fully decoded"
    )
    duration_note = (
        " (maximum stream duration per episode; stereo streams are not added together)"
        if "stereo_pair" in layouts
        else ""
    )
    measured_facts = (
        "IMU coverage and stereo association measurements"
        if "stereo_pair" in layouts
        else "IMU coverage measurements"
    )
    readme = (
        "# Actuate dataset\n\n"
        f"Episodes: {len(rows)}\n\n"
        f"Capture layouts: {', '.join(layout_text[item] for item in layouts)}\n\n"
        f"Raw bytes: {sum(row['raw_bytes'] for row in rows)}\n\n"
        f"Total camera-duration seconds: {round(sum(episode_durations), 9)}{duration_note}\n\n"
        "## Contents\n\n"
        "`episodes.csv` is the dataset index. Each episode contains byte-identical source "
        "files under `raw/` and readable sensor, timing, and factual validation files under "
        "`derived/`.\n\n" + telemetry_section + "## Validation\n\n"
        f"Every source file was SHA-256 verified. {video_validation}, and every CSV, JSON, "
        "and Parquet output was reopened and cross-checked. "
        "`episodes.csv`, `meta.json`, `derived/qc.json`, and `frame_timing.parquet` retain factual "
        f"{measured_facts}. IMU remains at native rate. Timing uses "
        "hardware SOF with Before, After, and Closest IMU references; it is not physical "
        "synchronization certification.\n" + calibration_section + preview_section
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary)
        index = staging / "episodes.csv"
        with index.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (staging / "README.md").write_text(readme)
        if calibration:
            path = staging / "calibration" / f"{calibration_id}.json"
            path.parent.mkdir()
            path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
        for episode_id, (meta, supplier_qc) in documents.items():
            episode = staging / "episodes" / episode_id
            (episode / "derived").mkdir(parents=True)
            (episode / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            (episode / "derived/qc.json").write_text(
                json.dumps(supplier_qc, indent=2, sort_keys=True) + "\n"
            )
        with index.open(newline="") as file:
            expected_rows = [
                {key: "" if value is None else str(value) for key, value in row.items()}
                for row in rows
            ]
            if list(csv.DictReader(file)) != expected_rows:
                raise PackageError("Supplier episode index does not match the projected rows")
        for episode_id, expected in documents.items():
            episode = staging / "episodes" / episode_id
            if json.loads((episode / "meta.json").read_text()) != expected[0]:
                raise PackageError(f"Supplier meta JSON changed during write: {episode_id}")
            if json.loads((episode / "derived/qc.json").read_text()) != expected[1]:
                raise PackageError(f"Supplier QC JSON changed during write: {episode_id}")
        staging.replace(output)
    return ProjectionArtifact(len(rows), sha256((output / "episodes.csv").read_bytes()).hexdigest())


def build_delivery(run_dir: Path, projection: Path, output: Path) -> DeliveryArtifact:
    if output.exists() or output.is_symlink():
        raise PackageError(f"Delivery output already exists: {output}")
    if not projection.is_dir() or projection.is_symlink():
        raise PackageError(f"Supplier projection is not a directory: {projection}")
    try:
        with (projection / "episodes.csv").open(newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
                raise PackageError("Supplier episode index schema changed before packaging")
            rows = list(reader)
        if not rows:
            raise PackageError("Supplier projection has no episodes")
        episode_ids = [row["episode_id"] for row in rows]
        if len(episode_ids) != len(set(episode_ids)):
            raise PackageError("Supplier projection contains duplicate episode IDs")
        calibration_files = (
            list((projection / "calibration").glob("*.json"))
            if (projection / "calibration").is_dir()
            else []
        )
        if len(calibration_files) > 1:
            raise PackageError("Supplier projection contains multiple calibration files")
        calibration = json.loads(calibration_files[0].read_text()) if calibration_files else None
        _calibration(calibration, episode_ids)
        projected_files = {"README.md", "episodes.csv"}
        if calibration_files:
            projected_files.add(f"calibration/{calibration_files[0].name}")
        for row in rows:
            episode_id = row["episode_id"]
            projected_files.update(
                (f"episodes/{episode_id}/meta.json", f"episodes/{episode_id}/derived/qc.json")
            )
        actual_projection = {
            path.relative_to(projection).as_posix()
            for path in projection.rglob("*")
            if path.is_file()
        }
        if actual_projection != projected_files or any(
            path.is_symlink() for path in projection.rglob("*")
        ):
            raise PackageError("Supplier projection file inventory changed before packaging")

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
            staging = Path(temporary)
            (staging / "README.md").write_bytes((projection / "README.md").read_bytes())
            (staging / "episodes.csv").write_bytes((projection / "episodes.csv").read_bytes())
            expected_files = {"README.md", "episodes.csv"}
            if calibration_files:
                relative = Path("calibration") / calibration_files[0].name
                target = staging / relative
                target.parent.mkdir()
                target.write_bytes(calibration_files[0].read_bytes())
                if json.loads(target.read_text()) != calibration:
                    raise PackageError("Dataset calibration changed during packaging")
                expected_files.add(relative.as_posix())
            with sqlite3.connect(run_dir / "run.sqlite") as database:
                database.row_factory = sqlite3.Row
                if database.execute("PRAGMA user_version").fetchone()[0] != 15:
                    raise PackageError("Delivery requires a schema-15 run ledger")
                for row in rows:
                    episode_id = row["episode_id"]
                    episode_number = int(episode_id.removeprefix("episode_"))
                    episode_record = database.execute(
                        "SELECT capture_id FROM delivery_episode WHERE episode_number=?",
                        (episode_number,),
                    ).fetchone()
                    if episode_record is None:
                        raise PackageError(
                            f"Supplier episode ID is not in the run ledger: {episode_id}"
                        )
                    capture_id = episode_record[0]
                    source_episode = projection / "episodes" / episode_id
                    meta = json.loads((source_episode / "meta.json").read_text())
                    qc = json.loads((source_episode / "derived/qc.json").read_text())
                    _validate_customer_qc(qc)
                    if meta["schema_version"] != "actuate_delivery.meta.v3":
                        raise PackageError(f"Unsupported supplier meta schema: {capture_id}")
                    if meta["episode_id"] != episode_id or qc["episode_id"] != episode_id:
                        raise PackageError(
                            f"Supplier episode cross-reference changed: {episode_id}"
                        )
                    binding = (
                        {
                            "rig_id": calibration["rig_id"],
                            "calibration_id": calibration["calibration_id"],
                        }
                        if calibration
                        else {}
                    )
                    if {key: meta[key] for key in binding} != binding or (
                        not calibration and ({"rig_id", "calibration_id"} & set(meta))
                    ):
                        raise PackageError(f"Supplier calibration binding changed: {capture_id}")
                    if episode_id != f"episode_{episode_number:06d}":
                        raise PackageError(
                            f"Supplier episode ID changed from the run ledger: {capture_id}"
                        )
                    decision = database.execute(
                        "SELECT status FROM delivery_decision WHERE capture_id=?", (capture_id,)
                    ).fetchone()
                    if (
                        not decision
                        or decision[0] != "include"
                        or row["capture_layout"] != meta["capture_layout"]
                    ):
                        raise PackageError(f"Delivery requires an included episode: {capture_id}")

                    target_episode = staging / "episodes" / episode_id
                    (target_episode / "derived").mkdir(parents=True)
                    (target_episode / "meta.json").write_bytes(
                        (source_episode / "meta.json").read_bytes()
                    )
                    expected_files.add(f"episodes/{episode_id}/meta.json")
                    (target_episode / "derived/qc.json").write_bytes(
                        (source_episode / "derived/qc.json").read_bytes()
                    )
                    expected_files.add(f"episodes/{episode_id}/derived/qc.json")

                    members = meta["raw_members"]
                    source_members = [
                        dict(member)
                        for member in database.execute(
                            """SELECT source_file.role AS role, camera_stream_id, relative_path,
                                  size_bytes, source_sha256
                           FROM capture_snapshot JOIN capture_member USING (parent_path, capture_key)
                           JOIN source_file USING (source_item_id)
                           WHERE capture_id=? AND is_canonical=1 ORDER BY relative_path""",
                            (capture_id,),
                        )
                    ]
                    expected_members = []
                    for member in source_members:
                        if member["role"] == "telemetry" and "telemetry" not in meta:
                            continue
                        expected = {
                            "role": member["role"],
                            "path": f"raw/{Path(member['relative_path']).name}",
                            "byte_count": member["size_bytes"],
                            "sha256": member["source_sha256"],
                        }
                        if member["camera_stream_id"] is not None:
                            expected["camera_stream_id"] = member["camera_stream_id"]
                        expected_members.append(expected)
                    source_capture = database.execute(
                        """SELECT parent_path, capture_key FROM capture_snapshot
                           WHERE capture_id=? AND is_canonical=1""",
                        (capture_id,),
                    ).fetchone()
                    if not source_capture:
                        raise PackageError(
                            f"Canonical capture is missing from the run ledger: {capture_id}"
                        )
                    expected_members.sort(key=lambda member: member["path"])
                    if members != expected_members:
                        raise PackageError(
                            f"Supplier raw members changed from the run ledger: {capture_id}"
                        )
                    if int(row["raw_file_count"]) != len(members) or int(row["raw_bytes"]) != sum(
                        member["byte_count"] for member in members
                    ):
                        raise PackageError(f"Supplier raw summary changed: {capture_id}")
                    member_paths = [member["path"] for member in members]
                    if len(member_paths) != len(set(member_paths)):
                        raise PackageError(f"Supplier raw paths are duplicated: {capture_id}")
                    for member in members:
                        relative = Path(member["path"])
                        if relative.parts != ("raw", relative.name):
                            raise PackageError(f"Unsafe supplier raw path: {member['path']}")
                        target = target_episode / relative
                        _materialize(
                            run_dir / "cache/blobs" / member["sha256"],
                            target,
                            member["sha256"],
                            member["byte_count"],
                        )
                        expected_files.add(f"episodes/{episode_id}/{relative.as_posix()}")
                    visualizations = _vendor_visualizations(
                        database, source_capture["parent_path"], source_capture["capture_key"]
                    )
                    expected_previews = []
                    for preview_index, member in enumerate(visualizations, 1):
                        name = (
                            "capture_preview.mp4"
                            if len(visualizations) == 1
                            else f"capture_preview_{preview_index}.mp4"
                        )
                        preview = {
                            "path": f"previews/{name}",
                            "byte_count": member["size_bytes"],
                            "sha256": member["source_sha256"],
                        }
                        expected_previews.append(preview)
                        target = target_episode / preview["path"]
                        _materialize(
                            run_dir / "cache/blobs" / preview["sha256"],
                            target,
                            preview["sha256"],
                            preview["byte_count"],
                        )
                        index = staging / f".validate-{capture_id}-preview-{preview_index}.parquet"
                        verify_video(target, index, preview["sha256"])
                        index.unlink()
                        expected_files.add(f"episodes/{episode_id}/{preview['path']}")
                    if meta.get("previews", []) != expected_previews:
                        raise PackageError(
                            f"Supplier previews changed from the run ledger: {capture_id}"
                        )
                    if int(row["preview_count"]) != len(expected_previews) or int(
                        row["preview_bytes"]
                    ) != sum(item["byte_count"] for item in expected_previews):
                        raise PackageError(f"Supplier preview summary changed: {capture_id}")
                    if qc["integrity"] != {
                        "raw_file_count": len(members),
                        "raw_bytes": sum(member["byte_count"] for member in members),
                        "raw_files": members,
                    }:
                        raise PackageError(
                            f"Supplier source integrity summary changed: {capture_id}"
                        )

                    imu = database.execute(
                        "SELECT * FROM imu_artifact WHERE capture_id=?", (capture_id,)
                    ).fetchone()
                    timing = database.execute(
                        "SELECT * FROM timing_artifact WHERE capture_id=?", (capture_id,)
                    ).fetchone()
                    telemetry = database.execute(
                        "SELECT * FROM tel_artifact WHERE capture_id=?", (capture_id,)
                    ).fetchone()
                    if (
                        not imu
                        or imu["status"] != "decoded"
                        or not timing
                        or timing["status"] != "ready"
                    ):
                        raise PackageError(
                            f"Required derived artifacts are not ready: {capture_id}"
                        )
                    derived = ((imu, "imu.parquet"), (timing, "frame_timing.parquet"))
                    for artifact, name in derived:
                        relative = Path(artifact["parquet_relative_path"])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise PackageError(f"Unsafe derived artifact path: {capture_id}/{name}")
                        _materialize(
                            run_dir / relative,
                            target_episode / "derived" / name,
                            artifact["parquet_sha256"],
                        )
                        expected_files.add(f"episodes/{episode_id}/derived/{name}")
                    if "telemetry" in meta:
                        if not telemetry or telemetry["status"] != "decoded":
                            raise PackageError(f"Supplier telemetry is not ready: {capture_id}")
                        relative = Path(telemetry["parquet_relative_path"])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise PackageError(f"Unsafe telemetry artifact path: {capture_id}")
                        _materialize(
                            run_dir / relative,
                            target_episode / "derived/telemetry.parquet",
                            telemetry["parquet_sha256"],
                        )
                        expected_files.add(f"episodes/{episode_id}/derived/telemetry.parquet")

                    imu_path = target_episode / "derived/imu.parquet"
                    imu_table = pq.read_table(imu_path, columns=["timestamp_ns"])
                    timestamps = imu_table.column("timestamp_ns")
                    shared_imu = meta["shared_imu"]
                    imu_metadata = pq.read_schema(imu_path).metadata
                    if (
                        not len(timestamps)
                        or len(timestamps) != imu["sample_count"]
                        or len(timestamps) != shared_imu["sample_count"]
                        or timestamps[0].as_py() != shared_imu["first_sample_timestamp_ns"]
                        or timestamps[-1].as_py() != shared_imu["last_sample_timestamp_ns"]
                        or imu_metadata[b"source_sha256"].decode() != shared_imu["sha256"]
                        or row["imu_sha256"] != shared_imu["sha256"]
                    ):
                        raise PackageError(f"Supplier IMU facts do not match Parquet: {capture_id}")
                    if qc["imu"] != {
                        "sample_count": shared_imu["sample_count"],
                        "measured_sample_rate_hz": shared_imu["measured_sample_rate_hz"],
                        "first_sample_timestamp_ns": shared_imu["first_sample_timestamp_ns"],
                        "last_sample_timestamp_ns": shared_imu["last_sample_timestamp_ns"],
                    }:
                        raise PackageError(f"Customer QC IMU facts changed: {capture_id}")

                    timing_path = target_episode / "derived/frame_timing.parquet"
                    stream_facts = {
                        item["camera_stream_id"]: item
                        for item in timing_stream_facts(timing_path, timing["parquet_sha256"])
                    }
                    if (
                        sum(item["row_count"] for item in stream_facts.values())
                        != timing["row_count"]
                    ):
                        raise PackageError(f"Supplier timing row count changed: {capture_id}")
                    if int(row["camera_stream_count"]) != len(meta["camera_streams"]):
                        raise PackageError(f"Supplier camera stream count changed: {capture_id}")
                    qc_streams = {item["camera_stream_id"]: item for item in qc["camera_streams"]}
                    for stream in meta["camera_streams"]:
                        stream_id = stream["camera_stream_id"]
                        facts = stream_facts.get(stream_id)
                        if (
                            facts is None
                            or facts["matched_rows"] != stream["timing_matched_frames"]
                            or facts["coverage_rows"] != stream["timing_covered_frames"]
                        ):
                            raise PackageError(
                                f"Supplier timing facts changed: {capture_id}/{stream_id}"
                            )
                        qc_stream = qc_streams.get(stream_id)
                        if (
                            qc_stream is None
                            or qc_stream["video"]
                            != {
                                "codec": stream["codec"],
                                "width": stream["width"],
                                "height": stream["height"],
                                "fps": stream["fps"],
                                "duration_s": stream["duration_s"],
                                "decoded_frames": stream["decoded_frames"],
                                "full_decode_completed": True,
                            }
                            or qc_stream["vts_frames"] != stream["vts_frames"]
                        ):
                            raise PackageError(
                                f"Customer QC video facts changed: {capture_id}/{stream_id}"
                            )
                        expected_timing = {
                            key: facts[key]
                            for key in (
                                "row_count",
                                "matched_rows",
                                "coverage_rows",
                                "outside_imu_coverage_rows",
                                "video_only_rows",
                                "vts_only_rows",
                                "outside_imu_coverage_ranges",
                                "video_vts_mismatch_ranges",
                            )
                        }
                        if qc_stream["timing"] != expected_timing:
                            raise PackageError(
                                f"Customer QC timing facts changed: {capture_id}/{stream_id}"
                            )
                        if (
                            stream["raw_video_path"] not in member_paths
                            or stream["preferred_video_path"] != stream["raw_video_path"]
                        ):
                            raise PackageError(
                                f"Supplier preferred video path changed: {capture_id}/{stream_id}"
                            )
                        video_path = target_episode / stream["preferred_video_path"]
                        validation_index = staging / f".validate-{capture_id}-{stream_id}.parquet"
                        video = verify_video(video_path, validation_index, stream["video_sha256"])
                        validation_index.unlink()
                        if (
                            video.frame_count != stream["decoded_frames"]
                            or video.codec != stream["codec"]
                            or video.width != stream["width"]
                            or video.height != stream["height"]
                            or video.average_frame_rate != stream["fps"]
                            or round(video.duration_ns / 1e9, 9) != stream["duration_s"]
                            or video.audio_stream_count != stream["audio"]["stream_count"]
                        ):
                            raise PackageError(
                                f"Supplier video facts changed: {capture_id}/{stream_id}"
                            )
                    if "telemetry" in meta:
                        tel_rows = pq.read_metadata(
                            target_episode / "derived/telemetry.parquet"
                        ).num_rows
                        if (
                            tel_rows != telemetry["record_count"]
                            or tel_rows != meta["telemetry"]["record_count"]
                            or row["tel_sha256"] != telemetry["source_sha256"]
                        ):
                            raise PackageError(f"Supplier telemetry facts changed: {capture_id}")
                        if qc.get("telemetry") != {"record_count": tel_rows}:
                            raise PackageError(f"Customer QC telemetry facts changed: {capture_id}")
                    elif row["tel_sha256"]:
                        raise PackageError(
                            f"Supplier manifest has unexpected telemetry: {capture_id}"
                        )
                    elif "telemetry" in qc:
                        raise PackageError(f"Customer QC has unexpected telemetry: {capture_id}")
                    if "stereo" in meta:
                        expected_stereo = {
                            "association_basis": meta["stereo"]["association_basis"],
                            "paired_frames": meta["stereo"]["paired_frames"],
                            "unmatched_frames": meta["stereo"]["unmatched_frames"],
                            "unmatched_ranges": {
                                stream_id: stream_facts[stream_id]["stereo_unmatched_ranges"]
                                for stream_id in sorted(stream_facts)
                            },
                        }
                        if qc.get("stereo") != expected_stereo:
                            raise PackageError(f"Customer QC stereo facts changed: {capture_id}")
                    elif "stereo" in qc:
                        raise PackageError(f"Customer QC has unexpected stereo facts: {capture_id}")

            actual_files = {
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*")
                if path.is_file()
            }
            if actual_files != expected_files or any(
                path.is_symlink() for path in staging.rglob("*")
            ):
                raise PackageError("Final delivery file inventory does not match its projection")
            with (staging / "episodes.csv").open(newline="") as file:
                if list(csv.DictReader(file)) != rows:
                    raise PackageError("Final delivery episode index changed during packaging")
            staging.replace(output)
    except PackageError:
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        sqlite3.Error,
        json.JSONDecodeError,
        pa.ArrowException,
        VideoError,
    ) as error:
        raise PackageError(f"Delivery validation failed: {error}") from error
    files = [path for path in output.rglob("*") if path.is_file()]
    return DeliveryArtifact(
        len(rows),
        len(files),
        sum(path.stat().st_size for path in files),
        _file_hash(output / "episodes.csv"),
    )
