import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class QcError(ValueError):
    pass


@dataclass(frozen=True)
class QcArtifact:
    json_sha256: str
    pass_count: int
    fail_count: int
    unknown_count: int
    not_applicable_count: int


SUPPLIER_BLOCKERS = (
    "capture_structure", "source_integrity", "imu_decode", "vts_decode:",
    "video_decode:", "video_vts_frame_count:", "timing_artifact",
    "timing_row_accounting",
)


def supplier_issues(internal: dict) -> tuple[list[str], list[str]]:
    if internal["schema_version"] != "actuate_delivery.qc_internal.v2":
        raise QcError(f"Unsupported internal QC schema: {internal['schema_version']}")
    checks = {check["check_id"]: check["result"] for check in internal["checks"]}
    unresolved = [name for name, result in checks.items()
                  if result not in ("pass", "not_applicable")]
    blocked = [name for name in unresolved
               if any(name == prefix or name.startswith(prefix) for prefix in SUPPLIER_BLOCKERS)]
    return blocked, [name for name in unresolved if name not in blocked]


def controlled_limitations(internal: dict) -> list[str]:
    _, material = supplier_issues(internal)
    if not material:
        return []
    supported = {"camera_imu_coverage", "stereo_sequence_pairing", "telemetry"}
    supported.update(
        f"camera_imu_coverage:{stream['camera_stream_id']}"
        for stream in internal["facts"]["timing"].get("streams", ()))
    unsupported = sorted(set(material) - supported)
    if unsupported:
        raise QcError(f"No controlled limitation text for: {', '.join(unsupported)}")
    facts = internal["facts"]
    limitations = []
    if any(check.startswith("camera_imu_coverage") for check in material):
        streams = [stream for stream in facts["timing"]["streams"]
                   if stream["outside_imu_coverage_rows"]]
        count = sum(stream["outside_imu_coverage_rows"] for stream in streams)
        details = "; ".join(
            f"{stream['camera_stream_id']}: {stream['outside_imu_coverage_rows']}"
            for stream in streams)
        limitations.append(
            f"{count} camera frame{'s are' if count != 1 else ' is'} outside IMU coverage"
            f" ({details})."
        )
    if "stereo_sequence_pairing" in material:
        count = facts["timing"]["stereo_unmatched_rows"]
        limitations.append(
            f"{count} camera frame{'s do' if count != 1 else ' does'} not have a stereo peer "
            "under unique equal encoder-sequence association."
        )
    if "telemetry" in material:
        limitations.append("Telemetry decoding did not complete.")
    return limitations


def _issue_ranges(rows, predicate):
    affected = [index for index, row in enumerate(rows) if predicate(row)]
    groups = []
    for index in affected:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    result = []
    for group in groups:
        first, last = rows[group[0]], rows[group[-1]]
        result.append({
            "position": "start" if group[0] == 0 else "end"
            if group[-1] == len(rows) - 1 else "interior",
            "frame_count": len(group),
            "start_frame": first["video_frame_index"],
            "end_frame": last["video_frame_index"],
            "start_time_s": None if first["mp4_pts_ns"] is None
            else first["mp4_pts_ns"] / 1e9,
            "end_time_s": None if last["mp4_pts_ns"] is None
            else last["mp4_pts_ns"] / 1e9,
        })
    return result


def timing_stream_facts(path: Path, expected_sha256: str) -> list[dict]:
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise QcError("Timing Parquet SHA-256 does not match its verified artifact")
    try:
        table = pq.read_table(path, columns=[
            "camera_stream_id", "video_frame_index", "mp4_pts_ns", "vts_match_status",
            "mapping_status", "stereo_pair_status"])
    except (OSError, pa.ArrowException) as error:
        raise QcError(f"Cannot read verified timing Parquet: {error}") from error
    rows = table.to_pylist()
    stream_ids = sorted({row["camera_stream_id"] for row in rows})
    if not stream_ids or None in stream_ids:
        raise QcError("Timing Parquet has no valid camera stream rows")
    facts = []
    for stream_id in stream_ids:
        stream_rows = [row for row in rows if row["camera_stream_id"] == stream_id]
        facts.append({
            "camera_stream_id": stream_id,
            "row_count": len(stream_rows),
            "matched_rows": sum(row["vts_match_status"] == "matched" for row in stream_rows),
            "coverage_rows": sum(row["mapping_status"] == "mapped" for row in stream_rows),
            "outside_imu_coverage_rows": sum(
                row["mapping_status"] == "outside_imu_coverage" for row in stream_rows),
            "missing_sof_rows": sum(row["mapping_status"] == "missing_sof" for row in stream_rows),
            "video_only_rows": sum(row["vts_match_status"] == "video_only" for row in stream_rows),
            "vts_only_rows": sum(row["vts_match_status"] == "vts_only" for row in stream_rows),
            "outside_imu_coverage_ranges": _issue_ranges(
                stream_rows, lambda row: row["mapping_status"] == "outside_imu_coverage"),
            "video_vts_mismatch_ranges": _issue_ranges(
                stream_rows, lambda row: row["vts_match_status"] != "matched"),
            "stereo_unmatched_ranges": _issue_ranges(
                stream_rows, lambda row: row["stereo_pair_status"] not in
                ("matched", "not_applicable")),
        })
    return facts


def _supplier_qc(internal: dict, decision: dict) -> dict:
    blocked, _ = supplier_issues(internal)
    if decision["status"] != "include":
        raise QcError("Supplier projection requires an explicit include decision")
    if not decision["decided_at"]:
        raise QcError("Include decision requires decided_at")
    if blocked:
        raise QcError(f"Supplier projection blocked by: {', '.join(blocked)}")
    facts = internal["facts"]
    streams = []
    for stream in facts["streams"]:
        stream_id = stream["camera_stream_id"]
        timing = next(item for item in facts["timing"]["streams"]
                      if item["camera_stream_id"] == stream_id)
        streams.append({
            "camera_stream_id": stream_id,
            "video": {"codec": stream["video"]["codec"],
                      "width": stream["video"]["width"],
                      "height": stream["video"]["height"],
                      "fps": stream["video"]["average_frame_rate"],
                      "duration_s": stream["video"]["duration_ns"] / 1e9,
                      "decoded_frames": stream["video"]["frame_count"],
                      "full_decode_completed": True},
            "vts_frames": stream["vts"]["frame_count"],
            "timing": {key: timing[key] for key in (
                "row_count", "matched_rows", "coverage_rows", "outside_imu_coverage_rows",
                "video_only_rows", "vts_only_rows", "outside_imu_coverage_ranges",
                "video_vts_mismatch_ranges")},
        })
    native = facts["imu"]["native"]
    supplier = {
        "schema_version": "actuate_delivery.qc_facts.v1",
        "integrity": {"raw_file_count": facts["source"]["file_count"],
                      "raw_bytes": facts["source"]["bytes"]},
        "imu": {"sample_count": facts["imu"]["sample_count"],
                "measured_sample_rate_hz": native["measured_sample_rate_hz"],
                "first_sample_timestamp_ns": native["first_sample_timestamp_ns"],
                "last_sample_timestamp_ns": native["last_sample_timestamp_ns"]},
        "camera_streams": streams,
        "timing_basis": {
            "camera_time": "native_vts_sof_timestamp",
            "imu_query": "before_after_closest_native_samples",
        },
    }
    if facts["capture_layout"] == "stereo_pair":
        supplier["stereo"] = {
            "association_basis": "unique_equal_venc_seq",
            "paired_frames": facts["timing"]["stereo_pair_count"],
            "unmatched_frames": facts["timing"]["stereo_unmatched_rows"],
            "unmatched_ranges": {
                stream["camera_stream_id"]: next(
                    item for item in facts["timing"]["streams"]
                    if item["camera_stream_id"] == stream["camera_stream_id"]
                )["stereo_unmatched_ranges"] for stream in facts["streams"]
            },
        }
    if facts["telemetry"]["status"] == "decoded":
        supplier["telemetry"] = {"record_count": facts["telemetry"]["record_count"]}
    return supplier


def build_qc(facts: dict, output: Path) -> QcArtifact:
    checks = []
    def add(check_id, result, evidence):
        assert result in ("pass", "fail", "unknown", "not_applicable")
        checks.append({"check_id": check_id, "result": result, "evidence": evidence})
    grouping = facts["grouping_status"]
    add("capture_structure", "pass" if grouping == "complete" else "fail", {"grouping_status": grouping})
    verified = facts["source"]["all_hashes_verified_in_current_run"]
    add("source_integrity", "pass" if verified else "fail",
        {"verified_members": facts["source"]["verified_members"],
         "file_count": facts["source"]["file_count"]})
    imu = facts["imu"]
    add("imu_decode", "pass" if imu["status"] == "decoded" else (
        "fail" if imu["status"] == "failed" else "unknown"), imu)
    for stream in facts["streams"]:
        stream_id = stream["camera_stream_id"]
        vts, video = stream["vts"], stream["video"]
        add(f"vts_decode:{stream_id}", "pass" if vts["status"] == "decoded" else (
            "fail" if vts["status"] == "failed" else "unknown"), vts)
        add(f"video_decode:{stream_id}", "pass" if video["status"] == "verified" else (
            "fail" if video["status"] == "failed" else "unknown"), video)
        comparable = vts["status"] == "decoded" and video["status"] == "verified"
        counts = {"vts_frame_count": vts.get("frame_count"), "video_frame_count": video.get("frame_count")}
        add(f"video_vts_frame_count:{stream_id}",
            "pass" if comparable and counts["vts_frame_count"] == counts["video_frame_count"]
            else "fail" if comparable else "unknown", counts)
    timing = facts["timing"]
    add("timing_artifact", "pass" if timing["status"] == "ready" else (
        "fail" if timing["status"] == "failed" else "unknown"), timing)
    ready = timing["status"] == "ready"
    expected_rows = sum(max(stream["vts"].get("frame_count") or 0, stream["video"].get("frame_count") or 0)
                        for stream in facts["streams"])
    add("timing_row_accounting",
        "pass" if ready and timing["row_count"] == expected_rows
        else "fail" if ready else "unknown",
        {"timing_rows": timing.get("row_count"), "expected_rows": expected_rows})
    matched = timing.get("matched_rows")
    coverage = timing.get("coverage_rows")
    add("camera_imu_coverage",
        "pass" if ready and matched and coverage == matched
        else "fail" if ready and matched else "unknown",
        {"matched_camera_rows": matched, "rows_within_imu_coverage": coverage,
         "rows_outside_imu_coverage": None if matched is None or coverage is None
         else matched - coverage})
    for stream in timing.get("streams", ()):
        stream_matched, stream_coverage = stream["matched_rows"], stream["coverage_rows"]
        add(f"camera_imu_coverage:{stream['camera_stream_id']}",
            "pass" if stream_matched and stream_coverage == stream_matched
            else "fail" if stream_matched else "unknown", stream)
    if facts["capture_layout"] != "stereo_pair":
        stereo_result = "not_applicable"
    elif not ready:
        stereo_result = "unknown"
    else:
        stereo_result = "pass" if timing["stereo_unmatched_rows"] == 0 else "fail"
    add("stereo_sequence_pairing", stereo_result, {"stereo_pair_count": timing.get("stereo_pair_count"),
         "stereo_unmatched_rows": timing.get("stereo_unmatched_rows")})
    telemetry = facts["telemetry"]
    add("telemetry", "not_applicable" if telemetry["status"] == "absent" else (
        "pass" if telemetry["status"] == "decoded" else "fail"), telemetry)
    counts = {result: sum(check["result"] == result for check in checks)
              for result in ("pass", "fail", "unknown", "not_applicable")}
    document = {"schema_version": "actuate_delivery.qc_internal.v2",
                "capture_id": facts["capture_id"], "facts": facts,
                "checks": checks, "summary": counts}
    try:
        encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise QcError(f"QC facts are not valid JSON: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        staging.write_bytes(encoded)
        if json.loads(staging.read_text()) != document:
            raise QcError("QC JSON does not match the computed facts and checks")
        output_hash = sha256(staging.read_bytes()).hexdigest()
        staging.replace(output)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QcError(f"QC JSON write or read failed: {error}") from error
    finally:
        staging.unlink(missing_ok=True)
    return QcArtifact(output_hash, counts["pass"], counts["fail"], counts["unknown"], counts["not_applicable"])
