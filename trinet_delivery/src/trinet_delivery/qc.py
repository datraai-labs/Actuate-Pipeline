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
    if internal["schema_version"] != "trinet_delivery.qc_internal.v2":
        raise QcError(f"Unsupported internal QC schema: {internal['schema_version']}")
    checks = {check["check_id"]: check["result"] for check in internal["checks"]}
    unresolved = [name for name, result in checks.items()
                  if result not in ("pass", "not_applicable")]
    blocked = [name for name in unresolved
               if any(name == prefix or name.startswith(prefix) for prefix in SUPPLIER_BLOCKERS)]
    return blocked, [name for name in unresolved if name not in blocked]


def timing_stream_facts(path: Path, expected_sha256: str) -> list[dict]:
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise QcError("Timing Parquet SHA-256 does not match its verified artifact")
    try:
        table = pq.read_table(path, columns=[
            "camera_stream_id", "vts_match_status", "mapping_status"])
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
        })
    return facts


def _supplier_qc(internal: dict, decision: dict) -> dict:
    blocked, material = supplier_issues(internal)
    if decision["status"] != "include":
        raise QcError("Supplier projection requires an explicit include decision")
    if not decision["decided_by"] or not decision["decided_at"]:
        raise QcError("Include decision requires decided_by and decided_at")
    checks = {check["check_id"]: check["result"] for check in internal["checks"]}
    if blocked:
        raise QcError(f"Supplier projection blocked by: {', '.join(blocked)}")
    limitations = decision["limitations"]
    if material and not limitations:
        raise QcError(f"Material checks require declared limitations: {', '.join(material)}")
    facts = internal["facts"]
    supplier_checks = [
        {"check": "source_integrity", "result": checks["source_integrity"],
         "evidence": {"raw_file_count": facts["source"]["file_count"],
                      "raw_bytes": facts["source"]["bytes"]}},
        {"check": "imu_decode", "result": checks["imu_decode"],
         "evidence": {"sample_count": facts["imu"]["sample_count"]}},
    ]
    for stream in facts["streams"]:
        stream_id = stream["camera_stream_id"]
        timing = next(item for item in facts["timing"]["streams"]
                      if item["camera_stream_id"] == stream_id)
        supplier_checks.extend((
            {"check": "video_decode", "stream": stream_id,
             "result": checks[f"video_decode:{stream_id}"],
             "evidence": {"decoded_frames": stream["video"]["frame_count"]}},
            {"check": "video_vts_frame_count", "stream": stream_id,
             "result": checks[f"video_vts_frame_count:{stream_id}"],
             "evidence": {"video_frames": stream["video"]["frame_count"],
                          "vts_frames": stream["vts"]["frame_count"]}},
            {"check": "camera_imu_coverage", "stream": stream_id,
             "result": checks[f"camera_imu_coverage:{stream_id}"],
             "evidence": {"matched_frames": timing["matched_rows"],
                          "covered_frames": timing["coverage_rows"]}},
        ))
    supplier = {
        "schema_version": "trinet_delivery.qc.v1", "capture_id": internal["capture_id"],
        "result": "pass_with_declared_limitation" if limitations else "pass",
        "checks": supplier_checks, "limitations": limitations, "transformations": [],
        "human_decision": {key: decision[key] for key in ("status", "decided_by", "decided_at")},
    }
    if facts["capture_layout"] == "stereo_pair":
        supplier["stereo"] = {
            "association_basis": "unique_equal_venc_seq",
            "paired_frames": facts["timing"]["stereo_pair_count"],
            "unmatched_frames": facts["timing"]["stereo_unmatched_rows"],
            "result": checks["stereo_sequence_pairing"],
        }
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
    document = {"schema_version": "trinet_delivery.qc_internal.v2",
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
