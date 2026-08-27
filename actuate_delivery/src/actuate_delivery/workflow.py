import json
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from actuate_delivery.run import (
    RunError,
    RunInputError,
    _delivery_review,
    complete_local_delivery,
    prepare_inventory,
    prepare_progress,
    process_imus,
    process_qc,
    process_sidecars,
    process_timing,
    process_videos,
    processing_progress,
    stage_progress_items,
)

STAGES = (
    ("inventory", "Inventory and preserve"),
    ("sensors", "Decode sensors"),
    ("video", "Verify video"),
    ("timing", "Align timing"),
    ("qc", "Run QC"),
    ("review", "Human review"),
    ("delivery", "Build delivery"),
)
APPROVAL_STAGES = tuple(stage for stage, _ in STAGES[:-1])


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    summary: dict
    output_signature: str


def _ensure_table(database_path: Path):
    with sqlite3.connect(database_path) as database:
        database.execute(
            """CREATE TABLE IF NOT EXISTS stage_checkpoint (
                   stage TEXT PRIMARY KEY,
                   status TEXT NOT NULL CHECK (status IN ('waiting', 'running', 'complete', 'failed')),
                   result_json TEXT,
                   output_signature TEXT,
                   approved_signature TEXT,
                   approved_by TEXT,
                   approved_at TEXT,
                   started_at TEXT,
                   completed_at TEXT,
                   error TEXT,
                   CHECK (stage IN ('inventory', 'sensors', 'video', 'timing', 'qc', 'review', 'delivery'))
               )"""
        )
        database.executemany(
            "INSERT OR IGNORE INTO stage_checkpoint (stage, status) VALUES (?, 'waiting')",
            [(stage,) for stage, _ in STAGES],
        )


def _signature(database_path: Path, stage: str):
    tables = {
        "inventory": (
            ("source_file", (
                "SELECT source_item_id, relative_path, role, size_bytes, selected, "
                "source_sha256, cache_relative_path FROM source_file "
                "WHERE selected=1 ORDER BY source_item_id"
            )),
            ("capture_candidate", "SELECT * FROM capture_candidate ORDER BY parent_path, capture_key"),
            ("capture_snapshot", "SELECT * FROM capture_snapshot ORDER BY parent_path, capture_key"),
        ),
        "sensors": (
            ("imu_artifact", "SELECT * FROM imu_artifact ORDER BY capture_id"),
            ("vts_artifact", "SELECT * FROM vts_artifact ORDER BY capture_id, camera_stream_id"),
            ("tel_artifact", "SELECT * FROM tel_artifact ORDER BY capture_id"),
        ),
        "video": (("video_artifact", "SELECT * FROM video_artifact ORDER BY capture_id, camera_stream_id"),),
        "timing": (("timing_artifact", "SELECT * FROM timing_artifact ORDER BY capture_id"),),
        "qc": (("qc_artifact", "SELECT * FROM qc_artifact ORDER BY capture_id"),),
        "review": (
            ("qc_artifact", "SELECT capture_id, json_sha256 FROM qc_artifact ORDER BY capture_id"),
            ("delivery_decision", "SELECT * FROM delivery_decision ORDER BY capture_id"),
        ),
        "delivery": (),
    }
    payload = []
    with sqlite3.connect(database_path) as database:
        existing = {row[0] for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table, query in tables[stage]:
            payload.append((table, database.execute(query).fetchall() if table in existing else []))
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _record(database_path: Path, stage: str, status: str, summary=None, error=None):
    _ensure_table(database_path)
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as database:
        if status == "running":
            database.execute(
                "UPDATE stage_checkpoint SET status='running', started_at=?, completed_at=NULL, error=NULL WHERE stage=?",
                (now, stage),
            )
            return ""
        signature = _signature(database_path, stage) if status == "complete" else None
        prior = database.execute(
            "SELECT output_signature FROM stage_checkpoint WHERE stage=?", (stage,)
        ).fetchone()[0]
        if status == "failed" or signature != prior:
            index = next(index for index, item in enumerate(STAGES) if item[0] == stage)
            database.executemany(
                """UPDATE stage_checkpoint SET status='waiting', result_json=NULL,
                          output_signature=NULL, approved_signature=NULL, approved_by=NULL,
                          approved_at=NULL, started_at=NULL, completed_at=NULL, error=NULL
                   WHERE stage=?""",
                [(item[0],) for item in STAGES[index + 1:]],
            )
        if status == "failed":
            database.execute(
                """UPDATE stage_checkpoint SET status='failed', result_json=NULL,
                          output_signature=NULL, approved_signature=NULL,
                          approved_by=NULL, approved_at=NULL, completed_at=?, error=?
                   WHERE stage=?""",
                (now, error, stage),
            )
            return ""
        database.execute(
            """UPDATE stage_checkpoint SET status='complete', result_json=?, output_signature=?,
                      approved_signature=CASE WHEN approved_signature=? THEN approved_signature END,
                      approved_by=CASE WHEN approved_signature=? THEN approved_by END,
                      approved_at=CASE WHEN approved_signature=? THEN approved_at END,
                      completed_at=?, error=NULL WHERE stage=?""",
            (json.dumps(summary, sort_keys=True), signature, signature, signature, signature,
             now, stage),
        )
    return signature


def workflow_state(run_dir: Path):
    database_path = run_dir / "run.sqlite"
    if not database_path.is_file():
        return [{"stage": stage, "name": name, "status": "waiting", "approved": False,
                 "summary": None, "error": None, "elapsed_seconds": None,
                 "progress": {"completed": 0, "total": 0, "current": None, "items": []}}
                for stage, name in STAGES]
    _ensure_table(database_path)
    with sqlite3.connect(database_path) as database:
        database.row_factory = sqlite3.Row
        rows = {row["stage"]: row for row in database.execute(
            "SELECT * FROM stage_checkpoint")}
    now = datetime.now(UTC)
    result = [{
        "stage": stage,
        "name": name,
        "status": rows[stage]["status"],
        "approved": bool(rows[stage]["output_signature"]
                         and rows[stage]["approved_signature"] == rows[stage]["output_signature"]),
        "summary": json.loads(rows[stage]["result_json"]) if rows[stage]["result_json"] else None,
        "error": rows[stage]["error"],
        "approved_by": rows[stage]["approved_by"],
        "approved_at": rows[stage]["approved_at"],
        "started_at": rows[stage]["started_at"],
        "completed_at": rows[stage]["completed_at"],
        "progress": processing_progress(database_path, stage),
    } for stage, name in STAGES]
    for item in result:
        started = item["started_at"]
        completed = item["completed_at"]
        item["elapsed_seconds"] = (round((datetime.fromisoformat(completed) -
                                           datetime.fromisoformat(started)).total_seconds(), 1)
                                   if started and completed else
                                   round((now - datetime.fromisoformat(started)).total_seconds(), 1)
                                   if started else None)
    return result


def artifact_failures(run_dir: Path, workflow_stage: str | None = None):
    database_path = run_dir / "run.sqlite"
    if not database_path.is_file():
        return []
    stage_tables = {
        "sensors": ("imu_artifact", "vts_artifact", "tel_artifact"),
        "video": ("video_artifact",),
        "timing": ("timing_artifact",),
        "qc": ("qc_artifact",),
    }
    wanted = stage_tables.get(workflow_stage, tuple(
        table for tables in stage_tables.values() for table in tables))
    failures = []
    with sqlite3.connect(database_path) as database:
        database.row_factory = sqlite3.Row
        tables = {row[0] for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in wanted:
            if table not in tables:
                continue
            columns = {row[1] for row in database.execute(f"PRAGMA table_info({table})")}
            message = "error" if "error" in columns else "reason"
            stream = "camera_stream_id" if "camera_stream_id" in columns else None
            selected = ["capture_id", message]
            if stream:
                selected.insert(1, stream)
            for row in database.execute(
                    f"SELECT {', '.join(selected)} FROM {table} WHERE status='failed'"):
                capture = database.execute(
                    """SELECT parent_path, capture_key FROM capture_snapshot
                       WHERE capture_id=? AND is_canonical=1""",
                    (row["capture_id"],),
                ).fetchone()
                failures.append({
                    "stage": table.removesuffix("_artifact"),
                    "episode": f"{capture['parent_path']}/{capture['capture_key']}"
                    if capture else row["capture_id"],
                    "camera_stream_id": row[stream] if stream else None,
                    "message": row[message] or "No error detail was recorded",
                })
    return failures


def approve_stage(run_dir: Path, stage: str, approved_by: str = ""):
    if stage not in APPROVAL_STAGES:
        raise RunError(f"Stage does not require approval: {stage}")
    approved_by = approved_by.strip()
    states = workflow_state(run_dir)
    current = next(item for item in states if item["stage"] == stage)
    if current["status"] != "complete":
        raise RunError(f"Stage is not complete: {stage}")
    index = APPROVAL_STAGES.index(stage)
    if any(not item["approved"] for item in states[:index]):
        raise RunError("Earlier stages must be approved first")
    if stage == "review" and current["summary"]["pending"]:
        raise RunError("Every eligible episode requires a review decision")
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        database.execute(
            """UPDATE stage_checkpoint SET approved_signature=output_signature,
                      approved_by=?, approved_at=? WHERE stage=?""",
            (approved_by, now, stage),
        )


def invalidate_from(run_dir: Path, stage: str):
    database_path = run_dir / "run.sqlite"
    if not database_path.is_file():
        return
    _ensure_table(database_path)
    index = next(index for index, item in enumerate(STAGES) if item[0] == stage)
    with sqlite3.connect(database_path) as database:
        database.executemany(
            """UPDATE stage_checkpoint SET status='waiting', result_json=NULL,
                      output_signature=NULL, approved_signature=NULL, approved_by=NULL,
                      approved_at=NULL, started_at=NULL, completed_at=NULL, error=NULL
               WHERE stage=?""",
            [(item[0],) for item in STAGES[index:]],
        )
        tables = {row[0] for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "processing_item" in tables:
            database.executemany(
                "DELETE FROM processing_item WHERE stage=?",
                [(item[0],) for item in STAGES[index:]],
            )


def _require_previous(run_dir: Path, stage: str):
    index = next(index for index, item in enumerate(STAGES) if item[0] == stage)
    if index and not workflow_state(run_dir)[index - 1]["approved"]:
        raise RunError(f"Approve {STAGES[index - 1][1]} before running {STAGES[index][1]}")


def _build_archive(output: Path):
    archive_path = output.with_suffix(".zip")
    if archive_path.exists() or archive_path.is_symlink():
        raise RunError(f"Delivery archive already exists: {archive_path}")
    staging = archive_path.with_name(f".{archive_path.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in sorted(output.rglob("*")):
                if (path.is_file() and path.name != ".DS_Store"
                        and "__MACOSX" not in path.parts):
                    archive.write(path, output.name / path.relative_to(output))
        with zipfile.ZipFile(staging) as archive:
            if archive.testzip() is not None:
                raise RunError("Delivery ZIP failed integrity verification")
        staging.replace(archive_path)
    finally:
        staging.unlink(missing_ok=True)
    return archive_path


def bind_calibration(run_dir: Path, template: dict):
    _, entries = _delivery_review(run_dir / "run.sqlite", run_dir)
    episode_ids = [entry["row"]["episode_id"] for entry in entries
                   if entry["row"]["grouping_status"] == "complete"
                   and entry["decision"] and entry["decision"]["status"] == "include"]
    calibration = dict(template)
    calibration["applies_to_episode_ids"] = episode_ids
    staging = run_dir / ".calibration.json.staging"
    staging.write_text(json.dumps(calibration, indent=2) + "\n")
    staging.replace(run_dir / "calibration.json")


def run_stage(source: str, run_dir: Path, stage: str, output: Path | None = None):
    if stage not in {item[0] for item in STAGES}:
        raise RunError(f"Unknown workflow stage: {stage}")
    if stage != "inventory":
        _require_previous(run_dir, stage)
    database_path = run_dir / "run.sqlite"
    if stage != "inventory":
        _record(database_path, stage, "running")
    try:
        if stage == "inventory":
            run_id, inventory, preservation = prepare_inventory(source, run_dir)
            _record(database_path, stage, "running")
            files, _, removed = preservation
            statuses = {name: sum(file.change == name for file in files)
                        for name in ("new", "changed", "unchanged")}
            summary = {
                "run_id": run_id,
                "files": len(inventory.files),
                "captures": len(inventory.captures),
                "complete_captures": sum(capture.grouping_status == "complete"
                                         for capture in inventory.captures),
                "incomplete_captures": sum(capture.grouping_status != "complete"
                                           for capture in inventory.captures),
                "removed": removed,
                **statuses,
            }
        elif stage == "sensors":
            prepare_progress(database_path, stage, stage_progress_items(database_path, stage))
            imu = process_imus(database_path, run_dir)
            sidecars = process_sidecars(database_path, run_dir)
            summary = dict(zip(("imu_decoded", "imu_reused", "imu_failed",
                                "vts_decoded", "vts_reused", "vts_failed",
                                "tel_decoded", "tel_reused", "tel_failed"), imu + sidecars))
            with sqlite3.connect(database_path) as database:
                totals = database.execute(
                    """SELECT COALESCE(SUM(sample_count),0) AS imu_samples,
                              (SELECT COALESCE(SUM(frame_count),0) FROM vts_artifact
                               WHERE status='decoded') AS camera_timestamps,
                              (SELECT COALESCE(SUM(record_count),0) FROM tel_artifact
                               WHERE status='decoded') AS telemetry_records
                       FROM imu_artifact WHERE status='decoded'"""
                ).fetchone()
            summary.update(dict(zip(
                ("imu_samples", "camera_timestamps", "telemetry_records"), totals)))
        elif stage == "video":
            prepare_progress(database_path, stage, stage_progress_items(database_path, stage))
            values = process_videos(database_path, run_dir)
            summary = dict(zip(("verified", "reused", "failed"), values))
            with sqlite3.connect(database_path) as database:
                streams, frames = database.execute(
                    "SELECT COUNT(*), COALESCE(SUM(frame_count),0) FROM video_artifact "
                    "WHERE status='verified'"
                ).fetchone()
            summary.update({"verified_streams": streams, "decoded_frames": frames})
        elif stage == "timing":
            prepare_progress(database_path, stage, stage_progress_items(database_path, stage))
            values = process_timing(database_path, run_dir)
            summary = dict(zip(("created", "reused", "unavailable", "failed"), values))
            with sqlite3.connect(database_path) as database:
                totals = database.execute(
                    """SELECT COALESCE(SUM(row_count),0), COALESCE(SUM(matched_rows),0),
                              COALESCE(SUM(coverage_rows),0),
                              COALESCE(SUM(stereo_pair_count),0),
                              COALESCE(SUM(stereo_unmatched_rows),0)
                       FROM timing_artifact WHERE status='ready'"""
                ).fetchone()
            summary.update(dict(zip(("frame_rows", "matched_rows", "imu_coverage_rows",
                                     "stereo_pairs", "unmatched_stereo_frames"), totals)))
        elif stage == "qc":
            prepare_progress(database_path, stage, stage_progress_items(database_path, stage))
            values = process_qc(database_path, run_dir)
            summary = dict(zip(("created", "reused", "failed"), values))
            if summary["failed"]:
                raise RunError(f"QC failed for {summary['failed']} capture(s)")
            review_path, entries = _delivery_review(database_path, run_dir)
            summary["review_path"] = str(review_path)
            summary["episodes"] = len(entries)
            eligible = [entry for entry in entries
                        if entry["row"]["grouping_status"] == "complete"]
            summary.update({
                "eligible_episodes": len(eligible),
                "passed_checks": sum(int(entry["row"]["pass_count"]) for entry in eligible),
                "failed_checks": sum(int(entry["row"]["fail_count"]) for entry in eligible),
                "unknown_checks": sum(int(entry["row"]["unknown_count"]) for entry in eligible),
                "blocking_checks": sorted({check for entry in eligible
                                           for check in entry["row"]["blocking_checks"].split("|")
                                           if check}),
                "material_checks": sorted({check for entry in eligible
                                           for check in entry["row"]["material_checks"].split("|")
                                           if check}),
            })
        elif stage == "review":
            review_path, entries = _delivery_review(database_path, run_dir)
            entries = tuple(entry for entry in entries
                            if entry["row"]["grouping_status"] == "complete")
            included = sum(bool(entry["decision"])
                           and entry["decision"]["status"] == "include"
                           for entry in entries)
            excluded = sum(bool(entry["decision"])
                           and entry["decision"]["status"] == "exclude"
                           for entry in entries)
            summary = {"review_path": str(review_path), "episodes": len(entries),
                       "included": included, "excluded": excluded,
                       "pending": len(entries) - included - excluded}
        else:
            if output is None:
                raise RunError("Delivery output path is required")
            result = complete_local_delivery(source, run_dir, output)
            if result.status != "complete":
                raise RunError(f"Delivery cannot be built: {result.status}")
            archive = _build_archive(result.output)
            summary = {"included": result.included, "excluded": result.excluded,
                       "output": str(result.output), "archive": str(archive),
                       "archive_bytes": archive.stat().st_size}
        altered = [failure for failure in artifact_failures(run_dir, stage)
                   if "changed after verification" in failure["message"]]
        if altered:
            raise RunError(altered[0]["message"])
        signature = _record(database_path, stage, "complete", summary)
        return StageResult(stage, "complete", summary, signature)
    except Exception as error:
        if database_path.is_file() and not isinstance(error, RunInputError):
            _record(database_path, stage, "failed", error=str(error))
        if isinstance(error, RunError):
            raise
        raise RunError(str(error)) from error
