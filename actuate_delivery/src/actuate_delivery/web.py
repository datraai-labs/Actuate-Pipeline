import csv
import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from hashlib import sha256
from math import ceil
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pyarrow import parquet as pq
from pydantic import BaseModel

from actuate_delivery.package import _vendor_visualizations
from actuate_delivery.qc import controlled_limitations
from actuate_delivery.run import (
    RunError,
    _delivery_review,
    _write_review,
    interrupt_progress,
    telemetry_policy,
)
from actuate_delivery.workflow import (
    _record,
    approve_stage,
    bind_calibration,
    invalidate_from,
    run_stage,
    workflow_state,
)


class Decision(BaseModel):
    qc_sha256: str
    expected_revision: str
    status: str


class SourceFile(BaseModel):
    relative_path: str
    size: int


class SelectedSourceFile(SourceFile):
    selected: bool


class BatchCreate(BaseModel):
    name: str
    files: list[SelectedSourceFile]
    use_configured_calibration: bool = False


class BatchPreflight(BaseModel):
    files: list[SourceFile]


class Approval(BaseModel):
    approved_by: str = ""


class TelemetryChoice(BaseModel):
    choice: str


class WebRun:
    def __init__(self, source: Path, run_dir: Path, output: Path):
        self.source = source.resolve()
        self.run_dir = run_dir.resolve()
        self.output = output.resolve()
        self.lock = Lock()
        self.job_status = "idle"
        self.job_error = None
        self.started_at = None
        self.finished_at = None
        self.started_clock = None
        self.elapsed_seconds = None
        self.job_stage = None
        self._load_job()

    @property
    def job_path(self):
        return self.run_dir / "web_job.json"

    def _load_job(self):
        if not self.job_path.is_file():
            return
        saved = json.loads(self.job_path.read_text())
        self.job_status = saved["status"]
        self.job_error = saved.get("error")
        self.started_at = saved.get("started_at")
        self.finished_at = saved.get("finished_at")
        self.elapsed_seconds = saved.get("elapsed_seconds")
        self.job_stage = saved.get("stage")
        if self.job_status == "running":
            self.job_status = "failed"
            self.job_error = "Processing was interrupted when the service restarted. Retry processing; uploaded files and completed artifacts are preserved."
            self.finished_at = datetime.now(timezone.utc).isoformat()
            if self.job_stage:
                interrupt_progress(self.run_dir / "run.sqlite", self.job_stage, self.job_error)
            self._save_job()

    def _save_job(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        staging = self.job_path.with_suffix(".staging")
        staging.write_text(json.dumps({
            "status": self.job_status,
            "error": self.job_error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "stage": self.job_stage,
        }, indent=2) + "\n")
        staging.replace(self.job_path)

    def start(self, stage):
        with self.lock:
            if self.job_status == "running":
                raise HTTPException(409, "Processing is already running")
            if self.output.exists() and stage != "archive":
                raise HTTPException(409, "This batch is already delivered and cannot be reprocessed")
            if stage == "archive" and not self.output.is_dir():
                raise HTTPException(409, "Build the customer folder before creating its ZIP")
            self.job_status = "running"
            self.job_stage = stage
            self.job_error = None
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.finished_at = None
            self.started_clock = monotonic()
            self.elapsed_seconds = None
            self._save_job()
        Thread(target=self._process, args=(stage,), daemon=True).start()

    def _process(self, stage):
        try:
            if stage == "archive":
                archive = self.build_archive()
                records, _ = self.episodes()
                _record(self.run_dir / "run.sqlite", "delivery", "complete", {
                    "included": sum(item["decision"]["status"] == "include"
                                    for item in records if item["decision"]),
                    "excluded": sum(item["decision"]["status"] == "exclude"
                                    for item in records if item["decision"]),
                    "output": str(self.output), "archive": str(self.archive),
                    "archive_bytes": archive["bytes"],
                    "archive_sha256": archive["sha256"],
                })
            else:
                run_stage(str(self.source), self.run_dir, stage,
                          self.output if stage == "delivery" else None)
                if stage == "delivery":
                    self.record_archive()
            status, error = "complete", None
        except (OSError, RunError) as exc:
            status, error = "failed", str(exc)
        with self.lock:
            self.job_status = status
            self.job_error = error
            self.finished_at = datetime.now(timezone.utc).isoformat()
            self.elapsed_seconds = round(monotonic() - self.started_clock, 1)
            self._save_job()

    def job(self):
        elapsed = (monotonic() - self.started_clock
                   if self.job_status == "running" and self.started_clock else self.elapsed_seconds)
        status = self.job_status
        failures = self.failure_count()
        error = self.job_error
        if status != "running" and self.output.is_dir():
            status = "delivered" if self.archive_record() else "complete"
            error = None
        elif status == "idle" and not self.source_files():
            status = "empty"
        elif status == "idle" and failures:
            status, error = "failed", f"Processing has {failures} failed artifacts"
        elif status == "idle" and self.archive.is_file():
            status = "delivered"
        elif status == "idle" and (self.run_dir / "run.sqlite").is_file():
            status = "processed"
        return {
            "status": status,
            "error": error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "stage": self.job_stage,
        }

    def source_files(self):
        if not self.source.is_dir():
            return []
        return [{"relative_path": path.relative_to(self.source).as_posix(),
                 "size": path.stat().st_size}
                for path in sorted(self.source.rglob("*"))
                if path.is_file() and not path.is_symlink()
                and "system volume information" not in {
                    part.casefold() for part in path.relative_to(self.source).parts}]

    def failures(self):
        database_path = self.run_dir / "run.sqlite"
        if not database_path.exists():
            return []
        failures = []
        with sqlite3.connect(database_path) as database:
            database.row_factory = sqlite3.Row
            tables = {row[0] for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("imu_artifact", "vts_artifact", "tel_artifact",
                          "video_artifact", "timing_artifact", "qc_artifact"):
                if table not in tables:
                    continue
                columns = {row[1] for row in database.execute(f"PRAGMA table_info({table})")}
                message = "error" if "error" in columns else "reason"
                stream = "camera_stream_id" if "camera_stream_id" in columns else None
                selected = ["capture_id", "status", message]
                if stream:
                    selected.insert(1, stream)
                for row in database.execute(
                        f"SELECT {', '.join(selected)} FROM {table} WHERE status='failed'"):
                    capture = database.execute(
                        """SELECT parent_path, capture_key FROM capture_snapshot
                           WHERE capture_id=? AND is_canonical=1""",
                        (row["capture_id"],),
                    ).fetchone() if "capture_snapshot" in tables else None
                    failures.append({
                        "stage": table.removesuffix("_artifact"),
                        "capture_id": row["capture_id"],
                        "episode": f"{capture['parent_path']}/{capture['capture_key']}"
                        if capture else None,
                        "camera_stream_id": row[stream] if stream else None,
                        "message": row[message] or "No error detail was recorded",
                    })
        return failures

    def candidates(self):
        database_path = self.run_dir / "run.sqlite"
        if not database_path.exists():
            return []
        with sqlite3.connect(database_path) as database:
            database.row_factory = sqlite3.Row
            tables = {row[0] for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "capture_candidate" not in tables:
                return []
            return [dict(row) for row in database.execute(
                """SELECT parent_path AS directory, capture_key AS 'group',
                          capture_layout AS layout, grouping_status, file_count
                   FROM capture_candidate ORDER BY parent_path, capture_key"""
            )]

    def failure_count(self):
        return len(self.failures())

    def steps(self):
        database_path = self.run_dir / "run.sqlite"
        if not database_path.exists():
            return [
                {"name": "Inventory and preserve", "status": "waiting",
                 "detail": "Waiting for processing to start"},
                *({"name": name, "status": "pending", "detail": detail} for name, detail in (
                    ("Decode sensors", "Decode IMU, VTS and telemetry"),
                    ("Verify video", "Probe and fully decode every camera stream"),
                    ("Align timing", "Map camera frames to IMU timestamps"),
                    ("Run QC", "Create per-episode facts and deterministic checks"),
                    ("Human review", "Include or exclude each episode"),
                    ("Build delivery", "Validate and package customer output"),
                )),
            ]
        with sqlite3.connect(database_path) as database:
            tables = {row[0] for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}

            def count(table, where="1"):
                if table not in tables:
                    return 0
                return database.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]

            files = count("source_file", "selected=1")
            preserved = count("source_file", "selected=1 AND source_sha256 IS NOT NULL")
            expected_sensors = count(
                "source_file", "selected=1 AND role IN ('imu', 'vts', 'telemetry')")
            expected_videos = count("source_file", "selected=1 AND role='video'")
            candidates = count("capture_snapshot", "is_canonical=1")
            captures = database.execute(
                """SELECT COUNT(*) FROM capture_snapshot JOIN capture_candidate
                   USING (parent_path, capture_key)
                   WHERE is_canonical=1 AND grouping_status='complete'"""
            ).fetchone()[0] if "capture_snapshot" in tables else 0
            sensors = sum(count(table, f"status='{status}'") for table, status in (
                ("imu_artifact", "decoded"), ("vts_artifact", "decoded"),
                ("tel_artifact", "decoded")))
            videos = count("video_artifact", "status='verified'")
            timing = count("timing_artifact", "status='ready'")
            qc = count("qc_artifact", "status='ready'")
            decisions = database.execute(
                """SELECT COUNT(*) FROM delivery_decision
                   JOIN capture_snapshot USING (capture_id)
                   JOIN capture_candidate USING (parent_path, capture_key)
                   WHERE is_canonical=1 AND grouping_status='complete'"""
            ).fetchone()[0] if "delivery_decision" in tables else 0
            sensor_failed = sum(count(table, "status='failed'") for table in (
                "imu_artifact", "vts_artifact", "tel_artifact"))
            video_failed = count("video_artifact", "status='failed'")
            timing_unavailable = count("timing_artifact", "status='unavailable'")
            timing_failed = count("timing_artifact", "status='failed'")
            qc_failed = count("qc_artifact", "status='failed'")
        sensor_done = sensors + sensor_failed
        video_done = videos + video_failed
        timing_done = timing + timing_unavailable + timing_failed
        qc_done = qc + qc_failed
        sensor_status = ("not_applicable" if not expected_sensors and not sensor_done else
                         "pending" if sensor_done < expected_sensors else
                         "failed" if sensor_failed else "complete")
        video_status = ("not_applicable" if not expected_videos and not video_done else
                        "pending" if video_done < expected_videos else
                        "failed" if video_failed else "complete")
        timing_status = ("not_applicable" if not captures else
                         "pending" if timing_done < captures else
                         "failed" if timing_failed or timing_unavailable else "complete")
        qc_status = ("pending" if qc_done < candidates else
                     "failed" if qc_failed else "complete")
        progress = [
            ("Inventory and preserve", "complete" if preserved == files and files > 0 else "pending",
             f"{preserved} of {files} selected source files verified and preserved"),
            ("Decode sensors", sensor_status,
             ("No IMU, VTS, or telemetry sidecars were recognized"
              if not expected_sensors else
              f"{sensor_done} of {max(expected_sensors, sensor_done)} sensor streams processed; "
              f"{sensor_failed} failed")),
            ("Verify video", video_status,
             (f"{video_done} of {max(expected_videos, video_done)} camera files processed; "
              f"{video_failed} failed")),
            ("Align timing", timing_status,
             f"{timing} of {captures} complete episodes mapped"),
            ("Run QC", qc_status,
             f"{qc} of {candidates} candidate reports ready"),
            ("Human review", "complete" if decisions == captures and captures > 0 else "pending",
             f"{decisions} of {captures} decisions saved"),
            ("Build delivery", "complete" if self.archive_record() else "pending",
             "Validated customer folder and ZIP are ready" if self.archive_record()
             else "Customer folder is ready; ZIP still needs to be built"
             if self.output.exists() else "Not built"),
        ]
        first_pending = next((index for index, item in enumerate(progress)
                              if item[1] == "pending"), None)
        result = [{"name": name,
                   "status": "running" if self.job_status == "running"
                   and index == first_pending else status,
                   "detail": detail} for index, (name, status, detail) in enumerate(progress)]
        if not captures and candidates:
            result[3]["detail"] = "No complete episodes are eligible for timing mapping"
            result[5] = {"name": "Human review", "status": "not_applicable",
                         "detail": "No complete episodes are eligible for review"}
            result[6] = {"name": "Build delivery", "status": "not_applicable",
                         "detail": "No complete episodes are eligible for delivery"}
        states = workflow_state(self.run_dir)
        for item, checkpoint in zip(result, states, strict=True):
            status = checkpoint["status"]
            if status == "complete" and checkpoint["approved"]:
                status = "approved"
            elif status == "complete" and checkpoint["stage"] != "delivery":
                status = "awaiting_approval"
            item["status"] = status
            item["stage"] = checkpoint["stage"]
            item["approved"] = checkpoint["approved"]
            item["summary"] = checkpoint["summary"]
            item["error"] = checkpoint["error"]
            item["elapsed_seconds"] = checkpoint["elapsed_seconds"]
            item["progress"] = checkpoint["progress"]
        if self.job_status == "running" and self.job_stage == "archive":
            result[-1].update(status="running", error=None)
        return result

    @property
    def archive(self):
        return self.output.with_suffix(".zip")

    @property
    def archive_record_path(self):
        return self.run_dir / "delivery_archive.json"

    def delivery_files(self):
        return [path for path in sorted(self.output.rglob("*"))
                if path.is_file() and path.name != ".DS_Store"
                and "__MACOSX" not in path.parts]

    def upload_paths(self, relative: PurePosixPath):
        destination = self.source.joinpath(*relative.parts)
        staging = self.run_dir / "cache/upload-staging" / sha256(str(relative).encode()).hexdigest()
        return destination, staging

    def build_archive(self):
        if not self.output.is_dir():
            raise RunError("Build the delivery before preparing its download")
        staging = self.archive.with_name(f".{self.archive.name}.staging")
        staging.unlink(missing_ok=True)
        try:
            with zipfile.ZipFile(staging, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
                for path in self.delivery_files():
                    archive.write(path, Path(self.output.name) / path.relative_to(self.output))
            with zipfile.ZipFile(staging) as archive:
                if archive.testzip() is not None:
                    raise RunError("Delivery ZIP failed integrity verification")
            staging.replace(self.archive)
        finally:
            staging.unlink(missing_ok=True)
        return self.record_archive()

    def record_archive(self):
        digest = sha256()
        with self.archive.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
        stat = self.archive.stat()
        record = {"bytes": stat.st_size, "modified_ns": stat.st_mtime_ns,
                  "sha256": digest.hexdigest()}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        staging = self.archive_record_path.with_suffix(".staging")
        staging.write_text(json.dumps(record, indent=2) + "\n")
        staging.replace(self.archive_record_path)
        return record

    def archive_record(self):
        if not self.archive.is_file() or not self.archive_record_path.is_file():
            return None
        record = json.loads(self.archive_record_path.read_text())
        stat = self.archive.stat()
        if (record["bytes"] != stat.st_size
                or record["modified_ns"] != stat.st_mtime_ns):
            return None
        return record

    def delivery_summary(self):
        archive = self.archive_record()
        files = []
        if self.output.is_dir():
            for path in self.delivery_files():
                relative = path.relative_to(self.output).as_posix()
                files.append({
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "kind": "raw" if "/raw/" in f"/{relative}" else "derived",
                })
        return {
            "exists": self.output.is_dir(),
            "downloadable": archive is not None,
            "archive_bytes": archive["bytes"] if archive else None,
            "archive_sha256": archive["sha256"] if archive else None,
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
            "raw_files": sum(item["kind"] == "raw" for item in files),
            "raw_bytes": sum(item["bytes"] for item in files if item["kind"] == "raw"),
            "derived_files": sum(item["kind"] == "derived" for item in files),
            "derived_bytes": sum(item["bytes"] for item in files if item["kind"] == "derived"),
            "files": files,
        }

    def has_preview(self, parent_path, capture_key):
        try:
            with sqlite3.connect(self.run_dir / "run.sqlite") as database:
                database.row_factory = sqlite3.Row
                return bool(_vendor_visualizations(database, parent_path, capture_key))
        except sqlite3.Error:
            return False

    def video_path(self, capture_id, stream):
        with sqlite3.connect(self.run_dir / "run.sqlite") as database:
            rows = database.execute(
                """SELECT cache_relative_path FROM capture_snapshot
                   JOIN capture_member USING (parent_path, capture_key)
                   JOIN source_file USING (source_item_id)
                   WHERE capture_id=? AND is_canonical=1 AND role='video'
                         AND camera_stream_id=?""",
                (capture_id, stream),
            ).fetchall()
        if len(rows) != 1:
            raise HTTPException(404, "Verified camera stream is not available")
        path = self.run_dir / rows[0][0]
        if not path.is_file():
            raise HTTPException(409, "Preserved camera stream is missing")
        return path

    def evidence(self, capture_id):
        with sqlite3.connect(self.run_dir / "run.sqlite") as database:
            database.row_factory = sqlite3.Row
            timing = database.execute(
                """SELECT parquet_relative_path, parquet_sha256 FROM timing_artifact
                   WHERE capture_id=? AND status='ready'""", (capture_id,)
            ).fetchone()
            imu = database.execute(
                """SELECT parquet_relative_path, parquet_sha256 FROM imu_artifact
                   WHERE capture_id=? AND status='decoded'""", (capture_id,)
            ).fetchone()
        if timing is None or imu is None:
            raise HTTPException(409, "Verified timing and IMU artifacts are required")
        paths = [self.run_dir / timing[0], self.run_dir / imu[0]]
        hashes = [timing[1], imu[1]]
        for path, expected in zip(paths, hashes, strict=True):
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
                raise HTTPException(409, f"Verified artifact changed: {path.name}")
        timing_rows = pq.read_table(paths[0]).to_pylist()
        imu_table = pq.read_table(paths[1])
        streams = sorted({row["camera_stream_id"] for row in timing_rows})
        if not streams:
            raise HTTPException(409, "Verified timing artifact contains no camera rows")
        frame_fields = (
            "camera_stream_id", "video_frame_index", "mp4_pts_ns", "vts_frame_number",
            "venc_seq", "sof_timestamp_ns", "vts_match_status", "before_imu_index",
            "before_imu_timestamp_ns", "before_delta_ns", "after_imu_index",
            "after_imu_timestamp_ns", "after_delta_ns", "closest_imu_index",
            "closest_imu_timestamp_ns", "closest_delta_ns", "within_imu_coverage",
            "mapping_status", "stereo_peer_stream_id", "stereo_peer_video_frame_index",
            "stereo_pair_status",
        )
        frames = [{key: row[key] for key in frame_fields} for row in timing_rows]
        issues = []
        conditions = (
            ("camera_imu_coverage", lambda row: not row["within_imu_coverage"],
             "Camera frame is outside the recorded IMU time extent"),
            ("video_vts_accounting", lambda row: row["vts_match_status"] != "matched",
             "Video frame and VTS row do not have a one-to-one row-order match"),
            ("stereo_pairing", lambda row: row["stereo_pair_status"] not in
             ("matched", "not_applicable"),
             "Camera frame has no unique equal encoder-sequence peer"),
        )
        for stream in streams:
            rows = [row for row in frames if row["camera_stream_id"] == stream]
            for kind, predicate, message in conditions:
                affected = [index for index, row in enumerate(rows) if predicate(row)]
                groups = []
                for index in affected:
                    if not groups or index != groups[-1][-1] + 1:
                        groups.append([index])
                    else:
                        groups[-1].append(index)
                for group in groups:
                    first, last = rows[group[0]], rows[group[-1]]
                    position = ("start" if group[0] == 0 else "end"
                                if group[-1] == len(rows) - 1 else "interior")
                    issues.append({
                        "kind": kind, "stream": stream, "position": position,
                        "frame_count": len(group),
                        "start_frame": first["video_frame_index"],
                        "end_frame": last["video_frame_index"],
                        "start_time_s": (first["mp4_pts_ns"] or 0) / 1e9,
                        "end_time_s": (last["mp4_pts_ns"] or 0) / 1e9,
                        "message": message, "evidence": first,
                    })
        primary = "left" if "left" in streams else streams[0]
        mapped = [row for row in frames if row["camera_stream_id"] == primary
                  and row["closest_imu_index"] is not None and row["mp4_pts_ns"] is not None]
        sampled = mapped[::max(1, ceil(len(mapped) / 1500))]
        columns = {name: imu_table[name].to_pylist() for name in (
            "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
            "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
        )}
        imu_plot = []
        for row in sampled:
            index = row["closest_imu_index"]
            imu_plot.append({"time_s": row["mp4_pts_ns"] / 1e9, "imu_index": index,
                             "closest_delta_ms": row["closest_delta_ns"] / 1e6,
                             **{name: values[index] for name, values in columns.items()}})
        frame_columns = {key: [row[key] for row in frames] for key in frame_fields}
        return {"streams": streams, "frames": frame_columns, "imu_plot": imu_plot,
                "issues": issues, "basis": {
                    "camera_time": "native VTS SOF timestamp",
                    "imu_query": "Before, After, and Closest native samples",
                    "stereo_pairing": "unique equal encoder sequence",
                    "physical_sync_certified": False,
                }}

    def episodes(self):
        database_path = self.run_dir / "run.sqlite"
        if not database_path.exists():
            return [], None
        with sqlite3.connect(database_path) as database:
            if database.execute("PRAGMA user_version").fetchone()[0] < 11:
                return [], None
        failures = self.failure_count()
        warning = (f"Processing has {failures} failed artifact(s). Complete episodes remain "
                   "reviewable; failed and incomplete candidates are excluded from delivery."
                   if failures else None)
        try:
            _, entries = _delivery_review(database_path, self.run_dir)
        except (OSError, RunError, sqlite3.Error) as exc:
            return [], str(exc)
        episodes = []
        for entry in entries:
            row = entry["row"]
            facts = entry["internal_qc"]["facts"]
            timing = facts["timing"]
            streams = [{
                "id": stream["camera_stream_id"],
                "frames": stream["video"].get("frame_count"),
                "codec": stream["video"].get("codec"),
                "width": stream["video"].get("width"),
                "height": stream["video"].get("height"),
                "duration_s": round((stream["video"].get("duration_ns") or 0) / 1e9, 3),
                "video": stream["video"],
                "vts": stream["vts"],
            } for stream in facts["streams"]]
            episodes.append({
                "episode_id": row["episode_id"],
                "capture_id": row["capture_id"],
                "directory": row["source_relative_directory"],
                "group": row["source_group"],
                "layout": row["capture_layout"],
                "grouping_status": row["grouping_status"],
                "source_files": facts["source"]["file_count"],
                "source_bytes": facts["source"]["bytes"],
                "streams": streams,
                "imu_samples": facts["imu"].get("sample_count"),
                "timing_rows": timing.get("row_count"),
                "coverage_rows": timing.get("coverage_rows"),
                "stereo_pairs": timing.get("stereo_pair_count"),
                "unmatched_rows": timing.get("stereo_unmatched_rows"),
                "timing": timing,
                "imu": facts["imu"],
                "telemetry": facts["telemetry"],
                "source_members": facts["source"]["members"],
                "has_preview": self.has_preview(entry["parent_path"], entry["capture_key"]),
                "checks": entry["internal_qc"]["checks"],
                "check_counts": {
                    "pass": row["pass_count"], "fail": row["fail_count"],
                    "unknown": row["unknown_count"],
                    "not_applicable": row["not_applicable_count"],
                },
                "blocking_checks": row["blocking_checks"].split("|") if row["blocking_checks"] else [],
                "material_checks": row["material_checks"].split("|") if row["material_checks"] else [],
                "controlled_limitations": (controlled_limitations(entry["internal_qc"])
                                           if row["grouping_status"] == "complete" else []),
                "qc_sha256": row["qc_sha256"],
                "decision": entry["decision"],
            })
        return episodes, warning

    def decide(self, capture_id: str, decision: Decision):
        with self.lock:
            if self.job_status == "running":
                raise HTTPException(409, "Wait for processing to finish before reviewing")
            if self.output.exists():
                raise HTTPException(409, "This delivery is already built and cannot be changed")
            if not next(item for item in workflow_state(self.run_dir)
                        if item["stage"] == "qc")["approved"]:
                raise HTTPException(409, "Approve QC before reviewing episodes")
            review_path, entries = _delivery_review(self.run_dir / "run.sqlite", self.run_dir)
            with review_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            row = next((row for row in rows if row["capture_id"] == capture_id), None)
            if row is None:
                raise HTTPException(404, "Episode is not current")
            if row["qc_sha256"] != decision.qc_sha256:
                raise HTTPException(409, "QC evidence changed; refresh before deciding")
            if row["decided_at"] != decision.expected_revision:
                raise HTTPException(409, "Decision changed; refresh before deciding")
            original = [dict(item) for item in rows]
            entry = next(item for item in entries if item["row"]["capture_id"] == capture_id)
            row.update({
                "decision": decision.status,
                "limitations_json": json.dumps(
                    controlled_limitations(entry["internal_qc"])
                    if decision.status == "include" else [],
                    separators=(",", ":"),
                ),
                "decided_at": decision.expected_revision,
            })
            try:
                _write_review(review_path, rows)
                _delivery_review(self.run_dir / "run.sqlite", self.run_dir)
                run_stage(str(self.source), self.run_dir, "review")
            except (OSError, RunError, sqlite3.Error) as exc:
                _write_review(review_path, original)
                _delivery_review(self.run_dir / "run.sqlite", self.run_dir)
                raise HTTPException(400, str(exc)) from exc


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise HTTPException(400, "Upload path is unsafe")
    if any(part.casefold() == "system volume information" for part in path.parts):
        raise HTTPException(400, "System Volume Information is ignored")
    return path


def _source_signature(files):
    return sorted((item["relative_path"], item["size"]) for item in files)


def _selection_manifest(files):
    manifest = [{
        "relative_path": str(_safe_relative(item.relative_path)),
        "size": item.size,
        "selected": item.selected,
    } for item in files]
    paths = [item["relative_path"] for item in manifest]
    if any(item["size"] < 0 for item in manifest):
        raise HTTPException(400, "Selection contains a negative file size")
    if len(paths) != len(set(paths)):
        raise HTTPException(400, "Selection contains a duplicate relative path")
    if not any(item["selected"] for item in manifest):
        raise HTTPException(400, "Select at least one file")
    return manifest


def _bind_calibration(web_run: WebRun, template: dict):
    bind_calibration(web_run.run_dir, template)


def _calibration_label(calibration: dict):
    method = calibration["source"]["method"]
    baseline_mm = calibration["transforms"]["baseline_m"] * 1000
    return f"Configured stereo calibration - {method} - {baseline_mm:.1f} mm baseline"


def create_app(source: Path, run_dir: Path, output: Path) -> FastAPI:
    source.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_batch_id = "initial"
    runs = {initial_batch_id: WebRun(source, run_dir, output)}
    names = {initial_batch_id: "New dataset"}
    selected_paths = {initial_batch_id: None}
    calibration_path = run_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
    calibration_selected = {initial_batch_id: calibration is not None}
    batch_root = run_dir.parent / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    for metadata_path in sorted(batch_root.glob("*/batch.json")):
        metadata = json.loads(metadata_path.read_text())
        batch_id = metadata_path.parent.name
        runs[batch_id] = WebRun(
            metadata_path.parent / "source", metadata_path.parent / "run",
            metadata_path.parent / "delivery")
        names[batch_id] = metadata["name"]
        selection = metadata.get("selection")
        selected_paths[batch_id] = ({item["relative_path"]: item["size"]
                                     for item in selection if item["selected"]}
                                    if selection else None)
        calibration_selected[batch_id] = metadata.get("use_configured_calibration", False)

    def batch(batch_id: str):
        if batch_id not in runs:
            raise HTTPException(404, "Batch does not exist")
        return runs[batch_id]

    def selected_path(batch_id: str, relative: PurePosixPath):
        allowed = selected_paths[batch_id]
        if allowed is not None and str(relative) not in allowed:
            raise HTTPException(409, "This file was not selected for this batch")
        return allowed[str(relative)] if allowed is not None else None
    review_file = Path(__file__).parents[2] / "review/index.html"
    if not review_file.is_file():
        review_file = Path(sys.prefix) / "share/actuate_delivery/index.html"
    app = FastAPI(title="Actuate")

    @app.get("/")
    def index():
        return FileResponse(review_file)

    @app.get("/api/batches")
    def batches():
        result = []
        for batch_id, web_run in runs.items():
            records, _ = web_run.episodes()
            source_files = web_run.source_files()
            candidates = web_run.candidates()
            result.append({
                "batch_id": batch_id, "name": names[batch_id],
                "status": web_run.job()["status"],
                "episodes": sum(item["grouping_status"] == "complete" for item in records),
                "incomplete": sum(item["grouping_status"] != "complete"
                                  for item in candidates),
                "delivery_ready": web_run.archive_record() is not None,
                "source_files": len(source_files),
                "source_bytes": sum(item["size"] for item in source_files),
            })
        return result

    @app.post("/api/batches/preflight")
    def preflight(request: BatchPreflight):
        selected = [{"relative_path": str(_safe_relative(item.relative_path)),
                     "size": item.size} for item in request.files]
        if not selected:
            raise HTTPException(400, "Select at least one file")
        signature = _source_signature(selected)
        matches = []
        for batch_id, web_run in runs.items():
            existing = _source_signature(web_run.source_files())
            if signature == existing:
                matches.append({
                    "batch_id": batch_id,
                    "name": names[batch_id],
                    "status": web_run.job()["status"],
                    "match": "same paths and sizes",
                })
        return {"matches": matches}

    @app.post("/api/batches")
    def create_batch(request: BatchCreate):
        name = request.name.strip()
        if not name:
            raise HTTPException(400, "Batch name is required")
        selection = _selection_manifest(request.files)
        batch_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        directory = batch_root / batch_id
        directory.mkdir()
        metadata = {
            "name": name,
            "use_configured_calibration": request.use_configured_calibration,
            "selection": selection,
        }
        (directory / "batch.json").write_text(json.dumps(metadata, indent=2) + "\n")
        runs[batch_id] = WebRun(directory / "source", directory / "run", directory / "delivery")
        names[batch_id] = name
        selected_paths[batch_id] = {item["relative_path"]: item["size"]
                                    for item in selection if item["selected"]}
        calibration_selected[batch_id] = request.use_configured_calibration
        return {"batch_id": batch_id, "name": name}

    @app.get("/api/state")
    def state(batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        records, review_error = web_run.episodes()
        episodes = [item for item in records if item["grouping_status"] == "complete"]
        incomplete = [item for item in records if item["grouping_status"] != "complete"]
        source_files = web_run.source_files()
        policy = None
        if episodes and all(item["decision"] for item in episodes):
            policy = telemetry_policy(web_run.run_dir)
        return {
            "batch": {"batch_id": batch_id, "name": names[batch_id]},
            "calibration": {
                "available": calibration is not None,
                "selected": calibration_selected[batch_id],
                "label": _calibration_label(calibration) if calibration else None,
            },
            "source": {"file_count": len(source_files),
                       "bytes": sum(item["size"] for item in source_files)},
            "job": web_run.job(),
            "steps": web_run.steps(),
            "workflow": workflow_state(web_run.run_dir),
            "failures": web_run.failures(),
            "candidates": web_run.candidates(),
            "episodes": episodes,
            "incomplete": incomplete,
            "review_error": review_error,
            "telemetry_policy": policy,
            "delivery": web_run.delivery_summary(),
        }

    @app.put("/api/upload/{relative_path:path}")
    async def upload(relative_path: str, size: int, request: Request,
                     batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        relative = _safe_relative(relative_path)
        expected_size = selected_path(batch_id, relative)
        if expected_size is not None and size != expected_size:
            raise HTTPException(400, "Upload size differs from the selected file")
        destination, staging = web_run.upload_paths(relative)
        if destination.exists() or destination.is_symlink():
            raise HTTPException(409, "Source file already exists and will not be overwritten")
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.unlink(missing_ok=True)
        digest, received = sha256(), 0
        try:
            with staging.open("xb") as file:
                async for chunk in request.stream():
                    file.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
            if received != size:
                raise HTTPException(400, "Uploaded byte count does not match the browser file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(destination)
            invalidate_from(web_run.run_dir, "inventory")
        finally:
            staging.unlink(missing_ok=True)
        return {"path": str(relative), "bytes": received, "sha256": digest.hexdigest()}

    @app.post("/api/upload/start/{relative_path:path}")
    def start_upload(relative_path: str, batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        relative = _safe_relative(relative_path)
        selected_path(batch_id, relative)
        _, staging = web_run.upload_paths(relative)
        if web_run.output.exists():
            raise HTTPException(409, "This batch is already delivered. Upload into a new batch.")
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.unlink(missing_ok=True)
        staging.touch(exist_ok=False)
        return {"path": str(relative), "offset": 0}

    @app.patch("/api/upload/chunk/{relative_path:path}")
    async def append_upload(relative_path: str, offset: int, request: Request,
                            batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        relative = _safe_relative(relative_path)
        selected_path(batch_id, relative)
        _, staging = web_run.upload_paths(relative)
        if not staging.is_file() or staging.stat().st_size != offset:
            raise HTTPException(409, "Upload offset changed; select the source again")
        received = 0
        with staging.open("ab") as file:
            async for chunk in request.stream():
                file.write(chunk)
                received += len(chunk)
        return {"path": str(relative), "offset": offset + received}

    @app.post("/api/upload/complete/{relative_path:path}")
    def finish_upload(relative_path: str, size: int, batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        relative = _safe_relative(relative_path)
        expected_size = selected_path(batch_id, relative)
        if expected_size is not None and size != expected_size:
            raise HTTPException(400, "Upload size differs from the selected file")
        destination, staging = web_run.upload_paths(relative)
        if not staging.is_file() or staging.stat().st_size != size:
            raise HTTPException(400, "Uploaded byte count does not match the browser file")
        digest = sha256()
        with staging.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest_hex = digest.hexdigest()
        if destination.exists():
            existing = sha256()
            with destination.open("rb") as file:
                while chunk := file.read(8 * 1024 * 1024):
                    existing.update(chunk)
            if destination.stat().st_size != size or existing.hexdigest() != digest_hex:
                staging.unlink(missing_ok=True)
                raise HTTPException(409, "A different source file already uses this path in the batch")
            staging.unlink()
            return {"path": str(relative), "bytes": size, "sha256": digest_hex,
                    "reused": True}
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        invalidate_from(web_run.run_dir, "inventory")
        return {"path": str(relative), "bytes": size, "sha256": digest_hex,
                "reused": False}

    @app.post("/api/process", status_code=202)
    def process(batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        states = workflow_state(web_run.run_dir)
        next_stage = next((item for item in states
                           if item["stage"] != "delivery" and not item["approved"]), None)
        if next_stage is None:
            raise HTTPException(409, "All processing and review checkpoints are approved")
        if next_stage["status"] == "complete":
            raise HTTPException(409, f"Approve {next_stage['name']} before continuing")
        if next_stage["stage"] == "review":
            raise HTTPException(409, "Complete episode review before continuing")
        web_run.start(next_stage["stage"])
        return {"status": "running", "stage": next_stage["stage"]}

    @app.post("/api/stages/{stage}/approve")
    def approve(stage: str, approval: Approval, batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        if web_run.job_status == "running":
            raise HTTPException(409, "Wait for the current stage to finish")
        try:
            approve_stage(web_run.run_dir, stage, approval.approved_by)
            if stage == "qc":
                run_stage(str(web_run.source), web_run.run_dir, "review")
        except RunError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "approved", "stage": stage}

    @app.put("/api/episodes/{capture_id}/decision")
    def decide(capture_id: str, decision: Decision, batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        web_run.decide(capture_id, decision)
        return {"status": "saved"}

    @app.get("/api/episodes/{capture_id}/preview")
    def preview(capture_id: str, batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        with sqlite3.connect(web_run.run_dir / "run.sqlite") as database:
            database.row_factory = sqlite3.Row
            capture = database.execute(
                """SELECT parent_path, capture_key FROM capture_snapshot
                   WHERE capture_id=? AND is_canonical=1""", (capture_id,),
            ).fetchone()
            if capture is None:
                raise HTTPException(404, "Episode is not current")
            videos = _vendor_visualizations(
                database, capture["parent_path"], capture["capture_key"])
        if not videos:
            raise HTTPException(404, "No vendor review video is present")
        return FileResponse(
            web_run.run_dir / f"cache/blobs/{videos[0]['source_sha256']}",
            media_type="video/mp4",
        )

    @app.get("/api/episodes/{capture_id}/video/{stream}")
    def episode_video(capture_id: str, stream: str, batch_id: str = initial_batch_id):
        return FileResponse(batch(batch_id).video_path(capture_id, stream), media_type="video/mp4")

    @app.get("/api/episodes/{capture_id}/evidence")
    def episode_evidence(capture_id: str, batch_id: str = initial_batch_id):
        return batch(batch_id).evidence(capture_id)

    @app.post("/api/delivery")
    def delivery(batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        try:
            review = next(item for item in workflow_state(web_run.run_dir)
                          if item["stage"] == "review")
            if not review["approved"]:
                raise RunError("Approve Human review before building delivery")
            policy = telemetry_policy(web_run.run_dir)
            if policy["requires_choice"]:
                raise RunError("Choose how to handle partial telemetry before building delivery")
            if calibration and calibration_selected[batch_id]:
                _bind_calibration(web_run, calibration)
            stage = "archive" if web_run.output.is_dir() else "delivery"
            web_run.start(stage)
        except RunError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "running", "stage": stage}

    @app.post("/api/delivery/telemetry")
    def choose_telemetry(request: TelemetryChoice, batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        if web_run.output.exists():
            raise HTTPException(409, "This delivery is already built and cannot be changed")
        review = next(item for item in workflow_state(web_run.run_dir)
                      if item["stage"] == "review")
        if not review["approved"]:
            raise HTTPException(409, "Approve Human review before choosing delivery contents")
        try:
            return telemetry_policy(web_run.run_dir, request.choice)
        except RunError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/delivery/archive")
    def archive_delivery(batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        try:
            web_run.build_archive()
        except RunError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "ready", "bytes": web_run.archive.stat().st_size}

    @app.get("/api/delivery/download")
    def download_delivery(batch_id: str = initial_batch_id):
        web_run = batch(batch_id)
        if web_run.archive_record() is None:
            raise HTTPException(404, "The delivery ZIP is not ready")
        return FileResponse(web_run.archive, media_type="application/zip",
                            filename=f"actuate-{batch_id}.zip")

    return app
