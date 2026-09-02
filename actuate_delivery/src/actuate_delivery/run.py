import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from uuid import uuid4

from actuate_delivery.inventory import (
    Preservation,
    SourceInventory,
    inventory_local,
    preserve_inventory,
    select_inventory,
)
from actuate_delivery.package import (
    PackageError,
    _vendor_visualizations,
    build_delivery,
    project_supplier,
)
from actuate_delivery.panoculon_trinet import (
    ImuError,
    SidecarError,
    convert_imu,
    convert_tel,
    decode_imu,
    decode_vts,
)
from actuate_delivery.qc import (
    QcError,
    build_qc,
    controlled_limitations,
    supplier_issues,
    timing_stream_facts,
)
from actuate_delivery.timing import TimingError, TimingStream, build_timing
from actuate_delivery.video import VideoError, verify_video


class RunError(ValueError):
    pass


class RunInputError(RunError):
    pass


class SourceMismatchError(RunInputError):
    pass


@dataclass(frozen=True)
class RunResult:
    run_id: str
    files: int
    captures: int
    new: int
    changed: int
    unchanged: int
    removed: int
    unique_captures: int
    duplicate_captures: int
    imu_decoded: int
    imu_reused: int
    imu_failed: int
    vts_decoded: int
    vts_reused: int
    vts_failed: int
    tel_decoded: int
    tel_reused: int
    tel_failed: int
    video_verified: int
    video_reused: int
    video_failed: int
    timing_created: int
    timing_reused: int
    timing_unavailable: int
    timing_failed: int
    qc_created: int
    qc_reused: int
    qc_failed: int


@dataclass(frozen=True)
class DeliveryRunResult:
    status: str
    review_path: Path
    included: int
    excluded: int
    pending: int
    output: Path | None


REVIEW_FIELDS = (
    "episode_id",
    "capture_id",
    "source_relative_directory",
    "source_group",
    "capture_layout",
    "grouping_status",
    "qc_sha256",
    "pass_count",
    "fail_count",
    "unknown_count",
    "not_applicable_count",
    "blocking_checks",
    "material_checks",
    "decision",
    "limitations_json",
    "decided_at",
)
REVIEW_FACT_FIELDS = REVIEW_FIELDS[:13]
LEGACY_REVIEW_FIELDS = (*REVIEW_FIELDS[:-1], "decided_by", REVIEW_FIELDS[-1])


def _ensure_progress(database):
    database.execute(
        """CREATE TABLE IF NOT EXISTS processing_item (
               stage TEXT NOT NULL, item_key TEXT NOT NULL, label TEXT NOT NULL,
               ordinal INTEGER NOT NULL, status TEXT NOT NULL,
               outcome TEXT, started_at TEXT, completed_at TEXT,
               elapsed_seconds REAL, error TEXT,
               PRIMARY KEY (stage, item_key),
               CHECK (status IN ('waiting', 'running', 'complete', 'failed', 'interrupted'))
           )"""
    )


def prepare_progress(database_path: Path, stage: str, items):
    with sqlite3.connect(database_path) as database:
        _ensure_progress(database)
        database.execute("DELETE FROM processing_item WHERE stage=?", (stage,))
        database.executemany(
            "INSERT INTO processing_item (stage, item_key, label, ordinal, status) "
            "VALUES (?, ?, ?, ?, 'waiting')",
            [(stage, key, label, index) for index, (key, label) in enumerate(items, 1)],
        )


def _start_progress(database, stage: str, item_key: str):
    _ensure_progress(database)
    database.execute(
        """UPDATE processing_item SET status='running', outcome=NULL, started_at=?,
                  completed_at=NULL, elapsed_seconds=NULL, error=NULL
           WHERE stage=? AND item_key=?""",
        (datetime.now(UTC).isoformat(), stage, item_key),
    )
    database.commit()
    return monotonic()


def _finish_progress(
    database, stage: str, item_key: str, started: float, outcome: str, error: str | None = None
):
    status = "failed" if error else "complete"
    database.execute(
        """UPDATE processing_item SET status=?, outcome=?, completed_at=?,
                  elapsed_seconds=?, error=? WHERE stage=? AND item_key=?""",
        (
            status,
            outcome,
            datetime.now(UTC).isoformat(),
            round(monotonic() - started, 3),
            error,
            stage,
            item_key,
        ),
    )
    database.commit()


def processing_progress(database_path: Path, stage: str):
    if not database_path.is_file():
        return {"completed": 0, "total": 0, "current": None, "items": []}
    with sqlite3.connect(database_path) as database:
        database.row_factory = sqlite3.Row
        tables = {
            row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "processing_item" not in tables:
            return {"completed": 0, "total": 0, "current": None, "items": []}
        items = [
            dict(row)
            for row in database.execute(
                "SELECT * FROM processing_item WHERE stage=? ORDER BY ordinal", (stage,)
            )
        ]
    terminal = {"complete", "failed", "interrupted"}
    current = next((item for item in items if item["status"] == "running"), None)
    return {
        "completed": sum(item["status"] in terminal for item in items),
        "total": len(items),
        "current": current,
        "items": items,
    }


def interrupt_progress(database_path: Path, stage: str, error: str):
    if not database_path.is_file():
        return
    with sqlite3.connect(database_path) as database:
        _ensure_progress(database)
        database.execute(
            """UPDATE processing_item SET status='interrupted', completed_at=?, error=?
               WHERE stage=? AND status='running'""",
            (datetime.now(UTC).isoformat(), error, stage),
        )


def stage_progress_items(database_path: Path, stage: str):
    with sqlite3.connect(database_path) as database:
        if stage in ("sensors", "video"):
            roles = ("imu", "vts", "telemetry") if stage == "sensors" else ("video",)
            placeholders = ",".join("?" for _ in roles)
            rows = database.execute(
                f"""SELECT role, capture_id, COALESCE(camera_stream_id, ''), relative_path
                    FROM capture_snapshot JOIN capture_member USING (parent_path, capture_key)
                    JOIN source_file USING (source_item_id)
                    WHERE is_canonical=1 AND role IN ({placeholders})
                    ORDER BY role, capture_id, camera_stream_id, relative_path""",
                roles,
            ).fetchall()
            grouped = {}
            for role, capture_id, stream, path in rows:
                key = f"{role}:{capture_id}:{stream}"
                grouped.setdefault(key, []).append(path)
            return [(key, " + ".join(paths)) for key, paths in grouped.items()]
        rows = database.execute(
            """SELECT capture_id, parent_path, capture_key FROM capture_snapshot
               WHERE is_canonical=1 ORDER BY parent_path, capture_key"""
        ).fetchall()
    return [(capture_id, f"{parent}/{key}" if parent else key) for capture_id, parent, key in rows]


def open_run(source: str, run_dir: Path) -> str:
    assert source
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "run.sqlite"
    initialize = not database_path.exists() or database_path.stat().st_size == 0

    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        if initialize:
            assert version == 0
            database.execute(
                """
                CREATE TABLE run (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    run_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_resumed_at TEXT NOT NULL
                )
                """
            )
            database.execute("PRAGMA user_version = 1")
            run_id = str(uuid4())
            now = datetime.now(UTC).isoformat()
            database.execute("INSERT INTO run VALUES (1, ?, ?, ?, ?)", (run_id, source, now, now))
            return run_id

        if version not in range(1, 16):
            raise RuntimeError(f"Unsupported run database version: {version}")
        stored = database.execute("SELECT run_id, source FROM run WHERE singleton = 1").fetchone()
        assert stored is not None
        run_id, stored_source = stored
        if stored_source != source:
            raise SourceMismatchError(f"Run directory belongs to {stored_source!r}, not {source!r}")
        database.execute(
            "UPDATE run SET last_resumed_at = ? WHERE singleton = 1",
            (datetime.now(UTC).isoformat(),),
        )
        return run_id


def load_previous(database_path: Path) -> dict[str, tuple[bool, str | None]]:
    with sqlite3.connect(database_path) as database:
        if database.execute("PRAGMA user_version").fetchone()[0] not in range(6, 16):
            return {}
        rows = database.execute(
            "SELECT source_item_id, present, source_sha256 FROM source_file"
        ).fetchall()
    return {item_id: (bool(present), source_hash) for item_id, present, source_hash in rows}


def load_previous_source_metadata(
    database_path: Path,
) -> dict[str, tuple[int, str | None, str | None]]:
    with sqlite3.connect(database_path) as database:
        columns = {row[1] for row in database.execute("PRAGMA table_info(source_file)")}
        required = {"size_bytes", "source_checksum_algorithm", "source_checksum"}
        if not required <= columns:
            return {}
        rows = database.execute(
            "SELECT source_item_id, size_bytes, source_checksum_algorithm, source_checksum "
            "FROM source_file WHERE present=1"
        ).fetchall()
    return {item_id: (size, algorithm, checksum) for item_id, size, algorithm, checksum in rows}


def store_inventory(
    database_path: Path, inventory: SourceInventory, selected: SourceInventory
) -> None:
    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(1, 16)
        if version < 6:
            database.executescript(
                """
                BEGIN IMMEDIATE;
                DROP TABLE IF EXISTS capture_snapshot_member;
                DROP TABLE IF EXISTS capture_snapshot;
                DROP TABLE IF EXISTS capture_member;
                DROP TABLE IF EXISTS capture_candidate;
                DROP TABLE IF EXISTS source_file;
                DROP TABLE IF EXISTS recording_group;
                CREATE TABLE source_file (
                    source_item_id TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    parent_path TEXT NOT NULL,
                    role TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    modified_time_ns INTEGER NOT NULL,
                    present INTEGER NOT NULL CHECK (present IN (0, 1)),
                    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),
                    source_sha256 TEXT CHECK (length(source_sha256) = 64 OR source_sha256 IS NULL),
                    cache_relative_path TEXT,
                    preservation_status TEXT CHECK (preservation_status IN
                        ('new', 'changed', 'unchanged', 'removed') OR preservation_status IS NULL),
                    source_type TEXT NOT NULL,
                    parent_source_item_id TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    source_checksum_algorithm TEXT,
                    source_checksum TEXT,
                    can_download INTEGER NOT NULL CHECK (can_download IN (0, 1))
                );
                CREATE TABLE capture_candidate (
                    parent_path TEXT NOT NULL,
                    capture_key TEXT NOT NULL,
                    capture_layout TEXT,
                    grouping_status TEXT NOT NULL,
                    file_count INTEGER NOT NULL CHECK (file_count > 0),
                    PRIMARY KEY (parent_path, capture_key),
                    CHECK (capture_layout IN ('single_video', 'stereo_pair') OR capture_layout IS NULL),
                    CHECK (grouping_status IN ('complete', 'incomplete', 'ambiguous'))
                );
                CREATE TABLE capture_member (
                    source_item_id TEXT PRIMARY KEY REFERENCES source_file(source_item_id),
                    parent_path TEXT NOT NULL,
                    capture_key TEXT NOT NULL,
                    camera_stream_id TEXT,
                    FOREIGN KEY (parent_path, capture_key) REFERENCES
                        capture_candidate(parent_path, capture_key),
                    CHECK (camera_stream_id IN ('single', 'left', 'right') OR camera_stream_id IS NULL)
                );
                CREATE TABLE capture_snapshot (
                    parent_path TEXT NOT NULL,
                    capture_key TEXT NOT NULL,
                    capture_id TEXT NOT NULL CHECK (length(capture_id) = 64),
                    is_canonical INTEGER NOT NULL CHECK (is_canonical IN (0, 1)),
                    PRIMARY KEY (parent_path, capture_key),
                    FOREIGN KEY (parent_path, capture_key) REFERENCES
                        capture_candidate(parent_path, capture_key)
                );
                PRAGMA user_version = 6;
                """
            )
        else:
            database.execute("BEGIN IMMEDIATE")

        columns = {row[1] for row in database.execute("PRAGMA table_info(source_file)")}
        additions = {
            "source_type": "TEXT NOT NULL DEFAULT 'local'",
            "parent_source_item_id": "TEXT NOT NULL DEFAULT '.'",
            "mime_type": "TEXT NOT NULL DEFAULT 'application/octet-stream'",
            "source_checksum_algorithm": "TEXT",
            "source_checksum": "TEXT",
            "can_download": "INTEGER NOT NULL DEFAULT 1 CHECK (can_download IN (0, 1))",
        }
        for name, declaration in additions.items():
            if name not in columns:
                database.execute(f"ALTER TABLE source_file ADD COLUMN {name} {declaration}")
        if version == 12:
            database.execute("PRAGMA user_version = 13")

        database.execute(
            "UPDATE source_file SET present = 0, selected = 0, preservation_status = 'removed'"
        )
        selected_ids = {file.source_item_id for file in selected.files}
        database.execute("DELETE FROM capture_snapshot")
        database.execute("DELETE FROM capture_member")
        database.execute("DELETE FROM capture_candidate")
        database.executemany(
            """
            INSERT INTO source_file (
                source_item_id, relative_path, parent_path, role, size_bytes,
                modified_time_ns, present, selected, source_sha256, cache_relative_path,
                preservation_status, source_type, parent_source_item_id, mime_type,
                source_checksum_algorithm, source_checksum, can_download
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_item_id) DO UPDATE SET
                relative_path=excluded.relative_path, parent_path=excluded.parent_path,
                role=excluded.role, size_bytes=excluded.size_bytes,
                modified_time_ns=excluded.modified_time_ns, present=1,
                selected=excluded.selected, source_type=excluded.source_type,
                parent_source_item_id=excluded.parent_source_item_id,
                mime_type=excluded.mime_type,
                source_checksum_algorithm=excluded.source_checksum_algorithm,
                source_checksum=excluded.source_checksum, can_download=excluded.can_download,
                source_sha256=CASE WHEN excluded.selected=0
                    AND source_file.size_bytes=excluded.size_bytes
                    AND source_file.modified_time_ns=excluded.modified_time_ns
                    THEN source_file.source_sha256 END,
                cache_relative_path=CASE WHEN excluded.selected=0
                    AND source_file.size_bytes=excluded.size_bytes
                    AND source_file.modified_time_ns=excluded.modified_time_ns
                    THEN source_file.cache_relative_path END,
                preservation_status=NULL
            """,
            [
                (
                    file.source_item_id,
                    file.relative_path,
                    file.parent_path,
                    file.role,
                    file.size_bytes,
                    file.modified_time_ns,
                    int(file.source_item_id in selected_ids),
                    file.source_type,
                    file.parent_source_item_id,
                    file.mime_type,
                    file.source_checksum_algorithm,
                    file.source_checksum,
                    int(file.can_download),
                )
                for file in inventory.files
            ],
        )
        database.executemany(
            "INSERT INTO capture_candidate VALUES (?, ?, ?, ?, ?)",
            [
                (
                    capture.parent_path,
                    capture.capture_key,
                    capture.capture_layout,
                    capture.grouping_status,
                    len(capture.members),
                )
                for capture in selected.captures
            ],
        )
        database.executemany(
            "INSERT INTO capture_member VALUES (?, ?, ?, ?)",
            [
                (
                    file.source_item_id,
                    capture.parent_path,
                    capture.capture_key,
                    file.camera_stream_id,
                )
                for capture in selected.captures
                for file in capture.members
            ],
        )


def store_preservation(database_path: Path, preservation: Preservation) -> None:
    files, captures, _ = preservation
    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        assert database.execute("PRAGMA user_version").fetchone()[0] in range(6, 16)
        database.execute("BEGIN IMMEDIATE")
        cursor = database.executemany(
            """UPDATE source_file
               SET source_sha256 = ?, cache_relative_path = ?, preservation_status = ?
               WHERE source_item_id = ? AND present = 1 AND selected = 1""",
            [
                (
                    file.source_sha256,
                    f"cache/blobs/{file.source_sha256}",
                    file.change,
                    file.source_item_id,
                )
                for file in files
            ],
        )
        assert cursor.rowcount == len(files)
        database.executemany("INSERT INTO capture_snapshot VALUES (?, ?, ?, ?)", captures)


def process_imus(database_path: Path, run_dir: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(6, 16)
        if version == 6:
            database.execute(
                """CREATE TABLE imu_artifact (
                    capture_id TEXT PRIMARY KEY CHECK (length(capture_id) = 64),
                    source_sha256 TEXT, status TEXT NOT NULL,
                    parquet_relative_path TEXT, parquet_sha256 TEXT,
                    sample_count INTEGER, error TEXT,
                    CHECK (status IN ('decoded', 'failed'))
                )"""
            )
            database.execute("PRAGMA user_version = 7")
        rows = database.execute(
            """SELECT capture_id, source_sha256, cache_relative_path
               FROM capture_snapshot
               JOIN capture_member USING (parent_path, capture_key)
               JOIN source_file USING (source_item_id)
               WHERE is_canonical = 1 AND role = 'imu'
               ORDER BY capture_id, source_item_id"""
        ).fetchall()
        members = {}
        for capture_id, source_hash, cache_path in rows:
            members.setdefault(capture_id, []).append((source_hash, cache_path))

        decoded = reused = failed = 0
        for capture_id, imus in members.items():
            item_key = f"imu:{capture_id}:"
            started = _start_progress(database, "sensors", item_key)
            source_hash = imus[0][0] if len(imus) == 1 else None
            relative = f"work/{capture_id}/imu.parquet"
            try:
                if len(imus) != 1:
                    raise ImuError(f"Capture has {len(imus)} IMU members; expected exactly one")
                assert source_hash is not None
                prior = database.execute(
                    "SELECT source_sha256, parquet_sha256, status FROM imu_artifact WHERE capture_id=?",
                    (capture_id,),
                ).fetchone()
                output = run_dir / relative
                if prior and prior[0] == source_hash and prior[2] == "decoded":
                    if not output.is_file() or sha256(output.read_bytes()).hexdigest() != prior[1]:
                        raise ImuError("Published IMU Parquet changed after verification")
                    reused += 1
                    _finish_progress(database, "sensors", item_key, started, "reused")
                    continue
                artifact = convert_imu(run_dir / imus[0][1], output, source_hash)
                result = (
                    source_hash,
                    "decoded",
                    relative,
                    artifact.parquet_sha256,
                    artifact.sample_count,
                    None,
                    capture_id,
                )
                decoded += 1
            except (OSError, ImuError) as error:
                result = (source_hash, "failed", None, None, None, str(error), capture_id)
                failed += 1
            database.execute(
                """INSERT INTO imu_artifact VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(capture_id) DO UPDATE SET
                   source_sha256=excluded.source_sha256, status=excluded.status,
                   parquet_relative_path=excluded.parquet_relative_path,
                   parquet_sha256=excluded.parquet_sha256,
                   sample_count=excluded.sample_count, error=excluded.error""",
                (result[-1], *result[:-1]),
            )
            _finish_progress(
                database,
                "sensors",
                item_key,
                started,
                "decoded" if result[1] == "decoded" else "failed",
                result[-2],
            )
    return decoded, reused, failed


def process_sidecars(database_path: Path, run_dir: Path) -> tuple[int, int, int, int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(7, 16)
        if version == 7:
            database.executescript(
                """BEGIN IMMEDIATE;
                CREATE TABLE vts_artifact (
                    capture_id TEXT NOT NULL, camera_stream_id TEXT NOT NULL,
                    source_sha256 TEXT, status TEXT NOT NULL,
                    native_version INTEGER, frame_rate_milli INTEGER, frame_count INTEGER,
                    first_timestamp_ns INTEGER, last_timestamp_ns INTEGER,
                    master_clock_offset_ns INTEGER, clock_skew_ppb INTEGER,
                    sync_quality_us INTEGER, sync_flags INTEGER, error TEXT,
                    PRIMARY KEY (capture_id, camera_stream_id), CHECK (status IN ('decoded', 'failed'))
                );
                CREATE TABLE tel_artifact (
                    capture_id TEXT PRIMARY KEY, source_sha256 TEXT, status TEXT NOT NULL,
                    parquet_relative_path TEXT, parquet_sha256 TEXT, record_count INTEGER, error TEXT,
                    CHECK (status IN ('decoded', 'failed'))
                );
                PRAGMA user_version = 8;
                COMMIT;"""
            )
        rows = database.execute(
            """SELECT capture_id, camera_stream_id, source_sha256, cache_relative_path
               FROM capture_snapshot JOIN capture_member USING (parent_path, capture_key)
               JOIN source_file USING (source_item_id)
               WHERE is_canonical = 1 AND role = 'vts'
               ORDER BY capture_id, camera_stream_id, source_item_id"""
        ).fetchall()
        grouped = {}
        for capture_id, stream, source_hash, cache_path in rows:
            grouped.setdefault((capture_id, stream), []).append((source_hash, cache_path))
        vts_decoded = vts_reused = vts_failed = 0
        for (capture_id, stream), members in grouped.items():
            item_key = f"vts:{capture_id}:{stream}"
            started = _start_progress(database, "sensors", item_key)
            source_hash = members[0][0] if len(members) == 1 else None
            try:
                if len(members) != 1:
                    raise SidecarError(
                        f"Camera stream has {len(members)} VTS members; expected one"
                    )
                prior = database.execute(
                    "SELECT source_sha256, status FROM vts_artifact WHERE capture_id=? AND camera_stream_id=?",
                    (capture_id, stream),
                ).fetchone()
                if prior == (source_hash, "decoded"):
                    vts_reused += 1
                    _finish_progress(database, "sensors", item_key, started, "reused")
                    continue
                data = decode_vts(run_dir / members[0][1], source_hash)
                field = "timestamp_ns" if data.version == 1 else "sof_timestamp_ns"
                usable = data.entries[field][data.entries[field] != 0]
                result = (
                    source_hash,
                    "decoded",
                    data.version,
                    data.frame_rate_milli,
                    len(data.entries),
                    int(usable[0]) if len(usable) else None,
                    int(usable[-1]) if len(usable) else None,
                    data.master_clock_offset_ns,
                    data.clock_skew_ppb,
                    data.sync_quality_us,
                    data.sync_flags,
                    None,
                )
                vts_decoded += 1
            except (OSError, SidecarError) as error:
                result = (
                    source_hash,
                    "failed",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    str(error),
                )
                vts_failed += 1
            database.execute(
                """INSERT INTO vts_artifact VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(capture_id, camera_stream_id) DO UPDATE SET
                   source_sha256=excluded.source_sha256, status=excluded.status,
                   native_version=excluded.native_version, frame_rate_milli=excluded.frame_rate_milli,
                   frame_count=excluded.frame_count, first_timestamp_ns=excluded.first_timestamp_ns,
                   last_timestamp_ns=excluded.last_timestamp_ns,
                   master_clock_offset_ns=excluded.master_clock_offset_ns,
                   clock_skew_ppb=excluded.clock_skew_ppb,
                   sync_quality_us=excluded.sync_quality_us, sync_flags=excluded.sync_flags,
                   error=excluded.error""",
                (capture_id, stream, *result),
            )
            _finish_progress(
                database,
                "sensors",
                item_key,
                started,
                "decoded" if result[1] == "decoded" else "failed",
                result[-1],
            )

        rows = database.execute(
            """SELECT capture_id, source_sha256, cache_relative_path
               FROM capture_snapshot JOIN capture_member USING (parent_path, capture_key)
               JOIN source_file USING (source_item_id)
               WHERE is_canonical = 1 AND role = 'telemetry'
               ORDER BY capture_id, source_item_id"""
        ).fetchall()
        grouped = {}
        for capture_id, source_hash, cache_path in rows:
            grouped.setdefault(capture_id, []).append((source_hash, cache_path))
        tel_decoded = tel_reused = tel_failed = 0
        for capture_id, members in grouped.items():
            item_key = f"telemetry:{capture_id}:"
            started = _start_progress(database, "sensors", item_key)
            source_hash = members[0][0] if len(members) == 1 else None
            relative = f"work/{capture_id}/telemetry.parquet"
            try:
                if len(members) != 1:
                    raise SidecarError(f"Capture has {len(members)} TEL members; expected one")
                prior = database.execute(
                    "SELECT source_sha256, parquet_sha256, status FROM tel_artifact WHERE capture_id=?",
                    (capture_id,),
                ).fetchone()
                output = run_dir / relative
                if prior and prior[0] == source_hash and prior[2] == "decoded":
                    if not output.is_file() or sha256(output.read_bytes()).hexdigest() != prior[1]:
                        raise SidecarError("Published TEL Parquet changed after verification")
                    tel_reused += 1
                    _finish_progress(database, "sensors", item_key, started, "reused")
                    continue
                artifact = convert_tel(run_dir / members[0][1], output, source_hash)
                result = (
                    source_hash,
                    "decoded",
                    relative,
                    artifact.parquet_sha256,
                    artifact.record_count,
                    None,
                )
                tel_decoded += 1
            except (OSError, SidecarError) as error:
                result = (source_hash, "failed", None, None, None, str(error))
                tel_failed += 1
            database.execute(
                """INSERT INTO tel_artifact VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(capture_id) DO UPDATE SET source_sha256=excluded.source_sha256,
                   status=excluded.status, parquet_relative_path=excluded.parquet_relative_path,
                   parquet_sha256=excluded.parquet_sha256,
                   record_count=excluded.record_count, error=excluded.error""",
                (capture_id, *result),
            )
            _finish_progress(
                database,
                "sensors",
                item_key,
                started,
                "decoded" if result[1] == "decoded" else "failed",
                result[-1],
            )
    return vts_decoded, vts_reused, vts_failed, tel_decoded, tel_reused, tel_failed


def process_videos(database_path: Path, run_dir: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(8, 16)
        if version == 8:
            database.executescript(
                """BEGIN IMMEDIATE;
                CREATE TABLE video_artifact (
                    capture_id TEXT NOT NULL, camera_stream_id TEXT NOT NULL,
                    source_sha256 TEXT, status TEXT NOT NULL,
                    frame_index_relative_path TEXT, frame_index_sha256 TEXT,
                    frame_count INTEGER, codec TEXT, width INTEGER, height INTEGER,
                    average_frame_rate TEXT, duration_ns INTEGER, audio_stream_count INTEGER,
                    facts_json TEXT, error TEXT,
                    PRIMARY KEY (capture_id, camera_stream_id),
                    CHECK (status IN ('verified', 'failed'))
                );
                PRAGMA user_version = 9;
                COMMIT;"""
            )
        rows = database.execute(
            """SELECT capture_id, camera_stream_id, source_sha256, cache_relative_path
               FROM capture_snapshot
               JOIN capture_member USING (parent_path, capture_key)
               JOIN source_file USING (source_item_id)
               WHERE is_canonical = 1 AND role = 'video'
               ORDER BY capture_id, camera_stream_id, source_item_id"""
        ).fetchall()
        grouped = {}
        for capture_id, stream, source_hash, cache_path in rows:
            grouped.setdefault((capture_id, stream), []).append((source_hash, cache_path))
        verified = reused = failed = 0
        for (capture_id, stream), members in grouped.items():
            item_key = f"video:{capture_id}:{stream}"
            started = _start_progress(database, "video", item_key)
            source_hash = members[0][0] if len(members) == 1 else None
            relative = f"work/{capture_id}/video_{stream}_frames.parquet"
            try:
                if len(members) != 1:
                    raise VideoError(
                        f"Camera stream has {len(members)} video members; expected one"
                    )
                prior = database.execute(
                    "SELECT source_sha256, frame_index_sha256, status FROM video_artifact "
                    "WHERE capture_id=? AND camera_stream_id=?",
                    (capture_id, stream),
                ).fetchone()
                output = run_dir / relative
                if prior and prior[0] == source_hash and prior[2] == "verified":
                    if not output.is_file() or sha256(output.read_bytes()).hexdigest() != prior[1]:
                        raise VideoError("Published video frame index changed after verification")
                    reused += 1
                    _finish_progress(database, "video", item_key, started, "reused")
                    continue
                artifact = verify_video(run_dir / members[0][1], output, source_hash)
                result = (
                    source_hash,
                    "verified",
                    relative,
                    artifact.parquet_sha256,
                    artifact.frame_count,
                    artifact.codec,
                    artifact.width,
                    artifact.height,
                    artifact.average_frame_rate,
                    artifact.duration_ns,
                    artifact.audio_stream_count,
                    artifact.facts_json,
                    None,
                )
                verified += 1
            except (OSError, VideoError) as error:
                result = (
                    source_hash,
                    "failed",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    str(error),
                )
                failed += 1
            database.execute(
                """INSERT INTO video_artifact VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(capture_id, camera_stream_id) DO UPDATE SET
                   source_sha256=excluded.source_sha256, status=excluded.status,
                   frame_index_relative_path=excluded.frame_index_relative_path,
                   frame_index_sha256=excluded.frame_index_sha256,
                   frame_count=excluded.frame_count, codec=excluded.codec,
                   width=excluded.width, height=excluded.height,
                   average_frame_rate=excluded.average_frame_rate,
                   duration_ns=excluded.duration_ns, audio_stream_count=excluded.audio_stream_count,
                   facts_json=excluded.facts_json, error=excluded.error""",
                (capture_id, stream, *result),
            )
            _finish_progress(
                database,
                "video",
                item_key,
                started,
                "verified" if result[1] == "verified" else "failed",
                result[-1],
            )
    return verified, reused, failed


def process_timing(database_path: Path, run_dir: Path) -> tuple[int, int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(9, 16)
        if version == 9:
            database.executescript(
                """BEGIN IMMEDIATE;
                CREATE TABLE timing_artifact (
                    capture_id TEXT PRIMARY KEY, input_signature TEXT,
                    status TEXT NOT NULL, parquet_relative_path TEXT, parquet_sha256 TEXT,
                    row_count INTEGER, matched_rows INTEGER, coverage_rows INTEGER,
                    stereo_pair_count INTEGER, stereo_unmatched_rows INTEGER, reason TEXT,
                    CHECK (status IN ('ready', 'unavailable', 'failed'))
                );
                PRAGMA user_version = 10;
                COMMIT;"""
            )
        captures = database.execute(
            """SELECT capture_id, capture_layout FROM capture_snapshot
               JOIN capture_candidate USING (parent_path, capture_key)
               WHERE is_canonical = 1 ORDER BY capture_id"""
        ).fetchall()
        created = reused = unavailable = failed = 0
        for capture_id, layout in captures:
            started = _start_progress(database, "timing", capture_id)
            expected = {"single_video": ("single",), "stereo_pair": ("left", "right")}.get(layout)
            inputs = database.execute(
                "SELECT parquet_relative_path, parquet_sha256 FROM imu_artifact "
                "WHERE capture_id=? AND status='decoded'",
                (capture_id,),
            ).fetchall()
            streams = []
            reason = None if expected else "capture layout is incomplete or ambiguous"
            if len(inputs) != 1:
                reason = "verified IMU artifact is unavailable"
            if reason is None:
                for stream_id in expected:
                    vts = database.execute(
                        """SELECT source_file.source_sha256, source_file.cache_relative_path,
                                  vts_artifact.status FROM capture_snapshot
                           JOIN capture_member USING (parent_path, capture_key)
                           JOIN source_file USING (source_item_id)
                           LEFT JOIN vts_artifact USING (capture_id, camera_stream_id)
                           WHERE capture_id=? AND is_canonical=1 AND camera_stream_id=? AND role='vts'""",
                        (capture_id, stream_id),
                    ).fetchall()
                    video = database.execute(
                        "SELECT frame_index_relative_path, frame_index_sha256, status "
                        "FROM video_artifact WHERE capture_id=? AND camera_stream_id=?",
                        (capture_id, stream_id),
                    ).fetchall()
                    if len(vts) != 1 or vts[0][2] != "decoded":
                        reason = f"verified VTS artifact is unavailable for {stream_id}"
                        break
                    if len(video) != 1 or video[0][2] != "verified":
                        reason = f"verified video artifact is unavailable for {stream_id}"
                        break
                    streams.append(
                        TimingStream(
                            stream_id,
                            run_dir / vts[0][1],
                            vts[0][0],
                            run_dir / video[0][0],
                            video[0][1],
                        )
                    )
            values = (None, "unavailable", None, None, None, None, None, None, None, reason)
            if reason is not None:
                unavailable += 1
            else:
                parts = ["frame_timing.v1", inputs[0][1]]
                for stream in streams:
                    parts.extend(
                        (stream.camera_stream_id, stream.vts_sha256, stream.video_index_sha256)
                    )
                signature = sha256("|".join(parts).encode()).hexdigest()
                relative = f"work/{capture_id}/frame_timing.parquet"
                output = run_dir / relative
                prior = database.execute(
                    "SELECT input_signature, parquet_sha256, status FROM timing_artifact "
                    "WHERE capture_id=?",
                    (capture_id,),
                ).fetchone()
                try:
                    if prior and prior[0] == signature and prior[2] == "ready":
                        if (
                            not output.is_file()
                            or sha256(output.read_bytes()).hexdigest() != prior[1]
                        ):
                            raise TimingError(
                                "Published frame timing Parquet changed after verification"
                            )
                        reused += 1
                        _finish_progress(database, "timing", capture_id, started, "reused")
                        continue
                    artifact = build_timing(
                        run_dir / inputs[0][0], inputs[0][1], tuple(streams), output
                    )
                    values = (
                        signature,
                        "ready",
                        relative,
                        artifact.parquet_sha256,
                        artifact.row_count,
                        artifact.matched_rows,
                        artifact.coverage_rows,
                        artifact.stereo_pair_count,
                        artifact.stereo_unmatched_rows,
                        None,
                    )
                    created += 1
                except (OSError, TimingError) as error:
                    values = (
                        signature,
                        "failed",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        str(error),
                    )
                    failed += 1
            database.execute(
                """INSERT INTO timing_artifact VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(capture_id) DO UPDATE SET input_signature=excluded.input_signature,
                   status=excluded.status, parquet_relative_path=excluded.parquet_relative_path,
                   parquet_sha256=excluded.parquet_sha256, row_count=excluded.row_count,
                   matched_rows=excluded.matched_rows, coverage_rows=excluded.coverage_rows,
                   stereo_pair_count=excluded.stereo_pair_count,
                   stereo_unmatched_rows=excluded.stereo_unmatched_rows, reason=excluded.reason""",
                (capture_id, *values),
            )
            _finish_progress(
                database,
                "timing",
                capture_id,
                started,
                values[1],
                values[-1] if values[1] == "failed" else None,
            )
    return created, reused, unavailable, failed


def process_qc(database_path: Path, run_dir: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as database:
        database.row_factory = sqlite3.Row
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(10, 16)
        if version == 10:
            database.executescript(
                """BEGIN IMMEDIATE;
                CREATE TABLE qc_artifact (
                    capture_id TEXT PRIMARY KEY, input_signature TEXT NOT NULL,
                    status TEXT NOT NULL, json_relative_path TEXT, json_sha256 TEXT,
                    pass_count INTEGER, fail_count INTEGER, unknown_count INTEGER,
                    not_applicable_count INTEGER, reason TEXT,
                    CHECK (status IN ('ready', 'failed'))
                );
                PRAGMA user_version = 11;
                COMMIT;"""
            )
        captures = database.execute(
            """SELECT capture_id, capture_layout, grouping_status
               FROM capture_snapshot JOIN capture_candidate USING (parent_path, capture_key)
               WHERE is_canonical=1 ORDER BY capture_id"""
        ).fetchall()
        created = reused = failed = 0
        for capture in captures:
            capture_id = capture["capture_id"]
            started = _start_progress(database, "qc", capture_id)
            members = [
                dict(row)
                for row in database.execute(
                    """SELECT relative_path, role, camera_stream_id, size_bytes, source_sha256,
                          cache_relative_path
                   FROM capture_snapshot JOIN capture_member USING (parent_path, capture_key)
                   JOIN source_file USING (source_item_id)
                   WHERE capture_id=? AND is_canonical=1 ORDER BY relative_path""",
                    (capture_id,),
                )
            ]
            verified = sum(
                len(member["source_sha256"] or "") == 64
                and member["cache_relative_path"] == f"cache/blobs/{member['source_sha256']}"
                for member in members
            )
            imu_row = database.execute(
                """SELECT status, source_sha256, parquet_relative_path, parquet_sha256,
                          sample_count, error FROM imu_artifact WHERE capture_id=?""",
                (capture_id,),
            ).fetchone()
            tel_row = database.execute(
                """SELECT status, source_sha256, parquet_relative_path, parquet_sha256,
                          record_count, error FROM tel_artifact WHERE capture_id=?""",
                (capture_id,),
            ).fetchone()
            timing_row = database.execute(
                """SELECT status, parquet_relative_path, parquet_sha256, row_count,
                          matched_rows, coverage_rows, stereo_pair_count,
                          stereo_unmatched_rows, reason FROM timing_artifact WHERE capture_id=?""",
                (capture_id,),
            ).fetchone()
            expected = {"single_video": ("single",), "stereo_pair": ("left", "right")}.get(
                capture["capture_layout"]
            )
            stream_ids = expected or tuple(
                row[0]
                for row in database.execute(
                    """SELECT DISTINCT camera_stream_id FROM capture_snapshot
                   JOIN capture_member USING (parent_path, capture_key)
                   WHERE capture_id=? AND is_canonical=1 AND camera_stream_id IS NOT NULL
                   ORDER BY camera_stream_id""",
                    (capture_id,),
                )
            )
            streams = []
            for stream_id in stream_ids:
                vts_row = database.execute(
                    """SELECT status, source_sha256, native_version, frame_rate_milli,
                              frame_count, first_timestamp_ns, last_timestamp_ns,
                              master_clock_offset_ns, clock_skew_ppb, sync_quality_us,
                              sync_flags, error FROM vts_artifact
                       WHERE capture_id=? AND camera_stream_id=?""",
                    (capture_id, stream_id),
                ).fetchone()
                video_row = database.execute(
                    """SELECT status, source_sha256, frame_index_relative_path,
                              frame_index_sha256, frame_count, codec, width, height,
                              average_frame_rate, duration_ns, audio_stream_count,
                              facts_json, error FROM video_artifact
                       WHERE capture_id=? AND camera_stream_id=?""",
                    (capture_id, stream_id),
                ).fetchone()
                video = dict(video_row) if video_row else {"status": "missing"}
                if video.get("facts_json") is not None:
                    video["probe"] = json.loads(video.pop("facts_json"))
                streams.append(
                    {
                        "camera_stream_id": stream_id,
                        "vts": dict(vts_row) if vts_row else {"status": "missing"},
                        "video": video,
                    }
                )
            facts = {
                "capture_id": capture_id,
                "capture_layout": capture["capture_layout"],
                "grouping_status": capture["grouping_status"],
                "source": {
                    "file_count": len(members),
                    "bytes": sum(member["size_bytes"] for member in members),
                    "verified_members": verified,
                    "all_hashes_verified_in_current_run": verified == len(members),
                    "members": members,
                },
                "imu": dict(imu_row) if imu_row else {"status": "missing"},
                "streams": streams,
                "telemetry": dict(tel_row) if tel_row else {"status": "absent"},
                "timing": dict(timing_row)
                if timing_row
                else {"status": "unavailable", "reason": "not processed"},
            }
            encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
            signature = sha256(f"qc_builder.v3|{encoded}".encode()).hexdigest()
            relative = f"work/{capture_id}/qc_internal.json"
            output = run_dir / relative
            prior = database.execute(
                """SELECT input_signature, json_sha256, status, pass_count, fail_count,
                          unknown_count, not_applicable_count FROM qc_artifact
                   WHERE capture_id=?""",
                (capture_id,),
            ).fetchone()
            try:
                if facts["imu"]["status"] == "decoded":
                    imu_members = [member for member in members if member["role"] == "imu"]
                    if len(imu_members) != 1:
                        raise QcError(
                            "Decoded IMU artifact does not have exactly one source member"
                        )
                    member = imu_members[0]
                    native = decode_imu(
                        run_dir / member["cache_relative_path"], member["source_sha256"]
                    )
                    first = int(native.samples["timestamp_ns"][0])
                    last = int(native.samples["timestamp_ns"][-1])
                    if len(native.samples) != facts["imu"]["sample_count"]:
                        raise QcError("Decoded IMU sample count changed before QC")
                    facts["imu"]["native"] = {
                        "version": native.version,
                        "declared_sample_rate_hz": native.sample_rate_hz,
                        "accel_full_scale_code": native.accel_fs,
                        "gyro_full_scale_code": native.gyro_fs,
                        "header_start_time_ns": native.start_time_ns,
                        "video_start_time_ns": native.video_start_ns,
                        "flags": native.flags,
                        "device_id_hex": native.device_id.hex(),
                        "ios_clock_offset_ns": native.ios_host_offset_ns,
                        "reserved_header_hex": native.reserved_header.hex(),
                        "first_sample_timestamp_ns": first,
                        "last_sample_timestamp_ns": last,
                        "measured_sample_rate_hz": None
                        if len(native.samples) == 1
                        else round((len(native.samples) - 1) * 1_000_000_000 / (last - first), 6),
                    }
                if facts["timing"]["status"] == "ready":
                    facts["timing"]["streams"] = timing_stream_facts(
                        run_dir / facts["timing"]["parquet_relative_path"],
                        facts["timing"]["parquet_sha256"],
                    )
                if prior and prior["input_signature"] == signature and prior["status"] == "ready":
                    if (
                        not output.is_file()
                        or sha256(output.read_bytes()).hexdigest() != prior["json_sha256"]
                    ):
                        raise QcError("Published internal QC JSON changed after verification")
                    reused += 1
                    _finish_progress(database, "qc", capture_id, started, "reused")
                    continue
                artifact = build_qc(facts, output)
                values = (
                    signature,
                    "ready",
                    relative,
                    artifact.json_sha256,
                    artifact.pass_count,
                    artifact.fail_count,
                    artifact.unknown_count,
                    artifact.not_applicable_count,
                    None,
                )
                created += 1
            except (OSError, TypeError, ValueError) as error:
                values = (signature, "failed", None, None, None, None, None, None, str(error))
                failed += 1
            database.execute(
                """INSERT INTO qc_artifact VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(capture_id) DO UPDATE SET input_signature=excluded.input_signature,
                   status=excluded.status, json_relative_path=excluded.json_relative_path,
                   json_sha256=excluded.json_sha256, pass_count=excluded.pass_count,
                   fail_count=excluded.fail_count, unknown_count=excluded.unknown_count,
                   not_applicable_count=excluded.not_applicable_count, reason=excluded.reason""",
                (capture_id, *values),
            )
            _finish_progress(
                database,
                "qc",
                capture_id,
                started,
                values[1],
                values[-1] if values[1] == "failed" else None,
            )
    return created, reused, failed


def _write_review(review_path: Path, rows: list[dict]) -> None:
    staging = review_path.with_name(f".{review_path.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        with staging.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with staging.open(newline="") as file:
            if list(csv.DictReader(file)) != rows:
                raise RunError("Review sheet changed during write")
        staging.replace(review_path)
    finally:
        staging.unlink(missing_ok=True)


def _delivery_review(database_path: Path, run_dir: Path):
    review_path = run_dir / "review.csv"
    if review_path.is_symlink():
        raise RunError(f"Review sheet cannot be a symlink: {review_path}")
    with sqlite3.connect(database_path) as database:
        database.row_factory = sqlite3.Row
        version = database.execute("PRAGMA user_version").fetchone()[0]
        if version == 11:
            database.executescript(
                """BEGIN IMMEDIATE;
                CREATE TABLE delivery_decision (
                    capture_id TEXT PRIMARY KEY CHECK (length(capture_id) = 64),
                    qc_sha256 TEXT NOT NULL CHECK (length(qc_sha256) = 64),
                    status TEXT NOT NULL CHECK (status IN ('include', 'exclude')),
                    limitations_json TEXT NOT NULL,
                    decided_at TEXT NOT NULL
                );
                CREATE TABLE delivery_episode (
                    capture_id TEXT PRIMARY KEY CHECK (length(capture_id) = 64),
                    episode_number INTEGER NOT NULL UNIQUE
                        CHECK (episode_number BETWEEN 1 AND 999999)
                );
                PRAGMA user_version = 15;
                COMMIT;"""
            )
        elif version in (12, 13, 14):
            episode_migration = (
                ""
                if version == 14
                else """
                CREATE TABLE delivery_episode (
                    capture_id TEXT PRIMARY KEY CHECK (length(capture_id) = 64),
                    episode_number INTEGER NOT NULL UNIQUE
                        CHECK (episode_number BETWEEN 1 AND 999999)
                );"""
            )
            database.executescript(
                f"""BEGIN IMMEDIATE;
                ALTER TABLE delivery_decision RENAME TO delivery_decision_with_reviewer;
                CREATE TABLE delivery_decision (
                    capture_id TEXT PRIMARY KEY CHECK (length(capture_id) = 64),
                    qc_sha256 TEXT NOT NULL CHECK (length(qc_sha256) = 64),
                    status TEXT NOT NULL CHECK (status IN ('include', 'exclude')),
                    limitations_json TEXT NOT NULL,
                    decided_at TEXT NOT NULL
                );
                INSERT INTO delivery_decision
                    SELECT capture_id, qc_sha256, status, limitations_json, decided_at
                    FROM delivery_decision_with_reviewer;
                DROP TABLE delivery_decision_with_reviewer;
                {episode_migration}
                PRAGMA user_version = 15;
                COMMIT;"""
            )
        elif version != 15:
            raise RunError(
                f"Delivery review requires a schema-11 through schema-15 run ledger, not {version}"
            )

        current = {}
        records = database.execute(
            """SELECT capture_id, parent_path, capture_key, capture_layout, grouping_status,
                      json_relative_path, json_sha256, pass_count, fail_count,
                      unknown_count, not_applicable_count
               FROM capture_snapshot JOIN capture_candidate USING (parent_path, capture_key)
               JOIN qc_artifact USING (capture_id)
               WHERE is_canonical=1 AND status='ready'
               ORDER BY CASE WHEN grouping_status='complete' THEN 0 ELSE 1 END,
                        parent_path, capture_key, capture_id"""
        ).fetchall()
        current_count = database.execute(
            "SELECT COUNT(*) FROM capture_snapshot WHERE is_canonical=1"
        ).fetchone()[0]
        if len(records) != current_count:
            raise RunError("Every current capture must have ready QC before delivery review")
        episode_numbers = dict(
            database.execute("SELECT capture_id, episode_number FROM delivery_episode").fetchall()
        )
        next_episode_number = max(episode_numbers.values(), default=0)
        for record in records:
            if record["capture_id"] in episode_numbers:
                continue
            next_episode_number += 1
            database.execute(
                "INSERT INTO delivery_episode VALUES (?, ?)",
                (record["capture_id"], next_episode_number),
            )
            episode_numbers[record["capture_id"]] = next_episode_number
        for record in records:
            relative = Path(record["json_relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RunError(f"Unsafe QC artifact path: {record['capture_id']}")
            encoded = (run_dir / relative).read_bytes()
            if sha256(encoded).hexdigest() != record["json_sha256"]:
                raise RunError(f"Current QC artifact changed: {record['capture_id']}")
            internal = json.loads(encoded)
            blocked, material = supplier_issues(internal)
            row = {
                "episode_id": f"episode_{episode_numbers[record['capture_id']]:06d}",
                "capture_id": record["capture_id"],
                "source_relative_directory": record["parent_path"],
                "source_group": record["capture_key"],
                "capture_layout": record["capture_layout"] or "",
                "grouping_status": record["grouping_status"],
                "qc_sha256": record["json_sha256"],
                "pass_count": str(record["pass_count"]),
                "fail_count": str(record["fail_count"]),
                "unknown_count": str(record["unknown_count"]),
                "not_applicable_count": str(record["not_applicable_count"]),
                "blocking_checks": "|".join(blocked),
                "material_checks": "|".join(material),
            }
            current[record["capture_id"]] = {
                "row": row,
                "internal_qc": internal,
                "parent_path": record["parent_path"],
                "capture_key": record["capture_key"],
            }

        prior = {
            row["capture_id"]: dict(row)
            for row in database.execute("SELECT * FROM delivery_decision")
        }
        if review_path.exists():
            with review_path.open(newline="") as file:
                reader = csv.DictReader(file)
                fields = tuple(reader.fieldnames or ())
                accepted_fields = (
                    REVIEW_FIELDS,
                    REVIEW_FIELDS[1:],
                    LEGACY_REVIEW_FIELDS,
                    LEGACY_REVIEW_FIELDS[1:],
                )
                if fields not in accepted_fields:
                    raise RunError("Review sheet columns changed")
                rows = list(reader)
            if fields in (REVIEW_FIELDS[1:], LEGACY_REVIEW_FIELDS[1:]):
                rows = [
                    {
                        "episode_id": current.get(row["capture_id"], {"row": {"episode_id": ""}})[
                            "row"
                        ]["episode_id"],
                        **row,
                    }
                    for row in rows
                ]
            legacy_review = "decided_by" in fields
            if legacy_review:
                for row in rows:
                    row.pop("decided_by")
            seen = set()
            for row in rows:
                capture_id = row["capture_id"]
                if capture_id in seen:
                    raise RunError(f"Review sheet has duplicate capture: {capture_id}")
                seen.add(capture_id)
                if capture_id not in current:
                    raise RunError(
                        f"Review sheet contains a capture that is no longer current: {capture_id}"
                    )
                existing = prior.get(capture_id)
                if (
                    row["qc_sha256"] != current[capture_id]["row"]["qc_sha256"]
                    and existing
                    and row["qc_sha256"] == existing["qc_sha256"]
                ):
                    continue
                if any(
                    row[field] != current[capture_id]["row"][field] for field in REVIEW_FACT_FIELDS
                ):
                    raise RunError(f"Review sheet facts changed or became stale: {capture_id}")
                expected_time = (
                    existing["decided_at"]
                    if (existing and existing["qc_sha256"] == row["qc_sha256"])
                    else ""
                )
                if row["decided_at"] != expected_time:
                    raise RunError(f"Review decision time was edited: {capture_id}")
                status = row["decision"].strip()
                try:
                    limitations = json.loads(row["limitations_json"] or "[]")
                except json.JSONDecodeError as error:
                    raise RunError(f"Invalid limitations JSON for {capture_id}: {error}") from error
                if not isinstance(limitations, list) or any(
                    not isinstance(item, str) or not item.strip() for item in limitations
                ):
                    raise RunError(
                        f"Limitations must be a JSON list of non-empty strings: {capture_id}"
                    )
                limitations = [item.strip() for item in limitations]
                if status not in ("", "include", "exclude"):
                    raise RunError(f"Invalid review decision for {capture_id}: {status!r}")
                if not status:
                    if limitations:
                        raise RunError(f"Pending decision has human fields: {capture_id}")
                    database.execute(
                        "DELETE FROM delivery_decision WHERE capture_id=?", (capture_id,)
                    )
                    continue
                if status == "include" and current[capture_id]["row"]["blocking_checks"]:
                    raise RunError(f"Blocking capture cannot be included: {capture_id}")
                expected_limitations = (
                    controlled_limitations(current[capture_id]["internal_qc"])
                    if status == "include"
                    else []
                )
                if legacy_review:
                    limitations = expected_limitations
                elif limitations != expected_limitations:
                    raise RunError(f"Review limitations are pipeline-controlled: {capture_id}")
                values = (status, json.dumps(limitations, separators=(",", ":")))
                unchanged = (
                    existing
                    and existing["qc_sha256"] == row["qc_sha256"]
                    and status == existing["status"]
                )
                decided_at = existing["decided_at"] if unchanged else datetime.now(UTC).isoformat()
                database.execute(
                    """INSERT INTO delivery_decision VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(capture_id) DO UPDATE SET qc_sha256=excluded.qc_sha256,
                       status=excluded.status, limitations_json=excluded.limitations_json,
                       decided_at=excluded.decided_at""",
                    (capture_id, row["qc_sha256"], *values, decided_at),
                )
        for capture_id, existing in prior.items():
            entry = current.get(capture_id)
            if not entry or existing["qc_sha256"] != entry["row"]["qc_sha256"]:
                continue
            limitations = (
                controlled_limitations(entry["internal_qc"])
                if existing["status"] == "include"
                else []
            )
            database.execute(
                "UPDATE delivery_decision SET limitations_json=? WHERE capture_id=?",
                (json.dumps(limitations, separators=(",", ":")), capture_id),
            )
        decisions = {
            row["capture_id"]: dict(row)
            for row in database.execute("SELECT * FROM delivery_decision")
        }
        output_rows = []
        for capture_id, entry in current.items():
            decision = decisions.get(capture_id)
            if decision and decision["qc_sha256"] == entry["row"]["qc_sha256"]:
                entry["decision"] = {
                    "status": decision["status"],
                    "decided_at": decision["decided_at"],
                    "limitations": json.loads(decision["limitations_json"]),
                }
                human = {
                    "decision": decision["status"],
                    "limitations_json": decision["limitations_json"],
                    "decided_at": decision["decided_at"],
                }
            else:
                entry["decision"] = None
                human = {"decision": "", "limitations_json": "[]", "decided_at": ""}
            output_rows.append(entry["row"] | human)
        _write_review(review_path, output_rows)
        database.commit()
    return review_path, tuple(current.values())


def _telemetry_policy(database_path: Path, included, choice=None):
    capture_ids = sorted(entry["row"]["capture_id"] for entry in included)
    with sqlite3.connect(database_path) as database:
        database.execute(
            """CREATE TABLE IF NOT EXISTS telemetry_policy (
                   singleton INTEGER PRIMARY KEY CHECK (singleton=1),
                   input_signature TEXT NOT NULL CHECK (length(input_signature)=64),
                   choice TEXT NOT NULL CHECK (choice IN ('include_available', 'exclude_all')),
                   decided_at TEXT NOT NULL
               )"""
        )
        artifacts = []
        for capture_id in capture_ids:
            row = database.execute(
                """SELECT status, source_sha256, parquet_sha256 FROM tel_artifact
                   WHERE capture_id=?""",
                (capture_id,),
            ).fetchone()
            artifacts.append((capture_id, *(row or ("absent", None, None))))
        signature = sha256(json.dumps(artifacts, separators=(",", ":")).encode()).hexdigest()
        available = sum(row[1] == "decoded" for row in artifacts)
        coverage = (
            "all" if available == len(capture_ids) else "none" if not available else "partial"
        )
        if coverage != "partial":
            database.execute("DELETE FROM telemetry_policy")
            return {
                "coverage": coverage,
                "included_episodes": len(capture_ids),
                "episodes_with_telemetry": available,
                "requires_choice": False,
                "choice": "include_available" if coverage == "all" else "exclude_all",
            }
        if choice is not None:
            if choice not in ("include_available", "exclude_all"):
                raise RunError(f"Invalid telemetry policy: {choice}")
            database.execute(
                """INSERT INTO telemetry_policy VALUES (1, ?, ?, ?)
                   ON CONFLICT(singleton) DO UPDATE SET
                       input_signature=excluded.input_signature, choice=excluded.choice,
                       decided_at=excluded.decided_at""",
                (signature, choice, datetime.now(UTC).isoformat()),
            )
        stored = database.execute(
            "SELECT input_signature, choice FROM telemetry_policy WHERE singleton=1"
        ).fetchone()
        effective = stored[1] if stored and stored[0] == signature else None
        return {
            "coverage": coverage,
            "included_episodes": len(capture_ids),
            "episodes_with_telemetry": available,
            "requires_choice": effective is None,
            "choice": effective,
        }


def telemetry_policy(run_dir: Path, choice=None):
    _, entries = _delivery_review(run_dir / "run.sqlite", run_dir)
    included = [
        entry
        for entry in entries
        if entry["row"]["grouping_status"] == "complete"
        and entry["decision"]
        and entry["decision"]["status"] == "include"
    ]
    return _telemetry_policy(run_dir / "run.sqlite", included, choice)


def complete_local_delivery(source: str, run_dir: Path, output: Path) -> DeliveryRunResult:
    source_path = Path(source).expanduser().resolve()
    output = output.expanduser().resolve()
    if output == source_path or source_path in output.parents:
        raise RunInputError("Delivery output must be outside the source directory")
    if output.exists() or output.is_symlink():
        raise RunInputError(f"Delivery output already exists: {output}")
    try:
        review_path, entries = _delivery_review(run_dir / "run.sqlite", run_dir)
        eligible = [entry for entry in entries if entry["row"]["grouping_status"] == "complete"]
        pending = sum(entry["decision"] is None for entry in eligible)
        included = [
            entry
            for entry in eligible
            if entry["decision"] and entry["decision"]["status"] == "include"
        ]
        excluded = len(entries) - pending - len(included)
        if pending:
            return DeliveryRunResult(
                "review_required", review_path, len(included), excluded, pending, None
            )
        if not included:
            return DeliveryRunResult("no_captures_included", review_path, 0, excluded, 0, None)
        policy = _telemetry_policy(run_dir / "run.sqlite", included)
        if policy["requires_choice"]:
            return DeliveryRunResult(
                "telemetry_choice_required", review_path, len(included), excluded, 0, None
            )
        with sqlite3.connect(run_dir / "run.sqlite") as database:
            database.row_factory = sqlite3.Row
            episodes = tuple(
                {
                    "episode_id": entry["row"]["episode_id"],
                    "internal_qc": entry["internal_qc"],
                    "source_relative_directory": entry["parent_path"],
                    "source_group": entry["capture_key"],
                    "vendor_visualizations": _vendor_visualizations(
                        database, entry["parent_path"], entry["capture_key"]
                    ),
                    "decision": entry["decision"],
                }
                for entry in included
            )
        calibration_path = run_dir / "calibration.json"
        calibration = (
            json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
        )
        signature = sha256(
            json.dumps(
                [
                    {
                        "episode_id": entry["row"]["episode_id"],
                        "capture_id": entry["row"]["capture_id"],
                        "qc_sha256": entry["row"]["qc_sha256"],
                        "decision": entry["decision"],
                    }
                    for entry in included
                ]
                + [
                    {
                        "projection_builder": "customer_facts.v1",
                        "calibration": calibration,
                        "telemetry_policy": policy,
                    }
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        projection = run_dir / "work/deliveries" / signature / "projection"
        if not projection.exists() and not projection.is_symlink():
            project_supplier(episodes, projection, calibration, policy["choice"])
        build_delivery(run_dir, projection, output)
        return DeliveryRunResult("complete", review_path, len(included), excluded, 0, output)
    except RunError:
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        sqlite3.Error,
        csv.Error,
        json.JSONDecodeError,
        QcError,
        PackageError,
    ) as error:
        raise RunError(str(error)) from error


def _finish_run(
    run_id: str, run_dir: Path, inventory: SourceInventory, preservation: Preservation
) -> RunResult:
    database_path = run_dir / "run.sqlite"
    imu_counts = process_imus(database_path, run_dir)
    sidecar_counts = process_sidecars(database_path, run_dir)
    video_counts = process_videos(database_path, run_dir)
    timing_counts = process_timing(database_path, run_dir)
    qc_counts = process_qc(database_path, run_dir)
    files, captures, removed = preservation
    counts = {status: 0 for status in ("new", "changed", "unchanged")}
    for file in files:
        counts[file.change] += 1
    unique = sum(capture[3] for capture in captures)
    return RunResult(
        run_id,
        len(inventory.files),
        len(inventory.captures),
        counts["new"],
        counts["changed"],
        counts["unchanged"],
        removed,
        unique,
        len(captures) - unique,
        *imu_counts,
        *sidecar_counts,
        *video_counts,
        *timing_counts,
        *qc_counts,
    )


def prepare_inventory(source: str, run_dir: Path, source_item_ids: tuple[str, ...] | None = None):
    try:
        inventory = inventory_local(source, run_dir)
        selected = select_inventory(inventory, source_item_ids)
    except (OSError, ValueError) as error:
        raise RunInputError(str(error)) from error

    run_id = open_run(inventory.source_identity, run_dir)
    database_path = run_dir / "run.sqlite"
    previous = load_previous(database_path)
    store_inventory(database_path, inventory, selected)
    prepare_progress(
        database_path,
        "inventory",
        [(file.source_item_id, file.relative_path) for file in selected.files],
    )
    started = {}

    def progress(file, status, error):
        with sqlite3.connect(database_path) as database:
            if status == "running":
                started[file.source_item_id] = _start_progress(
                    database, "inventory", file.source_item_id
                )
                return
            _finish_progress(
                database,
                "inventory",
                file.source_item_id,
                started.pop(file.source_item_id),
                "preserved",
                error,
            )

    try:
        preservation = preserve_inventory(
            Path(source).expanduser().resolve(),
            run_dir.resolve(),
            selected,
            previous,
            {file.source_item_id for file in inventory.files},
            progress,
        )
        store_preservation(database_path, preservation)
    except (OSError, ValueError) as error:
        raise RunError(str(error)) from error
    return run_id, selected, preservation


def prepare_local_run(
    source: str, run_dir: Path, source_item_ids: tuple[str, ...] | None = None
) -> RunResult:
    run_id, selected, preservation = prepare_inventory(source, run_dir, source_item_ids)
    return _finish_run(run_id, run_dir, selected, preservation)
