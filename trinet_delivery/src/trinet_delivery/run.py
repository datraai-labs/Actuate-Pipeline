import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from trinet_delivery.inventory import (
    Preservation,
    SourceInventory,
    inventory_local,
    preserve_inventory,
)
from trinet_delivery.timing import TimingError, TimingStream, build_timing
from trinet_delivery.trinet import ImuError, SidecarError, convert_imu, convert_tel, decode_vts
from trinet_delivery.video import VideoError, verify_video


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

        if version not in range(1, 11):
            raise RuntimeError(f"Unsupported run database version: {version}")
        stored = database.execute(
            "SELECT run_id, source FROM run WHERE singleton = 1"
        ).fetchone()
        assert stored is not None
        run_id, stored_source = stored
        if stored_source != source:
            raise SourceMismatchError(
                f"Run directory belongs to {stored_source!r}, not {source!r}"
            )
        database.execute(
            "UPDATE run SET last_resumed_at = ? WHERE singleton = 1",
            (datetime.now(UTC).isoformat(),),
        )
        return run_id


def load_previous(database_path: Path) -> dict[str, tuple[bool, str | None]]:
    with sqlite3.connect(database_path) as database:
        if database.execute("PRAGMA user_version").fetchone()[0] not in (6, 7, 8, 9, 10):
            return {}
        rows = database.execute(
            "SELECT source_item_id, present, source_sha256 FROM source_file"
        ).fetchall()
    return {item_id: (bool(present), source_hash) for item_id, present, source_hash in rows}

def store_inventory(database_path: Path, inventory: SourceInventory) -> None:
    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in range(1, 11)
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
                        ('new', 'changed', 'unchanged', 'removed') OR preservation_status IS NULL)
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

        database.execute(
            "UPDATE source_file SET present = 0, selected = 0, preservation_status = 'removed'"
        )
        database.execute("DELETE FROM capture_snapshot")
        database.execute("DELETE FROM capture_member")
        database.execute("DELETE FROM capture_candidate")
        database.executemany(
            """
            INSERT INTO source_file VALUES (?, ?, ?, ?, ?, ?, 1, 1, NULL, NULL, NULL)
            ON CONFLICT(source_item_id) DO UPDATE SET
                relative_path=excluded.relative_path, parent_path=excluded.parent_path,
                role=excluded.role, size_bytes=excluded.size_bytes,
                modified_time_ns=excluded.modified_time_ns, present=1, selected=1,
                source_sha256=NULL, cache_relative_path=NULL, preservation_status=NULL
            """,
            [(file.source_item_id, file.relative_path, file.parent_path, file.role,
              file.size_bytes, file.modified_time_ns) for file in inventory.files],
        )
        database.executemany(
            "INSERT INTO capture_candidate VALUES (?, ?, ?, ?, ?)",
            [(capture.parent_path, capture.capture_key, capture.capture_layout,
              capture.grouping_status, len(capture.members)) for capture in inventory.captures],
        )
        database.executemany(
            "INSERT INTO capture_member VALUES (?, ?, ?, ?)",
            [(file.source_item_id, capture.parent_path, capture.capture_key,
              file.camera_stream_id)
             for capture in inventory.captures for file in capture.members],
        )

def store_preservation(database_path: Path, preservation: Preservation) -> None:
    files, captures, _ = preservation
    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        assert database.execute("PRAGMA user_version").fetchone()[0] in (6, 7, 8, 9, 10)
        database.execute("BEGIN IMMEDIATE")
        cursor = database.executemany(
            """UPDATE source_file
               SET source_sha256 = ?, cache_relative_path = ?, preservation_status = ?
               WHERE source_item_id = ? AND present = 1 AND selected = 1""",
            [(file.source_sha256, f"cache/blobs/{file.source_sha256}", file.change,
              file.source_item_id) for file in files],
        )
        assert cursor.rowcount == len(files)
        database.executemany("INSERT INTO capture_snapshot VALUES (?, ?, ?, ?)", captures)

def process_imus(database_path: Path, run_dir: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in (6, 7, 8, 9, 10)
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
                    continue
                artifact = convert_imu(run_dir / imus[0][1], output, source_hash)
                result = (source_hash, "decoded", relative, artifact.parquet_sha256,
                          artifact.sample_count, None, capture_id)
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
    return decoded, reused, failed


def process_sidecars(database_path: Path, run_dir: Path) -> tuple[int, int, int, int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in (7, 8, 9, 10)
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
            source_hash = members[0][0] if len(members) == 1 else None
            try:
                if len(members) != 1:
                    raise SidecarError(f"Camera stream has {len(members)} VTS members; expected one")
                prior = database.execute(
                    "SELECT source_sha256, status FROM vts_artifact WHERE capture_id=? AND camera_stream_id=?",
                    (capture_id, stream),
                ).fetchone()
                if prior == (source_hash, "decoded"):
                    vts_reused += 1
                    continue
                data = decode_vts(run_dir / members[0][1], source_hash)
                field = "timestamp_ns" if data.version == 1 else "sof_timestamp_ns"
                usable = data.entries[field][data.entries[field] != 0]
                result = (source_hash, "decoded", data.version, data.frame_rate_milli,
                          len(data.entries), int(usable[0]) if len(usable) else None,
                          int(usable[-1]) if len(usable) else None,
                          data.master_clock_offset_ns, data.clock_skew_ppb,
                          data.sync_quality_us, data.sync_flags, None)
                vts_decoded += 1
            except (OSError, SidecarError) as error:
                result = (source_hash, "failed", None, None, None, None, None,
                          None, None, None, None, str(error))
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
                    continue
                artifact = convert_tel(run_dir / members[0][1], output, source_hash)
                result = (source_hash, "decoded", relative, artifact.parquet_sha256,
                          artifact.record_count, None)
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
    return vts_decoded, vts_reused, vts_failed, tel_decoded, tel_reused, tel_failed


def process_videos(database_path: Path, run_dir: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in (8, 9, 10)
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
            source_hash = members[0][0] if len(members) == 1 else None
            relative = f"work/{capture_id}/video_{stream}_frames.parquet"
            try:
                if len(members) != 1:
                    raise VideoError(f"Camera stream has {len(members)} video members; expected one")
                prior = database.execute(
                    "SELECT source_sha256, frame_index_sha256, status FROM video_artifact "
                    "WHERE capture_id=? AND camera_stream_id=?", (capture_id, stream),
                ).fetchone()
                output = run_dir / relative
                if prior and prior[0] == source_hash and prior[2] == "verified":
                    if not output.is_file() or sha256(output.read_bytes()).hexdigest() != prior[1]:
                        raise VideoError("Published video frame index changed after verification")
                    reused += 1
                    continue
                artifact = verify_video(run_dir / members[0][1], output, source_hash)
                result = (source_hash, "verified", relative, artifact.parquet_sha256,
                          artifact.frame_count, artifact.codec, artifact.width, artifact.height,
                          artifact.average_frame_rate, artifact.duration_ns,
                          artifact.audio_stream_count, artifact.facts_json, None)
                verified += 1
            except (OSError, VideoError) as error:
                result = (source_hash, "failed", None, None, None, None, None, None,
                          None, None, None, None, str(error))
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
    return verified, reused, failed


def process_timing(database_path: Path, run_dir: Path) -> tuple[int, int, int, int]:
    with sqlite3.connect(database_path) as database:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in (9, 10)
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
            expected = {"single_video": ("single",), "stereo_pair": ("left", "right")}.get(layout)
            inputs = database.execute(
                "SELECT parquet_relative_path, parquet_sha256 FROM imu_artifact "
                "WHERE capture_id=? AND status='decoded'", (capture_id,),
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
                    streams.append(TimingStream(
                        stream_id, run_dir / vts[0][1], vts[0][0],
                        run_dir / video[0][0], video[0][1],
                    ))
            values = (None, "unavailable", None, None, None, None, None, None, None, reason)
            if reason is not None:
                unavailable += 1
            else:
                parts = ["frame_timing.v1", inputs[0][1]]
                for stream in streams:
                    parts.extend((stream.camera_stream_id, stream.vts_sha256,
                                  stream.video_index_sha256))
                signature = sha256("|".join(parts).encode()).hexdigest()
                relative = f"work/{capture_id}/frame_timing.parquet"
                output = run_dir / relative
                prior = database.execute(
                    "SELECT input_signature, parquet_sha256, status FROM timing_artifact "
                    "WHERE capture_id=?", (capture_id,),
                ).fetchone()
                try:
                    if prior and prior[0] == signature and prior[2] == "ready":
                        if not output.is_file() or sha256(output.read_bytes()).hexdigest() != prior[1]:
                            raise TimingError("Published frame timing Parquet changed after verification")
                        reused += 1
                        continue
                    artifact = build_timing(
                        run_dir / inputs[0][0], inputs[0][1], tuple(streams), output)
                    values = (signature, "ready", relative, artifact.parquet_sha256,
                              artifact.row_count, artifact.matched_rows, artifact.coverage_rows,
                              artifact.stereo_pair_count, artifact.stereo_unmatched_rows, None)
                    created += 1
                except (OSError, TimingError) as error:
                    values = (signature, "failed", None, None, None, None, None, None, None, str(error))
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
    return created, reused, unavailable, failed

def prepare_local_run(source: str, run_dir: Path) -> RunResult:
    try:
        inventory = inventory_local(source, run_dir)
    except (OSError, ValueError) as error:
        raise RunInputError(str(error)) from error

    run_id = open_run(inventory.source_identity, run_dir)
    database_path = run_dir / "run.sqlite"
    previous = load_previous(database_path)
    store_inventory(database_path, inventory)
    try:
        preservation = preserve_inventory(
            Path(source).expanduser().resolve(), run_dir.resolve(), inventory, previous
        )
        store_preservation(database_path, preservation)
    except (OSError, ValueError) as error:
        raise RunError(str(error)) from error

    imu_counts = process_imus(database_path, run_dir)
    sidecar_counts = process_sidecars(database_path, run_dir)
    video_counts = process_videos(database_path, run_dir)
    timing_counts = process_timing(database_path, run_dir)
    files, captures, removed = preservation
    counts = {status: 0 for status in ("new", "changed", "unchanged")}
    for file in files:
        counts[file.change] += 1
    unique = sum(capture[3] for capture in captures)
    return RunResult(
        run_id, len(inventory.files), len(inventory.captures),
        counts["new"], counts["changed"], counts["unchanged"],
        removed, unique, len(captures) - unique,
        *imu_counts, *sidecar_counts, *video_counts, *timing_counts,
    )
