import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from trinet_delivery.inventory import (
    Preservation,
    SourceInventory,
    inventory_local,
    preserve_inventory,
)


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

        if version not in (1, 2, 3, 4, 5, 6):
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
        if database.execute("PRAGMA user_version").fetchone()[0] != 6:
            return {}
        rows = database.execute(
            "SELECT source_item_id, present, source_sha256 FROM source_file"
        ).fetchall()
    return {item_id: (bool(present), source_hash) for item_id, present, source_hash in rows}

def store_inventory(database_path: Path, inventory: SourceInventory) -> None:
    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        version = database.execute("PRAGMA user_version").fetchone()[0]
        assert version in (1, 2, 3, 4, 5, 6)
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
        assert database.execute("PRAGMA user_version").fetchone()[0] == 6
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

    files, captures, removed = preservation
    counts = {status: 0 for status in ("new", "changed", "unchanged")}
    for file in files:
        counts[file.change] += 1
    unique = sum(capture[3] for capture in captures)
    return RunResult(
        run_id, len(inventory.files), len(inventory.captures),
        counts["new"], counts["changed"], counts["unchanged"],
        removed, unique, len(captures) - unique,
    )
