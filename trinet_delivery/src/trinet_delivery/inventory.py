import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROLES = {"imu": "imu", "json": "json", "mp4": "video", "tel": "telemetry", "tmf": "tmf", "vts": "vts"}
REQUIRED = {
    "single_video": {("video", "single"), ("vts", "single"), ("imu", None)},
    "stereo_pair": {("video", "left"), ("vts", "left"),
                    ("video", "right"), ("vts", "right"), ("imu", None)},
}

@dataclass(frozen=True)
class FileFact:
    source_item_id: str
    relative_path: str
    parent_path: str
    role: str
    capture_key: str | None
    camera_stream_id: str | None
    size_bytes: int
    modified_time_ns: int

@dataclass(frozen=True)
class CaptureFact:
    parent_path: str
    capture_key: str
    capture_layout: str | None
    grouping_status: str
    members: tuple[FileFact, ...]

@dataclass(frozen=True)
class SourceInventory:
    source_identity: str
    files: tuple[FileFact, ...]
    captures: tuple[CaptureFact, ...]

@dataclass(frozen=True)
class PreservedFile:
    source_item_id: str
    source_sha256: str
    change: str
Preservation = tuple[tuple[PreservedFile, ...], tuple[tuple[str, str, str, bool], ...], int]

def group_captures(files: tuple[FileFact, ...]) -> tuple[CaptureFact, ...]:
    keys = {
        (file.parent_path, file.capture_key)
        for file in files
        if file.role in ("video", "vts", "imu")
        and file.capture_key is not None
    }
    captures = []
    for parent_path, capture_key in sorted(keys):
        assert capture_key is not None
        members = tuple(
            file
            for file in files
            if (file.parent_path, file.capture_key) == (parent_path, capture_key)
        )
        streams = {
            file.camera_stream_id
            for file in members
            if file.role in ("video", "vts")
        }
        has_single = "single" in streams
        has_stereo = bool(streams & {"left", "right"})
        layout = None
        if has_single != has_stereo:
            layout = "single_video" if has_single else "stereo_pair"

        core = [
            (file.role, file.camera_stream_id)
            for file in members
            if file.role in ("video", "vts", "imu")
        ]
        duplicate = len(core) != len(set(core))
        if has_single and has_stereo or duplicate:
            status = "ambiguous"
        elif layout is not None and REQUIRED[layout] <= set(core):
            status = "complete"
        else:
            status = "incomplete"
        captures.append(CaptureFact(parent_path, capture_key, layout, status, members))
    return tuple(captures)


def inventory_local(source: str, run_dir: Path) -> SourceInventory:
    root = Path(source).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Local source is not a directory: {root}")

    resolved_run_dir = run_dir.expanduser().resolve(strict=False)
    if resolved_run_dir == root or resolved_run_dir.is_relative_to(root):
        raise ValueError("RUN_DIR must not be equal to or inside SOURCE")

    files = []
    for parent, directory_names, file_names in os.walk(root, topdown=True):
        parent = Path(parent)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name.casefold() != "system volume information"
        )
        for name in directory_names:
            path = parent / name
            if path.is_symlink():
                raise ValueError(
                    f"Local source contains a symlink: {path.relative_to(root)}"
                )
        for name in sorted(file_names):
            path = parent / name
            if path.is_symlink():
                raise ValueError(
                    f"Local source contains a symlink: {path.relative_to(root)}"
                )
            if not path.is_file():
                continue

            relative = path.relative_to(root)
            extension = path.suffix.lower()
            role = ROLES.get(extension.removeprefix("."), "other")
            stem = path.stem
            capture_key = stem
            camera_stream_id = None
            if role == "video" and (
                stem == "visualization" or stem.endswith("_stereo_depth_imu")
            ):
                role = "auxiliary"
                capture_key = None
            elif role in ("video", "vts"):
                camera_stream_id = "single"
                if stem.endswith("_L"):
                    capture_key = stem[:-2]
                    camera_stream_id = "left"
                elif stem.endswith("_R"):
                    capture_key = stem[:-2]
                    camera_stream_id = "right"

            metadata = path.stat()
            relative_path = relative.as_posix()
            files.append(
                FileFact(
                    relative_path,
                    relative_path,
                    relative.parent.as_posix(),
                    role,
                    capture_key,
                    camera_stream_id,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                )
            )

    facts = tuple(sorted(files, key=lambda file: file.relative_path))
    return SourceInventory(root.as_uri(), facts, group_captures(facts))

def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _verify_blob(path: Path, size_bytes: int, expected_hash: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Immutable cache blob is missing: {expected_hash}")
    if path.stat().st_size != size_bytes or _hash_file(path) != expected_hash:
        raise ValueError(f"Immutable cache blob changed: {expected_hash}")

def _check_source(path: Path, fact: FileFact) -> None:
    metadata = path.stat()
    if metadata.st_size != fact.size_bytes or metadata.st_mtime_ns != fact.modified_time_ns:
        raise ValueError(f"Source member changed: {fact.relative_path}")

def _copy_bytes(source, destination, digest) -> None:
    while chunk := source.read(1024 * 1024):
        destination.write(chunk)
        digest.update(chunk)

def _copy_to_blob(source: Path, run_dir: Path, fact: FileFact) -> str:
    _check_source(source, fact)
    staging = run_dir / "cache/staging" / sha256(fact.source_item_id.encode()).hexdigest()
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists() or staging.is_symlink():
        staging.unlink()

    digest = sha256()
    with source.open("rb") as source_file, staging.open("xb") as destination:
        _copy_bytes(source_file, destination, digest)
    source_hash = digest.hexdigest()
    _check_source(source, fact)
    if staging.stat().st_size != fact.size_bytes or _hash_file(staging) != source_hash:
        raise ValueError(f"Copied bytes changed: {fact.relative_path}")

    blob = run_dir / f"cache/blobs/{source_hash}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    if blob.exists() or blob.is_symlink():
        _verify_blob(blob, fact.size_bytes, source_hash)
        staging.unlink()
    else:
        staging.replace(blob)
    return source_hash

def preserve_inventory(source_root: Path, run_dir: Path, inventory: SourceInventory,
                       previous: dict[str, tuple[bool, str | None]]) -> Preservation:
    if source_root.is_relative_to((run_dir / "cache").resolve()):
        raise ValueError("Cannot preserve a source inside RUN_DIR/cache")

    preserved = []
    hashes = {}
    for fact in inventory.files:
        source = source_root / fact.relative_path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Source member changed: {fact.relative_path}")
        prior = previous.get(fact.source_item_id)
        if prior and prior[1]:
            _check_source(source, fact)
            source_hash = _hash_file(source)
            _check_source(source, fact)
            if source_hash == prior[1]:
                _verify_blob(run_dir / f"cache/blobs/{source_hash}", fact.size_bytes, source_hash)
            else:
                copied_hash = _copy_to_blob(source, run_dir, fact)
                if copied_hash != source_hash:
                    raise ValueError(f"Source member changed: {fact.relative_path}")
                source_hash = copied_hash
        else:
            source_hash = _copy_to_blob(source, run_dir, fact)
        change = "unchanged" if prior and prior[0] and prior[1] == source_hash else (
            "changed" if prior and prior[0] and prior[1] else "new"
        )
        hashes[fact.source_item_id] = source_hash
        preserved.append(PreservedFile(fact.source_item_id, source_hash, change))

    snapshots = []
    canonical_ids = set()
    for capture in inventory.captures:
        identity = sorted(
            (member.role, member.camera_stream_id or "", hashes[member.source_item_id],
             member.size_bytes)
            for member in capture.members
        )
        capture_id = sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        snapshots.append((capture.parent_path, capture.capture_key, capture_id,
                          capture_id not in canonical_ids))
        canonical_ids.add(capture_id)
    current_ids = {file.source_item_id for file in inventory.files}
    removed = sum(present and item_id not in current_ids for item_id, (present, _) in previous.items())
    return tuple(preserved), tuple(snapshots), removed
