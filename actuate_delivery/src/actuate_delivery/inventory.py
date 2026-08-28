import json
import mimetypes
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROLES = {
    "imu": "imu",
    "json": "json",
    "mp4": "video",
    "tel": "telemetry",
    "tmf": "tmf",
    "vts": "vts",
}
REQUIRED = {
    "single_video": {("video", "single"), ("vts", "single"), ("imu", None)},
    "stereo_pair": {
        ("video", "left"),
        ("vts", "left"),
        ("video", "right"),
        ("vts", "right"),
        ("imu", None),
    },
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
    source_type: str
    parent_source_item_id: str
    mime_type: str
    source_checksum_algorithm: str | None
    source_checksum: str | None
    can_download: bool


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


def classify_name(name: str) -> tuple[str, str | None, str | None]:
    path = Path(name)
    role = ROLES.get(path.suffix.lower().removeprefix("."), "other")
    stem = path.stem
    capture_key = stem
    camera_stream_id = None
    if role == "video" and (stem == "visualization" or stem.endswith("_stereo_depth_imu")):
        return "auxiliary", None, None
    if role in ("video", "vts"):
        camera_stream_id = "single"
        if stem.endswith("_L"):
            capture_key = stem[:-2]
            camera_stream_id = "left"
        elif stem.endswith("_R"):
            capture_key = stem[:-2]
            camera_stream_id = "right"
    return role, capture_key, camera_stream_id


def group_captures(files: tuple[FileFact, ...]) -> tuple[CaptureFact, ...]:
    keys = {
        (file.parent_path, file.capture_key)
        for file in files
        if file.role in ("video", "vts", "imu") and file.capture_key is not None
    }
    captures = []
    for parent_path, capture_key in sorted(keys):
        assert capture_key is not None
        members = tuple(
            file
            for file in files
            if (file.parent_path, file.capture_key) == (parent_path, capture_key)
        )
        streams = {file.camera_stream_id for file in members if file.role in ("video", "vts")}
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


def select_inventory(
    inventory: SourceInventory, source_item_ids: tuple[str, ...] | None
) -> SourceInventory:
    if source_item_ids is None:
        return inventory
    files = {file.source_item_id: file for file in inventory.files}
    unknown = sorted(set(source_item_ids) - files.keys())
    if unknown:
        raise ValueError(f"Unknown source item ID: {unknown[0]}")

    selected = set(source_item_ids)
    groups = {
        (files[item_id].parent_path, files[item_id].capture_key)
        for item_id in selected
        if files[item_id].capture_key is not None
    }
    captures_by_parent = {}
    for capture in inventory.captures:
        captures_by_parent.setdefault(capture.parent_path, []).append(capture.capture_key)
    for item_id in tuple(selected):
        file = files[item_id]
        if file.role != "auxiliary":
            continue
        name = Path(file.relative_path).name
        keys = captures_by_parent.get(file.parent_path, [])
        exact = [key for key in keys if name == f"{key}_stereo_depth_imu.mp4"]
        if exact:
            groups.add((file.parent_path, exact[0]))
        elif name == "visualization.mp4" and len(keys) == 1:
            groups.add((file.parent_path, keys[0]))
    for file in inventory.files:
        if (file.parent_path, file.capture_key) in groups:
            selected.add(file.source_item_id)
        if file.role == "auxiliary":
            name = Path(file.relative_path).name
            keys = [key for parent, key in groups if parent == file.parent_path]
            if any(name == f"{key}_stereo_depth_imu.mp4" for key in keys) or (
                name == "visualization.mp4"
                and len(captures_by_parent.get(file.parent_path, [])) == 1
                and keys
            ):
                selected.add(file.source_item_id)
    selected_files = tuple(file for file in inventory.files if file.source_item_id in selected)
    return SourceInventory(
        inventory.source_identity, selected_files, group_captures(selected_files)
    )


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
            name for name in directory_names if name.casefold() != "system volume information"
        )
        for name in directory_names:
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"Local source contains a symlink: {path.relative_to(root)}")
        for name in sorted(file_names):
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"Local source contains a symlink: {path.relative_to(root)}")
            if not path.is_file():
                continue

            relative = path.relative_to(root)
            role, capture_key, camera_stream_id = classify_name(name)

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
                    "local",
                    relative.parent.as_posix(),
                    mimetypes.guess_type(name)[0] or "application/octet-stream",
                    None,
                    None,
                    True,
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


def _publish_staging(staging: Path, run_dir: Path, fact: FileFact, source_hash: str) -> None:
    if staging.stat().st_size != fact.size_bytes or _hash_file(staging) != source_hash:
        raise ValueError(f"Copied bytes changed: {fact.relative_path}")
    blob = run_dir / f"cache/blobs/{source_hash}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    if blob.exists() or blob.is_symlink():
        _verify_blob(blob, fact.size_bytes, source_hash)
        staging.unlink()
    else:
        staging.replace(blob)


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
    _publish_staging(staging, run_dir, fact, source_hash)
    return source_hash


def _preservation_result(
    inventory: SourceInventory,
    hashes: dict[str, str],
    previous: dict[str, tuple[bool, str | None]],
    present_item_ids: set[str],
) -> Preservation:
    preserved = []
    for fact in inventory.files:
        prior = previous.get(fact.source_item_id)
        source_hash = hashes[fact.source_item_id]
        change = (
            "unchanged"
            if prior and prior[0] and prior[1] == source_hash
            else ("changed" if prior and prior[0] and prior[1] else "new")
        )
        preserved.append(PreservedFile(fact.source_item_id, source_hash, change))

    snapshots = []
    canonical_ids = set()
    for capture in inventory.captures:
        identity = sorted(
            (
                member.role,
                member.camera_stream_id or "",
                hashes[member.source_item_id],
                member.size_bytes,
            )
            for member in capture.members
        )
        capture_id = sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        snapshots.append(
            (capture.parent_path, capture.capture_key, capture_id, capture_id not in canonical_ids)
        )
        canonical_ids.add(capture_id)
    removed = sum(
        present and item_id not in present_item_ids for item_id, (present, _) in previous.items()
    )
    return tuple(preserved), tuple(snapshots), removed


def preserve_inventory(
    source_root: Path,
    run_dir: Path,
    inventory: SourceInventory,
    previous: dict[str, tuple[bool, str | None]],
    present_item_ids: set[str],
    progress,
) -> Preservation:
    if source_root.is_relative_to((run_dir / "cache").resolve()):
        raise ValueError("Cannot preserve a source inside RUN_DIR/cache")

    hashes = {}
    for fact in inventory.files:
        progress(fact, "running", None)
        source = source_root / fact.relative_path
        try:
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Source member changed: {fact.relative_path}")
            prior = previous.get(fact.source_item_id)
            if prior and prior[1]:
                _check_source(source, fact)
                source_hash = _hash_file(source)
                _check_source(source, fact)
                if source_hash == prior[1]:
                    _verify_blob(
                        run_dir / f"cache/blobs/{source_hash}", fact.size_bytes, source_hash
                    )
                else:
                    copied_hash = _copy_to_blob(source, run_dir, fact)
                    if copied_hash != source_hash:
                        raise ValueError(f"Source member changed: {fact.relative_path}")
                    source_hash = copied_hash
            else:
                source_hash = _copy_to_blob(source, run_dir, fact)
        except (OSError, ValueError) as error:
            progress(fact, "failed", str(error))
            raise
        hashes[fact.source_item_id] = source_hash
        progress(fact, "complete", None)
    return _preservation_result(inventory, hashes, previous, present_item_ids)
