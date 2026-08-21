import sqlite3
from dataclasses import replace
from hashlib import sha256

import pytest
import trinet_delivery.inventory as inventory_module
from trinet_delivery.cli import app
from trinet_delivery.inventory import (
    FileFact,
    PreservedFile,
    SourceInventory,
    group_captures,
)
from trinet_delivery.run import open_run, store_inventory, store_preservation
from typer.testing import CliRunner

runner = CliRunner()


def write_capture(folder, name, layout="single_video", complete=True, prefix=b""):
    folder.mkdir(parents=True, exist_ok=True)
    names = [f"{name}.mp4", f"{name}.vts", f"{name}.imu"]
    if layout == "stereo_pair":
        names = [
            f"{name}_L.mp4", f"{name}_L.vts",
            f"{name}_R.mp4", f"{name}_R.vts", f"{name}.imu",
        ]
    if not complete:
        names.pop(-2 if layout == "stereo_pair" else 0)
    for filename in names:
        (folder / filename).write_bytes(prefix + filename.encode())


def rows(database_path, query):
    with sqlite3.connect(database_path) as database:
        return database.execute(query).fetchall()


def test_batch_preserves_complete_incomplete_stereo_auxiliary_and_other(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source / "mono", "recording1")
    write_capture(source / "stereo", "take0003", "stereo_pair")
    write_capture(source / "incomplete", "take0004", "stereo_pair", complete=False)
    (source / "stereo" / "visualization.mp4").write_bytes(b"aux")
    (source / "notes.txt").write_bytes(b"notes")
    ignored = source / "stereo/System Volume Information/nested"
    write_capture(ignored, "ignored", "stereo_pair")

    result = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert result.exit_code == 0, result.output
    database_path = run_dir / "run.sqlite"
    files = rows(
        database_path,
        "SELECT relative_path, role, present, source_sha256, cache_relative_path "
        "FROM source_file ORDER BY relative_path",
    )
    captures = rows(
        database_path,
        "SELECT parent_path, capture_layout, grouping_status FROM capture_candidate ORDER BY 1",
    )
    snapshots = rows(
        database_path,
        "SELECT parent_path FROM capture_snapshot ORDER BY 1",
    )

    assert len(files) == 14
    assert all(file[2] == 1 and len(file[3]) == 64 for file in files)
    assert all(file[4] == f"cache/blobs/{file[3]}" for file in files)
    assert all("System Volume Information" not in file[0] for file in files)
    assert any(file[:3] == ("notes.txt", "other", 1) for file in files)
    assert any(file[:2] == ("stereo/visualization.mp4", "auxiliary") for file in files)
    assert captures == [
        ("incomplete", "stereo_pair", "incomplete"),
        ("mono", "single_video", "complete"),
        ("stereo", "stereo_pair", "complete"),
    ]
    assert snapshots == [("incomplete",), ("mono",), ("stereo",)]
    assert "preserved=14" in result.output
    assert "captures=3" in result.output
    assert "new=14" in result.output


def test_unchanged_rerun_verifies_without_copying(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    first = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert first.exit_code == 0, first.output

    def reject_copy(*args):
        raise AssertionError("unchanged files must not be copied")

    monkeypatch.setattr(inventory_module, "_copy_to_blob", reject_copy)
    second = runner.invoke(app, ["run", str(source), str(run_dir)])

    assert second.exit_code == 0, second.output
    assert "new=0" in second.output
    assert "unchanged=3" in second.output


def test_exact_duplicate_capture_is_canonical_once(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source / "a", "take", prefix=b"same-")
    write_capture(source / "b", "take", prefix=b"same-")
    write_capture(source / "c", "take", prefix=b"diff-")

    result = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert result.exit_code == 0, result.output
    snapshots = rows(
        run_dir / "run.sqlite",
        "SELECT parent_path, capture_id, is_canonical FROM capture_snapshot ORDER BY 1",
    )

    assert snapshots[0][1] == snapshots[1][1]
    assert snapshots[0][2:] == (1,)
    assert snapshots[1][2:] == (0,)
    assert snapshots[2][1] != snapshots[0][1]
    assert "unique_captures=2" in result.output
    assert "duplicate_captures=1" in result.output
    assert len(list((run_dir / "cache/blobs").iterdir())) == 6


def test_add_change_remove_reports_transitions_and_keeps_removed_mapping(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    notes = source / "notes.txt"
    notes.write_bytes(b"notes")
    assert runner.invoke(app, ["run", str(source), str(run_dir)]).exit_code == 0
    old_mapping = rows(
        run_dir / "run.sqlite",
        "SELECT source_sha256, cache_relative_path FROM source_file WHERE relative_path='notes.txt'",
    )[0]

    (source / "recording.mp4").write_bytes(b"X" * len(b"recording.mp4"))
    notes.unlink()
    (source / "recording.tel").write_bytes(b"tel")
    result = runner.invoke(app, ["run", str(source), str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "new=1" in result.output
    assert "changed=1" in result.output
    assert "unchanged=2" in result.output
    assert "removed=1" in result.output
    removed = rows(
        run_dir / "run.sqlite",
        "SELECT present, selected, preservation_status, source_sha256, cache_relative_path "
        "FROM source_file WHERE relative_path='notes.txt'",
    )[0]
    assert removed == (0, 0, "removed", *old_mapping)


def test_wrong_parent_and_duplicate_camera_member_are_not_complete(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    (source / "a").mkdir(parents=True)
    (source / "b").mkdir()
    for name in ("take_L.mp4", "take_L.vts", "take.imu"):
        (source / "a" / name).write_bytes(b"a")
    for name in ("take_R.mp4", "take_R.vts"):
        (source / "b" / name).write_bytes(b"b")
    result = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert result.exit_code == 0, result.output
    assert rows(
        run_dir / "run.sqlite",
        "SELECT parent_path, grouping_status FROM capture_candidate ORDER BY 1",
    ) == [("a", "incomplete"), ("b", "incomplete")]

    left = FileFact("one", "take_L.mp4", ".", "video", "take", "left", 1, 1)
    duplicate = replace(left, source_item_id="two", relative_path="copy/take_L.mp4")
    capture = group_captures((left, duplicate))[0]
    assert capture.grouping_status == "ambiguous"


def test_run_directory_inside_source_and_source_inside_cache_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = source / "recording.imu"
    original.write_bytes(b"immutable")
    before_hash = sha256(original.read_bytes()).hexdigest()
    inside = runner.invoke(app, ["run", str(source), str(source / "run")])
    assert inside.exit_code != 0
    assert "RUN_DIR must not be equal to or inside SOURCE" in inside.output
    assert sha256(original.read_bytes()).hexdigest() == before_hash

    run_dir = tmp_path / "run"
    cached_source = run_dir / "cache/source"
    cached_source.mkdir(parents=True)
    (cached_source / "recording.vts").write_bytes(b"vts")
    cached = runner.invoke(app, ["run", str(cached_source), str(run_dir)])
    assert cached.exit_code == 1
    assert "Cannot preserve a source inside RUN_DIR/cache" in cached.output


def test_inventory_write_rolls_back_as_one_transaction(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    source.mkdir()
    open_run(source.as_uri(), run_dir)
    database_path = run_dir / "run.sqlite"
    before = database_path.read_bytes()
    first = FileFact("duplicate", "one", ".", "video", "one", "single", 1, 1)
    broken = SourceInventory(
        source.as_uri(), (first, replace(first, source_item_id="other", size_bytes=-1)), ()
    )

    with pytest.raises(sqlite3.IntegrityError):
        store_inventory(database_path, broken)

    assert database_path.read_bytes() == before


def test_interrupted_copy_is_not_published_and_retry_succeeds(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    copy_bytes = inventory_module._copy_bytes

    def interrupt(source_file, destination, digest):
        destination.write(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(inventory_module, "_copy_bytes", interrupt)
    failed = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert failed.exit_code == 1
    assert "run_error=simulated interrupted copy" in failed.output
    assert not (run_dir / "cache/blobs").exists()
    assert rows(
        run_dir / "run.sqlite",
        "SELECT COUNT(*) FROM source_file WHERE source_sha256 IS NOT NULL",
    ) == [(0,)]

    monkeypatch.setattr(inventory_module, "_copy_bytes", copy_bytes)
    resumed = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert resumed.exit_code == 0, resumed.output
    assert "new=3" in resumed.output


def test_changed_copy_bytes_are_rejected_before_publish(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    source.mkdir()
    (source / "notes.txt").write_bytes(b"notes")

    def change_bytes(source_file, destination, digest):
        destination.write(b"x" * len(source_file.read()))

    monkeypatch.setattr(inventory_module, "_copy_bytes", change_bytes)
    failed = runner.invoke(app, ["run", str(source), str(run_dir)])

    assert failed.exit_code == 1
    assert "Copied bytes changed:" in failed.output
    assert not (run_dir / "cache/blobs").exists()


def test_corrupt_blob_fails_closed_without_overwrite(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    first = runner.invoke(app, ["run", str(source), str(run_dir)])
    assert first.exit_code == 0, first.output
    cache_path = rows(
        run_dir / "run.sqlite",
        "SELECT cache_relative_path FROM source_file WHERE relative_path='recording.imu'",
    )[0][0]
    blob = run_dir / cache_path
    blob.write_bytes(b"X" * blob.stat().st_size)

    failed = runner.invoke(app, ["run", str(source), str(run_dir)])

    assert failed.exit_code == 1
    assert "Immutable cache blob changed:" in failed.output
    assert blob.read_bytes() == b"X" * len(b"recording.imu")
    assert rows(
        run_dir / "run.sqlite",
        "SELECT COUNT(*) FROM source_file WHERE present=1 AND source_sha256 IS NOT NULL",
    ) == [(0,)]
    assert rows(run_dir / "run.sqlite", "SELECT COUNT(*) FROM capture_snapshot") == [(0,)]


def test_preservation_write_rolls_back_on_mid_insert_failure(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    inventory = inventory_module.inventory_local(str(source), run_dir)
    open_run(inventory.source_identity, run_dir)
    database_path = run_dir / "run.sqlite"
    store_inventory(database_path, inventory)
    file = PreservedFile("recording.imu", "a" * 64, "new")
    capture = (".", "recording", "b" * 64, True)
    broken = ((file,), (capture, capture), 0)

    with pytest.raises(sqlite3.IntegrityError):
        store_preservation(database_path, broken)

    assert rows(
        database_path,
        "SELECT source_sha256 FROM source_file WHERE source_item_id='recording.imu'",
    ) == [(None,)]
    assert rows(database_path, "SELECT COUNT(*) FROM capture_snapshot") == [(0,)]
