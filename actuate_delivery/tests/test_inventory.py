import json
import sqlite3
import struct
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import actuate_delivery.inventory as inventory_module
import actuate_delivery.run as run_module
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from actuate_delivery.inventory import (
    FileFact,
    PreservedFile,
    SourceInventory,
    group_captures,
    select_inventory,
)
from actuate_delivery.run import open_run, store_inventory, store_preservation
from actuate_delivery.video import VideoArtifact


def run_pipeline(source, run_dir):
    try:
        result = run_module.prepare_local_run(str(source), run_dir)
    except run_module.RunError as error:
        return SimpleNamespace(exit_code=1, output=f"run_error={error}\n")
    values = result.__dict__ | {"preserved": result.files}
    output = "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"
    failed = sum(value for key, value in values.items() if key.endswith("_failed"))
    return SimpleNamespace(exit_code=int(bool(failed)), output=output)


@pytest.fixture(autouse=True)
def stub_video_verification(monkeypatch):
    def verify(source, output, expected_sha256):
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"video_frame_index": [0, 1],
                                 "mp4_pts_ns": [0, 100]}), output)
        output_hash = sha256(output.read_bytes()).hexdigest()
        return VideoArtifact(output_hash, 2, "test", 1, 1,
                             "30/1", 1, 0, "{}")

    monkeypatch.setattr(run_module, "verify_video", verify)


def valid_imu(seed=b""):
    header = bytearray(64)
    struct.pack_into("<8sIIHHQQI", header, 0, b"TRIMU001", 4, 400, 2, 3, 50, 75, 1)
    value = float(sum(seed) % 10)
    rows = [struct.pack("<Q18f", timestamp, *([value + row] * 18))
            for row, timestamp in enumerate((100, 200))]
    return bytes(header) + b"".join(rows)


def valid_vts(seed=b""):
    offset = sum(seed) % 10
    header = struct.pack("<8sIIqiHH", b"TRIVTS01", 4, 30000, 0, 0, 0, 0)
    rows = [struct.pack("<IQIQIII", frame, 1_000_000 + offset + frame * 100,
                        frame + 3, 1_000 + frame, 5000, 15, 26000)
            for frame in range(2)]
    return header + b"".join(rows)


def valid_tel():
    header = bytearray(32)
    struct.pack_into("<8sIII", header, 0, b"TRTEL01\0", 1, 32, 2)
    rows = [struct.pack("<QiIIHBB", timestamp, 33910 + row, 0, 10240, 30, 0, 0)
            for row, timestamp in enumerate((100, 200))]
    return bytes(header) + b"".join(rows)


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
        content = valid_imu(prefix) if filename.endswith(".imu") else (
            valid_vts(prefix) if filename.endswith(".vts") else prefix + filename.encode())
        (folder / filename).write_bytes(content)


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

    result = run_pipeline(source, run_dir)
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
    assert "timing_created=2" in result.output
    assert "timing_unavailable=1" in result.output
    assert "qc_created=3" in result.output
    assert rows(
        database_path,
        "SELECT status, COUNT(*) FROM timing_artifact GROUP BY status ORDER BY status",
    ) == [("ready", 2), ("unavailable", 1)]
    assert rows(
        database_path,
        "SELECT status, COUNT(*) FROM qc_artifact GROUP BY status",
    ) == [("ready", 3)]
    mono_qc = rows(
        database_path,
        "SELECT json_relative_path FROM qc_artifact JOIN capture_snapshot USING(capture_id) "
        "WHERE parent_path='mono'",
    )[0][0]
    native = json.loads((run_dir / mono_qc).read_text())["facts"]["imu"]["native"]
    assert native["declared_sample_rate_hz"] == 400
    assert native["header_start_time_ns"] == 50
    assert native["first_sample_timestamp_ns"] == 100
    assert native["device_id_hex"] == "00" * 16


def test_explicit_selection_expands_stereo_sidecars_and_visualization(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source / "selected", "take", "stereo_pair")
    (source / "selected/visualization.mp4").write_bytes(b"visualization")
    write_capture(source / "unselected", "recording")
    (source / "notes.txt").write_bytes(b"notes")

    inventory = inventory_module.inventory_local(str(source), run_dir)
    selected = select_inventory(inventory, ("selected/take_L.mp4",))
    selected_paths = {file.relative_path for file in selected.files}
    assert selected_paths == {
        "selected/take.imu", "selected/take_L.mp4", "selected/take_L.vts",
        "selected/take_R.mp4", "selected/take_R.vts", "selected/visualization.mp4",
    }

    copy_to_blob = inventory_module._copy_to_blob

    def selected_copy(path, target_run_dir, fact):
        assert fact.relative_path in selected_paths
        return copy_to_blob(path, target_run_dir, fact)

    monkeypatch.setattr(inventory_module, "_copy_to_blob", selected_copy)

    result = run_module.prepare_local_run(
        str(source), run_dir, ("selected/take_L.mp4",)
    )
    assert result.files == 6
    assert result.captures == 1
    assert len(list((run_dir / "cache/blobs").iterdir())) == 5
    assert rows(
        run_dir / "run.sqlite",
        "SELECT relative_path, selected, source_sha256 IS NOT NULL "
        "FROM source_file ORDER BY relative_path",
    ) == [
        ("notes.txt", 0, 0),
        ("selected/take.imu", 1, 1),
        ("selected/take_L.mp4", 1, 1),
        ("selected/take_L.vts", 1, 1),
        ("selected/take_R.mp4", 1, 1),
        ("selected/take_R.vts", 1, 1),
        ("selected/visualization.mp4", 1, 1),
        ("unselected/recording.imu", 0, 0),
        ("unselected/recording.mp4", 0, 0),
        ("unselected/recording.vts", 0, 0),
    ]
    assert rows(
        run_dir / "run.sqlite",
        "SELECT source_type, parent_source_item_id, mime_type, can_download "
        "FROM source_file WHERE relative_path='selected/take_L.mp4'",
    ) == [("local", "selected", "video/mp4", 1)]


def test_unknown_selection_fails_before_run_creation(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")

    with pytest.raises(run_module.RunInputError, match="Unknown source item ID"):
        run_module.prepare_local_run(str(source), run_dir, ("missing",))

    assert not run_dir.exists()


def test_narrower_selection_retains_prior_unselected_blobs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source / "one", "take1")
    write_capture(source / "two", "take2")
    run_module.prepare_local_run(str(source), run_dir)
    prior = rows(
        run_dir / "run.sqlite",
        "SELECT relative_path, source_sha256 FROM source_file ORDER BY relative_path",
    )

    hash_file = inventory_module._hash_file

    def selected_hash(path):
        assert "/two/" not in path.as_posix()
        return hash_file(path)

    monkeypatch.setattr(inventory_module, "_hash_file", selected_hash)
    result = run_module.prepare_local_run(
        str(source), run_dir, ("one/take1.mp4",)
    )

    assert result.files == 3
    assert result.removed == 0
    assert rows(
        run_dir / "run.sqlite",
        "SELECT relative_path, source_sha256 FROM source_file ORDER BY relative_path",
    ) == prior
    assert rows(
        run_dir / "run.sqlite",
        "SELECT relative_path FROM source_file WHERE selected=0 ORDER BY relative_path",
    ) == [("two/take2.imu",), ("two/take2.mp4",), ("two/take2.vts",)]
    assert rows(
        run_dir / "run.sqlite", "SELECT parent_path FROM capture_snapshot"
    ) == [("one",)]


def test_unchanged_rerun_verifies_without_copying(tmp_path, monkeypatch):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    first = run_pipeline(source, run_dir)
    assert first.exit_code == 0, first.output

    def reject_copy(*args):
        raise AssertionError("unchanged files must not be copied")

    monkeypatch.setattr(inventory_module, "_copy_to_blob", reject_copy)
    second = run_pipeline(source, run_dir)

    assert second.exit_code == 0, second.output
    assert "new=0" in second.output
    assert "unchanged=3" in second.output
    assert "timing_reused=1" in second.output
    assert "qc_reused=1" in second.output


def test_changed_qc_output_fails_closed(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    assert run_pipeline(source, run_dir).exit_code == 0
    relative = rows(
        run_dir / "run.sqlite",
        "SELECT json_relative_path FROM qc_artifact WHERE status='ready'",
    )[0][0]
    (run_dir / relative).write_bytes(b"changed")

    result = run_pipeline(source, run_dir)

    assert result.exit_code == 1
    assert "qc_failed=1" in result.output
    assert rows(
        run_dir / "run.sqlite", "SELECT reason FROM qc_artifact WHERE status='failed'"
    ) == [("Published internal QC JSON changed after verification",)]


def test_changed_timing_output_fails_closed(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    assert run_pipeline(source, run_dir).exit_code == 0
    relative = rows(
        run_dir / "run.sqlite",
        "SELECT parquet_relative_path FROM timing_artifact WHERE status='ready'",
    )[0][0]
    (run_dir / relative).write_bytes(b"changed")

    result = run_pipeline(source, run_dir)

    assert result.exit_code == 1
    assert "timing_failed=1" in result.output
    assert rows(
        run_dir / "run.sqlite", "SELECT reason FROM timing_artifact WHERE status='failed'"
    ) == [("Published frame timing Parquet changed after verification",)]


def test_changed_video_index_fails_closed(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    assert run_pipeline(source, run_dir).exit_code == 0
    relative = rows(
        run_dir / "run.sqlite",
        "SELECT frame_index_relative_path FROM video_artifact WHERE status='verified'",
    )[0][0]
    (run_dir / relative).write_bytes(b"changed")

    result = run_pipeline(source, run_dir)

    assert result.exit_code == 1
    assert "video_failed=1" in result.output
    assert rows(
        run_dir / "run.sqlite", "SELECT error FROM video_artifact WHERE status='failed'"
    ) == [("Published video frame index changed after verification",)]


def test_exact_duplicate_capture_is_canonical_once(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source / "a", "take", prefix=b"same-")
    write_capture(source / "b", "take", prefix=b"same-")
    write_capture(source / "c", "take", prefix=b"diff-")

    result = run_pipeline(source, run_dir)
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
    assert run_pipeline(source, run_dir).exit_code == 0
    old_mapping = rows(
        run_dir / "run.sqlite",
        "SELECT source_sha256, cache_relative_path FROM source_file WHERE relative_path='notes.txt'",
    )[0]

    (source / "recording.mp4").write_bytes(b"X" * len(b"recording.mp4"))
    notes.unlink()
    (source / "recording.tel").write_bytes(valid_tel())
    result = run_pipeline(source, run_dir)

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
        content = valid_imu() if name.endswith(".imu") else (
            valid_vts() if name.endswith(".vts") else b"a")
        (source / "a" / name).write_bytes(content)
    for name in ("take_R.mp4", "take_R.vts"):
        (source / "b" / name).write_bytes(valid_vts() if name.endswith(".vts") else b"b")
    result = run_pipeline(source, run_dir)
    assert result.exit_code == 0, result.output
    assert rows(
        run_dir / "run.sqlite",
        "SELECT parent_path, grouping_status FROM capture_candidate ORDER BY 1",
    ) == [("a", "incomplete"), ("b", "incomplete")]

    left = FileFact("one", "take_L.mp4", ".", "video", "take", "left", 1, 1,
                    "local", ".", "video/mp4", None, None, True)
    duplicate = replace(left, source_item_id="two", relative_path="copy/take_L.mp4")
    capture = group_captures((left, duplicate))[0]
    assert capture.grouping_status == "ambiguous"


def test_run_directory_inside_source_and_source_inside_cache_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = source / "recording.imu"
    original.write_bytes(b"immutable")
    before_hash = sha256(original.read_bytes()).hexdigest()
    inside = run_pipeline(source, source / "run")
    assert inside.exit_code != 0
    assert "RUN_DIR must not be equal to or inside SOURCE" in inside.output
    assert sha256(original.read_bytes()).hexdigest() == before_hash

    run_dir = tmp_path / "run"
    cached_source = run_dir / "cache/source"
    cached_source.mkdir(parents=True)
    (cached_source / "recording.vts").write_bytes(b"vts")
    cached = run_pipeline(cached_source, run_dir)
    assert cached.exit_code == 1
    assert "Cannot preserve a source inside RUN_DIR/cache" in cached.output


def test_inventory_write_rolls_back_as_one_transaction(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    source.mkdir()
    open_run(source.as_uri(), run_dir)
    database_path = run_dir / "run.sqlite"
    before = database_path.read_bytes()
    first = FileFact("duplicate", "one", ".", "video", "one", "single", 1, 1,
                     "local", ".", "video/mp4", None, None, True)
    broken = SourceInventory(
        source.as_uri(), (first, replace(first, source_item_id="other", size_bytes=-1)), ()
    )

    with pytest.raises(sqlite3.IntegrityError):
        store_inventory(database_path, broken, broken)

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
    failed = run_pipeline(source, run_dir)
    assert failed.exit_code == 1
    assert "run_error=simulated interrupted copy" in failed.output
    assert not (run_dir / "cache/blobs").exists()
    assert rows(
        run_dir / "run.sqlite",
        "SELECT COUNT(*) FROM source_file WHERE source_sha256 IS NOT NULL",
    ) == [(0,)]

    monkeypatch.setattr(inventory_module, "_copy_bytes", copy_bytes)
    resumed = run_pipeline(source, run_dir)
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
    failed = run_pipeline(source, run_dir)

    assert failed.exit_code == 1
    assert "Copied bytes changed:" in failed.output
    assert not (run_dir / "cache/blobs").exists()


def test_corrupt_blob_fails_closed_without_overwrite(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_capture(source, "recording")
    first = run_pipeline(source, run_dir)
    assert first.exit_code == 0, first.output
    cache_path = rows(
        run_dir / "run.sqlite",
        "SELECT cache_relative_path FROM source_file WHERE relative_path='recording.imu'",
    )[0][0]
    blob = run_dir / cache_path
    blob_size = blob.stat().st_size
    blob.write_bytes(b"X" * blob_size)

    failed = run_pipeline(source, run_dir)

    assert failed.exit_code == 1
    assert "Immutable cache blob changed:" in failed.output
    assert blob.read_bytes() == b"X" * blob_size
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
    store_inventory(database_path, inventory, inventory)
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
