import sqlite3
import struct
from hashlib import sha256
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from actuate_delivery.inventory import FileFact, PreservedFile, SourceInventory, group_captures
from actuate_delivery.panoculon_trinet import SidecarError, convert_tel, decode_vts
from actuate_delivery.run import (
    open_run,
    prepare_local_run,
    process_imus,
    process_sidecars,
    store_inventory,
    store_preservation,
)


def run_pipeline(source, run_dir):
    result = prepare_local_run(str(source), run_dir)
    output = "\n".join(f"{key}={value}" for key, value in result.__dict__.items()) + "\n"
    failed = sum(value for key, value in result.__dict__.items() if key.endswith("_failed"))
    return SimpleNamespace(exit_code=int(bool(failed)), output=output)


def vts_bytes(version=4, frames=(0, 1), timestamps=(100, 200)):
    header = struct.pack("<8sIIqiHH", b"TRIVTS01", version, 30000, -9, 4, 2, 3)
    rows = []
    for frame, timestamp in zip(frames, timestamps, strict=True):
        if version == 1:
            rows.append(struct.pack("<IQ", frame, timestamp))
        elif version in (2, 3):
            rows.append(struct.pack("<IQIQ", frame, timestamp, frame + 3, timestamp // 1000))
        else:
            rows.append(struct.pack(
                "<IQIQIII", frame, timestamp, frame + 3, timestamp // 1000, 5000, 15, 26000))
    return header + b"".join(rows)


def tel_bytes(timestamps=(100, 200), declared=None):
    count = len(timestamps) if declared is None else declared
    header = bytearray(32)
    struct.pack_into("<8sIII", header, 0, b"TRTEL01\0", 1, 32, count)
    header[24:32] = b"device01"
    rows = [struct.pack("<QiIIHBB", timestamp, 33910 + row, 0, 10240, 30, 0, 1)
            for row, timestamp in enumerate(timestamps)]
    return bytes(header) + b"".join(rows)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256(content).hexdigest()


@pytest.mark.parametrize("version", range(1, 5))
def test_every_documented_vts_version_decodes(tmp_path, version):
    source = tmp_path / f"v{version}.vts"
    source_hash = write(source, vts_bytes(version))

    data = decode_vts(source, source_hash)

    assert data.version == version
    assert data.frame_rate_milli == 30000
    assert data.entries["frame_number"].tolist() == [0, 1]
    if version >= 3:
        assert (data.master_clock_offset_ns, data.clock_skew_ppb,
                data.sync_quality_us, data.sync_flags) == (-9, 4, 2, 3)
    else:
        assert (data.master_clock_offset_ns, data.sync_flags) == (0, 0)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (lambda: b"BADVTS01" + vts_bytes()[8:], "Invalid VTS magic"),
        (lambda: vts_bytes(4)[:8] + struct.pack("<I", 9) + vts_bytes(4)[12:], "Unsupported"),
        (lambda: vts_bytes()[:20], "header is truncated"),
        (lambda: vts_bytes() + b"x", "whole number"),
        (lambda: vts_bytes(frames=(), timestamps=()), "zero entries"),
        (lambda: vts_bytes(frames=(0, 0)), "contiguous from zero"),
        (lambda: vts_bytes(timestamps=(100, 100)), "strictly increasing"),
    ],
)
def test_invalid_vts_fails_for_exact_reason(tmp_path, content, message):
    source = tmp_path / "broken.vts"
    source_hash = write(source, content())

    with pytest.raises(SidecarError, match=message):
        decode_vts(source, source_hash)


def test_tel_round_trip_preserves_native_rows_and_metadata(tmp_path):
    source = tmp_path / "take.tel"
    source_hash = write(source, tel_bytes())
    output = tmp_path / "telemetry.parquet"

    artifact = convert_tel(source, output, source_hash)
    table = pq.read_table(output)

    assert artifact.record_count == 2
    assert artifact.parquet_sha256 == sha256(output.read_bytes()).hexdigest()
    assert table["timestamp_ns"].to_pylist() == [100, 200]
    assert table["device_temperature_c"].to_pylist() == pytest.approx([33.91, 33.911])
    assert table["configured_bitrate_kbps"].to_pylist() == [10240, 10240]
    assert table.schema.metadata[b"source_sha256"].decode() == source_hash
    assert table.schema.metadata[b"native_device_id"] == b"6465766963653031"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (lambda: b"BADTEL01" + tel_bytes()[8:], "Invalid TEL header"),
        (lambda: tel_bytes()[:20], "header is truncated"),
        (lambda: tel_bytes((), declared=0), "zero records"),
        (lambda: tel_bytes(declared=3), "record count"),
        (lambda: tel_bytes((100, 100)), "strictly increasing"),
    ],
)
def test_invalid_tel_fails_for_exact_reason(tmp_path, content, message):
    source = tmp_path / "broken.tel"
    source_hash = write(source, content())

    with pytest.raises(SidecarError, match=message):
        convert_tel(source, tmp_path / "telemetry.parquet", source_hash)


def test_run_reuses_sidecars_and_rejects_changed_tel_artifact(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write(source / "take.vts", vts_bytes())
    write(source / "take.tel", tel_bytes())

    first = run_pipeline(source, run_dir)
    second = run_pipeline(source, run_dir)
    assert first.exit_code == second.exit_code == 0
    assert "vts_decoded=1" in first.output and "tel_decoded=1" in first.output
    assert "vts_reused=1" in second.output and "tel_reused=1" in second.output

    with sqlite3.connect(run_dir / "run.sqlite") as database:
        relative = database.execute(
            "SELECT parquet_relative_path FROM tel_artifact WHERE status='decoded'"
        ).fetchone()[0]
    (run_dir / relative).write_bytes(b"changed")
    third = run_pipeline(source, run_dir)

    assert third.exit_code == 1
    assert "vts_reused=1" in third.output and "tel_failed=1" in third.output


def test_duplicate_vts_and_tel_members_fail_without_selection(tmp_path):
    run_dir = tmp_path / "run"
    files = (
        FileFact("a.vts", "a.vts", ".", "vts", "take", "left", 1, 1,
                 "local", ".", "application/octet-stream", None, None, True),
        FileFact("b.vts", "b.vts", ".", "vts", "take", "left", 1, 1,
                 "local", ".", "application/octet-stream", None, None, True),
        FileFact("a.tel", "a.tel", ".", "telemetry", "take", None, 1, 1,
                 "local", ".", "application/octet-stream", None, None, True),
        FileFact("b.tel", "b.tel", ".", "telemetry", "take", None, 1, 1,
                 "local", ".", "application/octet-stream", None, None, True),
    )
    inventory = SourceInventory("file:///source", files, group_captures(files))
    open_run(inventory.source_identity, run_dir)
    store_inventory(run_dir / "run.sqlite", inventory, inventory)
    preserved = tuple(
        PreservedFile(file.source_item_id, str(index) * 64, "new")
        for index, file in enumerate(files, 1)
    )
    store_preservation(
        run_dir / "run.sqlite", (preserved, ((".", "take", "f" * 64, True),), 0))
    process_imus(run_dir / "run.sqlite", run_dir)

    assert process_sidecars(run_dir / "run.sqlite", run_dir) == (0, 0, 1, 0, 0, 1)
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        vts_error = database.execute("SELECT error FROM vts_artifact").fetchone()[0]
        tel_error = database.execute("SELECT error FROM tel_artifact").fetchone()[0]
    assert vts_error == "Camera stream has 2 VTS members; expected one"
    assert tel_error == "Capture has 2 TEL members; expected one"
