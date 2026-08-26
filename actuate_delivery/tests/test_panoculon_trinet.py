import sqlite3
import struct
from hashlib import sha256
from types import SimpleNamespace

import actuate_delivery.panoculon_trinet as trinet_module
import numpy as np
import pyarrow.parquet as pq
import pytest
from actuate_delivery.inventory import FileFact, PreservedFile, SourceInventory, group_captures
from actuate_delivery.panoculon_trinet import COLUMNS, ImuError, convert_imu
from actuate_delivery.run import (
    open_run,
    prepare_local_run,
    process_imus,
    store_inventory,
    store_preservation,
)


def run_pipeline(source, run_dir):
    result = prepare_local_run(str(source), run_dir)
    output = "\n".join(f"{key}={value}" for key, value in result.__dict__.items()) + "\n"
    failed = sum(value for key, value in result.__dict__.items() if key.endswith("_failed"))
    return SimpleNamespace(exit_code=int(bool(failed)), output=output)


def imu_bytes(version=4, timestamps=(100, 200), bad_float=None):
    sizes = {1: 9, 2: 17, 3: 18, 4: 18, 5: 18, 6: 18}
    floats = sizes.get(version, 18)
    header = bytearray(64)
    struct.pack_into("<8sIIHHQQ", header, 0, b"TRIMU001", version, 400, 2, 3, 50, 75)
    if version >= 3:
        struct.pack_into("<I", header, 36, 6 if version >= 5 else 1)
        header[40:56] = bytes(range(16))
    if version >= 4:
        struct.pack_into("<q", header, 56, -123)
    rows = []
    for row, timestamp in enumerate(timestamps):
        values = [float(row + index + 1) for index in range(floats)]
        if bad_float is not None and row == 0:
            values[bad_float] = float("nan")
        rows.append(struct.pack(f"<Q{floats}f", timestamp, *values))
    return bytes(header) + b"".join(rows)


def write_imu(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256(content).hexdigest()


def test_v4_round_trip_preserves_native_rows_and_metadata(tmp_path):
    source = tmp_path / "recording.imu"
    source_hash = write_imu(source, imu_bytes())
    output = tmp_path / "imu.parquet"

    artifact = convert_imu(source, output, source_hash)
    table = pq.read_table(output)

    assert artifact.sample_count == 2
    assert artifact.parquet_sha256 == sha256(output.read_bytes()).hexdigest()
    assert table.column_names == COLUMNS
    assert table["timestamp_ns"].to_pylist() == [100, 200]
    assert table["accel_x_mps2"].to_pylist() == [1.0, 2.0]
    assert table["gyro_z_rad_s"].to_pylist() == [6.0, 7.0]
    assert table["mag_x_ut"].to_pylist() == [7.0, 8.0]
    assert table["temperature_c"].to_pylist() == [10.0, 11.0]
    assert table["mag_age_us"].null_count == 2
    assert table.schema.metadata[b"source_sha256"].decode() == source_hash
    assert table.schema.metadata[b"native_sample_size"] == b"80"


def test_v1_has_nullable_temperature_without_changing_other_values(tmp_path):
    source = tmp_path / "legacy.imu"
    source_hash = write_imu(source, imu_bytes(version=1))
    output = tmp_path / "imu.parquet"

    convert_imu(source, output, source_hash)
    table = pq.read_table(output)

    assert table["temperature_c"].null_count == 2
    assert table["accel_z_mps2"].to_pylist() == [3.0, 4.0]
    assert table["mag_z_ut"].to_pylist() == [9.0, 10.0]
    assert table["mag_age_us"].null_count == 2


def test_v5_preserves_magnetometer_age_without_interpreting_negative_values(tmp_path):
    content = bytearray(imu_bytes(version=5))
    struct.pack_into("<f", content, 64 + 76, -41.0)
    source = tmp_path / "magnetometer.imu"
    source_hash = write_imu(source, bytes(content))

    convert_imu(source, tmp_path / "imu.parquet", source_hash)
    table = pq.read_table(tmp_path / "imu.parquet")

    assert table["mag_x_ut"].to_pylist() == [7.0, 8.0]
    assert table["mag_y_ut"].to_pylist() == [8.0, 9.0]
    assert table["mag_z_ut"].to_pylist() == [9.0, 10.0]
    assert table["mag_age_us"].to_pylist() == [-41.0, 19.0]


@pytest.mark.parametrize("version", range(1, 7))
def test_every_documented_version_decodes(tmp_path, version):
    source = tmp_path / f"v{version}.imu"
    source_hash = write_imu(source, imu_bytes(version=version))

    artifact = convert_imu(source, tmp_path / f"v{version}.parquet", source_hash)

    assert artifact.sample_count == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: b"NOTIMU01" + raw[8:], "Invalid IMU magic"),
        (lambda raw: raw[:8] + struct.pack("<I", 99) + raw[12:], "Unsupported TRIMU001 version"),
        (lambda raw: raw[:20], "header is truncated"),
        (lambda raw: raw + b"x", "not a whole number"),
        (lambda raw: raw[:64], "zero samples"),
        (lambda raw: imu_bytes(bad_float=0), "non-finite value"),
        (lambda raw: imu_bytes(timestamps=(100, 100)), "strictly increasing"),
        (lambda raw: imu_bytes(timestamps=(200, 100)), "strictly increasing"),
    ],
)
def test_invalid_native_structure_fails_for_the_exact_reason(tmp_path, change, message):
    source = tmp_path / "broken.imu"
    content = change(imu_bytes())
    source_hash = write_imu(source, content)

    with pytest.raises(ImuError, match=message):
        convert_imu(source, tmp_path / "imu.parquet", source_hash)


def test_wrong_preserved_hash_and_failed_reverification_publish_nothing(tmp_path, monkeypatch):
    source = tmp_path / "recording.imu"
    source_hash = write_imu(source, imu_bytes())
    output = tmp_path / "imu.parquet"
    with pytest.raises(ImuError, match="source SHA-256"):
        convert_imu(source, output, "0" * 64)

    def reject(*args):
        raise ImuError("simulated row mismatch")

    monkeypatch.setattr(trinet_module, "_verify_parquet", reject)
    with pytest.raises(ImuError, match="simulated row mismatch"):
        convert_imu(source, output, source_hash)
    assert not output.exists()
    assert not (tmp_path / ".imu.parquet.staging").exists()


def test_run_continues_other_captures_and_rejects_changed_artifact(tmp_path):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    write_imu(source / "good.imu", imu_bytes())
    write_imu(source / "bad.imu", imu_bytes(timestamps=(100, 100)))

    first = run_pipeline(source, run_dir)
    assert first.exit_code == 1
    assert "imu_decoded=1" in first.output
    assert "imu_failed=1" in first.output
    with sqlite3.connect(run_dir / "run.sqlite") as database:
        rows = database.execute(
            "SELECT status, parquet_relative_path, error FROM imu_artifact ORDER BY status"
        ).fetchall()
    assert rows[0][0:2] == ("decoded", rows[0][1])
    assert (run_dir / rows[0][1]).is_file()
    assert rows[1][0] == "failed"
    assert "strictly increasing" in rows[1][2]

    (run_dir / rows[0][1]).write_bytes(b"changed")
    second = run_pipeline(source, run_dir)
    assert second.exit_code == 1
    assert "imu_failed=2" in second.output
    assert sqlite3.connect(run_dir / "run.sqlite").execute(
        "SELECT COUNT(*) FROM imu_artifact "
        "WHERE status='failed' AND error='Published IMU Parquet changed after verification'"
    ).fetchone() == (1,)


def test_ambiguous_capture_does_not_choose_between_two_imus(tmp_path):
    run_dir = tmp_path / "run"
    first = FileFact("a.imu", "a.imu", ".", "imu", "take", None, 224, 1,
                     "local", ".", "application/octet-stream", None, None, True)
    second = FileFact("b.imu", "b.imu", ".", "imu", "take", None, 224, 1,
                      "local", ".", "application/octet-stream", None, None, True)
    inventory = SourceInventory("file:///source", (first, second), group_captures((first, second)))
    open_run(inventory.source_identity, run_dir)
    store_inventory(run_dir / "run.sqlite", inventory, inventory)
    source_hashes = ("a" * 64, "b" * 64)
    preservation = (
        (PreservedFile("a.imu", source_hashes[0], "new"),
         PreservedFile("b.imu", source_hashes[1], "new")),
        ((".", "take", "c" * 64, True),),
        0,
    )
    store_preservation(run_dir / "run.sqlite", preservation)

    assert process_imus(run_dir / "run.sqlite", run_dir) == (0, 0, 1)
    assert sqlite3.connect(run_dir / "run.sqlite").execute(
        "SELECT error FROM imu_artifact"
    ).fetchone() == ("Capture has 2 IMU members; expected exactly one",)


def test_float32_bit_patterns_survive_parquet(tmp_path):
    source = tmp_path / "recording.imu"
    source_hash = write_imu(source, imu_bytes())
    output = tmp_path / "imu.parquet"
    convert_imu(source, output, source_hash)

    stored = pq.read_table(output)["accel_x_mps2"].to_numpy().view(np.uint32)
    native = np.array([1.0, 2.0], dtype=np.float32).view(np.uint32)
    assert np.array_equal(stored, native)
