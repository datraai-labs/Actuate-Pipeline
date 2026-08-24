import struct
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HEADER_SIZE = 64
TOOLKIT_REVISION = "aee862fbd1c475e43ad59d8ef8297e9bc6411878"
BASE_FIELDS = [
    ("timestamp_ns", "<u8"),
    ("accel", "<f4", (3,)),
    ("gyro", "<f4", (3,)),
    ("mag", "<f4", (3,)),
]
V1_DTYPE = np.dtype(BASE_FIELDS)
V2_DTYPE = np.dtype(BASE_FIELDS + [
    ("temperature_c", "<f4"),
    ("quat_xyzw", "<f4", (4,)),
    ("linear_accel", "<f4", (3,)),
])
V3_DTYPE = np.dtype(V2_DTYPE.descr + [("trailing_value", "<f4")])
DTYPES = {1: V1_DTYPE, 2: V2_DTYPE, 3: V3_DTYPE, 4: V3_DTYPE, 5: V3_DTYPE, 6: V3_DTYPE}
COLUMNS = ["sample_index", "timestamp_ns",
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s", "temperature_c"]
VTS_DTYPES = {
    1: np.dtype([("frame_number", "<u4"), ("timestamp_ns", "<u8")]),
    2: np.dtype([("frame_number", "<u4"), ("sof_timestamp_ns", "<u8"),
        ("venc_seq", "<u4"), ("venc_pts_us", "<u8")]),
}
VTS_DTYPES[3] = VTS_DTYPES[2]
VTS_DTYPES[4] = np.dtype(VTS_DTYPES[2].descr + [("exposure_us", "<u4"),
    ("timing_flags", "<u4"), ("readout_time_us", "<u4")])
TEL_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"), ("device_temperature_millic", "<i4"),
    ("cpu_frequency_khz", "<u4"), ("configured_bitrate_kbps", "<u4"),
    ("configured_framerate_fps", "<u2"), ("thermal_state_code", "u1"), ("led_state_code", "u1")])
TEL_COLUMNS = ["record_index", "timestamp_ns", "device_temperature_c",
    "cpu_frequency_khz", "configured_bitrate_kbps", "configured_framerate_fps",
    "thermal_state_code", "led_state_code"]


class ImuError(ValueError):
    pass


class SidecarError(ValueError):
    pass


@dataclass(frozen=True)
class ImuData:
    version: int
    sample_rate_hz: int
    accel_fs: int
    gyro_fs: int
    start_time_ns: int
    video_start_ns: int
    flags: int
    device_id: bytes
    ios_host_offset_ns: int | None
    reserved_header: bytes
    samples: np.ndarray


@dataclass(frozen=True)
class ImuArtifact:
    parquet_sha256: str
    sample_count: int


@dataclass(frozen=True)
class VtsData:
    version: int
    frame_rate_milli: int
    master_clock_offset_ns: int
    clock_skew_ppb: int
    sync_quality_us: int
    sync_flags: int
    entries: np.ndarray


@dataclass(frozen=True)
class TelArtifact:
    parquet_sha256: str
    record_count: int


def decode_imu(source: Path, expected_sha256: str) -> ImuData:
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ImuError("IMU source SHA-256 does not match the preserved mapping")
    if len(raw) < HEADER_SIZE:
        raise ImuError(f"TRIMU001 header is truncated: {len(raw)} of {HEADER_SIZE} bytes")

    magic, version, rate, accel_fs, gyro_fs, start, video = struct.unpack_from(
        "<8sIIHHQQ", raw
    )
    if magic != b"TRIMU001":
        raise ImuError(f"Invalid IMU magic: {magic!r}")
    if version not in DTYPES:
        raise ImuError(f"Unsupported TRIMU001 version: {version}")
    if rate == 0:
        raise ImuError("IMU sample_rate_hz must be greater than zero")
    if accel_fs not in range(4) or gyro_fs not in range(4):
        raise ImuError(f"Invalid IMU full-scale codes: accel={accel_fs}, gyro={gyro_fs}")

    dtype = DTYPES[version]
    body_size = len(raw) - HEADER_SIZE
    if body_size % dtype.itemsize:
        raise ImuError(f"IMU body is not a whole number of {dtype.itemsize}-byte samples: {body_size} bytes")
    if body_size == 0:
        raise ImuError("IMU file contains zero samples")
    samples = np.frombuffer(raw, dtype=dtype, offset=HEADER_SIZE).copy()
    if np.any(samples["timestamp_ns"][1:] <= samples["timestamp_ns"][:-1]):
        raise ImuError("IMU timestamps must be strictly increasing")
    for name in dtype.names:
        if np.issubdtype(dtype[name].base, np.floating) and not np.isfinite(samples[name]).all():
            raise ImuError(f"IMU field contains a non-finite value: {name}")

    flags = struct.unpack_from("<I", raw, 36)[0] if version >= 3 else 0
    device_id = raw[40:56] if version >= 3 else b"\0" * 16
    ios_offset = struct.unpack_from("<q", raw, 56)[0] if version >= 4 else None
    return ImuData(version, rate, accel_fs, gyro_fs, start, video, flags, device_id,
                   ios_offset, raw[36:64], samples)


def _metadata(data: ImuData, source_sha256: str) -> dict[bytes, bytes]:
    values = {
        "schema_version": "trinet_delivery.imu.v1",
        "source_sha256": source_sha256,
        "native_format": "TRIMU001",
        "native_version": str(data.version),
        "native_sample_rate_hz": str(data.sample_rate_hz),
        "native_sample_size": str(data.samples.dtype.itemsize),
        "native_sample_count": str(len(data.samples)),
        "decoder": "trinet_delivery.trinet",
        "decoder_reference": f"Panoculon-Labs/Trinet-tools@{TOOLKIT_REVISION}",
        "write_parameters": "parquet=2.6;compression=zstd;dictionary=false;statistics=true",
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def _table(data: ImuData, source_sha256: str) -> pa.Table:
    samples = data.samples
    arrays = [
        pa.array(np.arange(len(samples), dtype=np.int64)),
        pa.array(samples["timestamp_ns"], type=pa.uint64()),
        *(pa.array(samples[field][:, axis], type=pa.float32())
          for field in ("accel", "gyro") for axis in range(3)),
        (pa.nulls(len(samples), type=pa.float32()) if data.version == 1
         else pa.array(samples["temperature_c"], type=pa.float32())),
    ]
    return pa.Table.from_arrays(arrays, names=COLUMNS).replace_schema_metadata(
        _metadata(data, source_sha256))


def _verify_parquet(path: Path, data: ImuData, source_sha256: str) -> None:
    table = pq.read_table(path)
    if table.column_names != COLUMNS or table.num_rows != len(data.samples):
        raise ImuError("IMU Parquet schema or row count does not match the native decode")
    if table.schema.metadata != _metadata(data, source_sha256):
        raise ImuError("IMU Parquet provenance metadata does not match the conversion")
    if not np.array_equal(table["sample_index"].to_numpy(), np.arange(len(data.samples))):
        raise ImuError("IMU Parquet sample indexes do not match native row order")
    if not np.array_equal(table["timestamp_ns"].to_numpy(), data.samples["timestamp_ns"]):
        raise ImuError("IMU Parquet timestamps do not match native values")

    native_columns = [
        *(data.samples[field][:, axis] for field in ("accel", "gyro") for axis in range(3)),
        None if data.version == 1 else data.samples["temperature_c"],
    ]
    for name, native in zip(COLUMNS[2:], native_columns, strict=True):
        column = table[name]
        if native is None:
            if column.null_count != len(data.samples):
                raise ImuError("IMU Parquet temperature must be null for native version 1")
            continue
        stored = column.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        if not np.array_equal(stored.view(np.uint32), native.view(np.uint32)):
            raise ImuError(f"IMU Parquet float bits do not match native values: {name}")


def convert_imu(source: Path, output: Path, expected_sha256: str) -> ImuArtifact:
    data = decode_imu(source, expected_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        pq.write_table(
            _table(data, expected_sha256), staging, version="2.6", compression="zstd",
            use_dictionary=False, write_statistics=True,
        )
        _verify_parquet(staging, data, expected_sha256)
        parquet_hash = sha256(staging.read_bytes()).hexdigest()
        staging.replace(output)
    except (OSError, pa.ArrowException) as error:
        raise ImuError(f"IMU Parquet write or read failed: {error}") from error
    finally:
        staging.unlink(missing_ok=True)
    return ImuArtifact(parquet_hash, len(data.samples))


def _verified_bytes(source: Path, expected_sha256: str, name: str) -> bytes:
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise SidecarError(f"{name} source SHA-256 does not match the preserved mapping")
    return raw


def decode_vts(source: Path, expected_sha256: str) -> VtsData:
    raw = _verified_bytes(source, expected_sha256, "VTS")
    if len(raw) < 32:
        raise SidecarError(f"TRIVTS01 header is truncated: {len(raw)} of 32 bytes")
    magic, version, rate = struct.unpack_from("<8sII", raw)
    if magic != b"TRIVTS01":
        raise SidecarError(f"Invalid VTS magic: {magic!r}")
    if version not in VTS_DTYPES:
        raise SidecarError(f"Unsupported TRIVTS01 version: {version}")
    if rate == 0:
        raise SidecarError("VTS frame_rate_milli must be greater than zero")
    dtype = VTS_DTYPES[version]
    body_size = len(raw) - 32
    if body_size % dtype.itemsize:
        raise SidecarError(f"VTS body is not a whole number of {dtype.itemsize}-byte entries")
    if body_size == 0:
        raise SidecarError("VTS file contains zero entries")
    entries = np.frombuffer(raw, dtype=dtype, offset=32).copy()
    expected = np.arange(len(entries), dtype=np.uint32)
    if not np.array_equal(entries["frame_number"], expected):
        raise SidecarError("VTS frame_number must be contiguous from zero")
    timestamp_field = "timestamp_ns" if version == 1 else "sof_timestamp_ns"
    timestamps = entries[timestamp_field]
    usable = timestamps[timestamps != 0]
    if np.any(usable[1:] <= usable[:-1]):
        raise SidecarError(f"VTS {timestamp_field} values must be strictly increasing when nonzero")
    sync = struct.unpack_from("<qiHH", raw, 16) if version >= 3 else (0, 0, 0, 0)
    return VtsData(version, rate, *sync, entries)


def convert_tel(source: Path, output: Path, expected_sha256: str) -> TelArtifact:
    raw = _verified_bytes(source, expected_sha256, "TEL")
    if len(raw) < 32:
        raise SidecarError(f"TRTEL01 header is truncated: {len(raw)} of 32 bytes")
    magic, version, header_size, count = struct.unpack_from("<8sIII", raw)
    if magic != b"TRTEL01\0" or version != 1 or header_size != 32:
        raise SidecarError(
            f"Invalid TEL header: magic={magic!r}, version={version}, header_size={header_size}")
    if count == 0:
        raise SidecarError("TEL file declares zero records")
    if len(raw) != 32 + count * TEL_DTYPE.itemsize:
        raise SidecarError("TEL declared record count does not match file size")
    records = np.frombuffer(raw, dtype=TEL_DTYPE, offset=32).copy()
    if np.any(records["timestamp_ns"][1:] <= records["timestamp_ns"][:-1]):
        raise SidecarError("TEL timestamps must be strictly increasing")
    table = pa.Table.from_arrays([pa.array(np.arange(count), type=pa.int64()),
        pa.array(records["timestamp_ns"], type=pa.uint64()),
        pa.array(records["device_temperature_millic"] / np.float32(1000), type=pa.float32()),
        *(pa.array(records[name]) for name in TEL_DTYPE.names[2:]),
    ], names=TEL_COLUMNS)
    metadata = {
        "schema_version": "trinet_delivery.telemetry.v1", "source_sha256": expected_sha256,
        "native_format": "TRTEL01", "native_version": str(version),
        "native_record_count": str(count), "native_device_id": raw[24:32].hex(),
        "temperature_conversion": "float32(temp_milli_c/1000)",
        "decoder_reference": f"Panoculon-Labs/Trinet-tools@{TOOLKIT_REVISION}",
        "write_parameters": "parquet=2.6;compression=zstd;dictionary=false;statistics=true",
    }
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        pq.write_table(table, staging, version="2.6", compression="zstd",
                       use_dictionary=False, write_statistics=True)
        if not pq.read_table(staging).equals(table, check_metadata=True):
            raise SidecarError("TEL Parquet does not match the native decode")
        parquet_hash = sha256(staging.read_bytes()).hexdigest()
        staging.replace(output)
    except (OSError, pa.ArrowException) as error:
        raise SidecarError(f"TEL Parquet write or read failed: {error}") from error
    finally:
        staging.unlink(missing_ok=True)
    return TelArtifact(parquet_hash, count)
