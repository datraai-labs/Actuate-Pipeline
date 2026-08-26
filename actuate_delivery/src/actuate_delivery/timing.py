import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from actuate_delivery.panoculon_trinet import SidecarError, decode_vts


class TimingError(ValueError):
    pass


@dataclass(frozen=True)
class TimingStream:
    camera_stream_id: str
    vts_path: Path
    vts_sha256: str
    video_index_path: Path
    video_index_sha256: str


@dataclass(frozen=True)
class TimingArtifact:
    parquet_sha256: str
    row_count: int
    matched_rows: int
    coverage_rows: int
    stereo_pair_count: int
    stereo_unmatched_rows: int


SCHEMA = pa.schema([
    ("camera_stream_id", pa.string()),
    ("video_frame_index", pa.int64()),
    ("mp4_pts_ns", pa.int64()),
    ("vts_frame_number", pa.uint32()),
    ("venc_seq", pa.uint32()),
    ("venc_pts_us", pa.uint64()),
    ("sof_timestamp_ns", pa.uint64()),
    ("vts_match_status", pa.string()),
    ("before_imu_index", pa.int64()),
    ("before_imu_timestamp_ns", pa.uint64()),
    ("before_delta_ns", pa.int64()),
    ("after_imu_index", pa.int64()),
    ("after_imu_timestamp_ns", pa.uint64()),
    ("after_delta_ns", pa.int64()),
    ("closest_imu_index", pa.int64()),
    ("closest_imu_timestamp_ns", pa.uint64()),
    ("closest_delta_ns", pa.int64()),
    ("within_imu_coverage", pa.bool_()),
    ("mapping_status", pa.string()),
    ("stereo_peer_stream_id", pa.string()),
    ("stereo_peer_video_frame_index", pa.int64()),
    ("stereo_peer_vts_frame_number", pa.uint32()),
    ("stereo_pair_status", pa.string()),
])


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _query_imu(timestamps: np.ndarray, frame_time: int) -> dict:
    before = int(np.searchsorted(timestamps, frame_time, side="right")) - 1
    after = int(np.searchsorted(timestamps, frame_time, side="left"))
    before = before if before >= 0 else None
    after = after if after < len(timestamps) else None
    if before is None:
        closest = after
    elif after is None:
        closest = before
    else:
        before_distance = frame_time - int(timestamps[before])
        after_distance = int(timestamps[after]) - frame_time
        closest = before if before_distance <= after_distance else after
    values = {}
    for name, index in (("before", before), ("after", after), ("closest", closest)):
        values[f"{name}_imu_index"] = index
        values[f"{name}_imu_timestamp_ns"] = None if index is None else int(timestamps[index])
        values[f"{name}_delta_ns"] = None if index is None else int(timestamps[index]) - frame_time
    values["within_imu_coverage"] = int(timestamps[0]) <= frame_time <= int(timestamps[-1])
    return values


def build_timing(
    imu_path: Path,
    imu_sha256: str,
    streams: tuple[TimingStream, ...],
    output: Path,
) -> TimingArtifact:
    stream_ids = tuple(stream.camera_stream_id for stream in streams)
    if stream_ids not in (("single",), ("left", "right")):
        raise TimingError(f"Unsupported camera stream layout: {stream_ids}")
    if _hash(imu_path) != imu_sha256:
        raise TimingError("IMU Parquet SHA-256 does not match its verified artifact")
    try:
        imu = pq.read_table(imu_path)
    except (OSError, pa.ArrowException) as error:
        raise TimingError(f"Cannot read verified IMU Parquet: {error}") from error
    if "timestamp_ns" not in imu.column_names or imu.num_rows == 0:
        raise TimingError("IMU Parquet has no timestamp rows")
    imu_timestamps = imu["timestamp_ns"].to_numpy()
    if np.any(imu_timestamps[1:] <= imu_timestamps[:-1]):
        raise TimingError("IMU Parquet timestamps must be strictly increasing")

    rows = []
    input_hashes = {"imu.parquet": imu_sha256}
    for stream in streams:
        if _hash(stream.video_index_path) != stream.video_index_sha256:
            raise TimingError(f"Video frame index changed: {stream.camera_stream_id}")
        try:
            video = pq.read_table(stream.video_index_path)
            vts = decode_vts(stream.vts_path, stream.vts_sha256)
        except (OSError, pa.ArrowException, SidecarError) as error:
            raise TimingError(f"Cannot read timing input for {stream.camera_stream_id}: {error}") from error
        required = {"video_frame_index", "mp4_pts_ns"}
        if not required <= set(video.column_names):
            raise TimingError(f"Video frame index lacks required columns: {stream.camera_stream_id}")
        video_indexes = video["video_frame_index"].to_pylist()
        if video_indexes != list(range(len(video_indexes))):
            raise TimingError(f"Video frame indexes are not contiguous: {stream.camera_stream_id}")
        input_hashes[f"{stream.camera_stream_id}.vts"] = stream.vts_sha256
        input_hashes[f"{stream.camera_stream_id}.video_index"] = stream.video_index_sha256
        for index in range(max(len(video_indexes), len(vts.entries))):
            has_video, has_vts = index < len(video_indexes), index < len(vts.entries)
            row = dict.fromkeys(SCHEMA.names)
            row["camera_stream_id"] = stream.camera_stream_id
            row["video_frame_index"] = index if has_video else None
            row["mp4_pts_ns"] = video["mp4_pts_ns"][index].as_py() if has_video else None
            row["vts_match_status"] = "matched" if has_video and has_vts else (
                "video_only" if has_video else "vts_only")
            row["mapping_status"] = "no_vts" if not has_vts else "no_video"
            row["stereo_pair_status"] = "not_applicable" if len(streams) == 1 else "not_available"
            if has_vts:
                entry = vts.entries[index]
                row["vts_frame_number"] = int(entry["frame_number"])
                time_field = "timestamp_ns" if vts.version == 1 else "sof_timestamp_ns"
                row["sof_timestamp_ns"] = int(entry[time_field])
                if vts.version >= 2:
                    for name in ("venc_seq", "venc_pts_us"):
                        row[name] = int(entry[name])
            if has_video and has_vts:
                if row["sof_timestamp_ns"] == 0:
                    row["mapping_status"] = "missing_sof"
                else:
                    row.update(_query_imu(imu_timestamps, row["sof_timestamp_ns"]))
                    row["mapping_status"] = (
                        "mapped" if row["within_imu_coverage"] else "outside_imu_coverage")
            rows.append(row)

    stereo_pairs = stereo_unmatched = 0
    if len(streams) == 2:
        by_stream = {}
        for stream_id in ("left", "right"):
            candidates = [row for row in rows
                          if row["camera_stream_id"] == stream_id and row["vts_frame_number"] is not None]
            if any(row["venc_seq"] is None for row in candidates):
                raise TimingError(f"Stereo VTS has no encoder sequence: {stream_id}")
            sequence_map = {row["venc_seq"]: row for row in candidates}
            if len(sequence_map) != len(candidates):
                raise TimingError(f"Stereo VTS has duplicate encoder sequence: {stream_id}")
            by_stream[stream_id] = sequence_map
        for stream_id, peer_id in (("left", "right"), ("right", "left")):
            for sequence, row in by_stream[stream_id].items():
                peer = by_stream[peer_id].get(sequence)
                row["stereo_pair_status"] = "matched" if peer else "unmatched"
                if peer:
                    row["stereo_peer_stream_id"] = peer_id
                    row["stereo_peer_video_frame_index"] = peer["video_frame_index"]
                    row["stereo_peer_vts_frame_number"] = peer["vts_frame_number"]
        stereo_pairs = len(set(by_stream["left"]) & set(by_stream["right"]))
        stereo_unmatched = len(set(by_stream["left"]) ^ set(by_stream["right"]))

    metadata = {
        "schema_version": "actuate_delivery.frame_timing.v1",
        "camera_time_basis": "native_vts_timestamp_v1_or_sof_timestamp_v2_plus",
        "video_vts_join": "outer_by_per_stream_row_order;native_identities_preserved",
        "imu_query": "Before<=t;After>=t;Closest=min_abs;exact_tie=Before",
        "imu_delta_ns": "imu_timestamp_ns-sof_timestamp_ns",
        "stereo_join": "unique_equal_venc_seq",
        "input_hashes": json.dumps(input_hashes, sort_keys=True, separators=(",", ":")),
        "write_parameters": "parquet=2.6;compression=zstd;dictionary=false;statistics=true",
    }
    table = pa.Table.from_pylist(rows, schema=SCHEMA).replace_schema_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        pq.write_table(table, staging, version="2.6", compression="zstd",
                       use_dictionary=False, write_statistics=True)
        if not pq.read_table(staging).equals(table, check_metadata=True):
            raise TimingError("Frame timing Parquet does not match the computed mapping")
        parquet_hash = sha256(staging.read_bytes()).hexdigest()
        staging.replace(output)
    except (OSError, pa.ArrowException) as error:
        raise TimingError(f"Frame timing write or read failed: {error}") from error
    finally:
        staging.unlink(missing_ok=True)
    matched = sum(row["vts_match_status"] == "matched" for row in rows)
    coverage = sum(row["mapping_status"] == "mapped" for row in rows)
    return TimingArtifact(parquet_hash, len(rows), matched, coverage,
                          stereo_pairs, stereo_unmatched)
