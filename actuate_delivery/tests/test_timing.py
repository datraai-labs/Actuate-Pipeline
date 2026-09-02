import struct
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from actuate_delivery.timing import TimingError, TimingStream, build_timing


def write_hash(path, content):
    path.write_bytes(content)
    return sha256(content).hexdigest()


def write_imu(path, timestamps=(100, 200, 300)):
    pq.write_table(pa.table({"timestamp_ns": pa.array(timestamps, type=pa.uint64())}), path)
    return sha256(path.read_bytes()).hexdigest()


def write_video(path, count):
    pq.write_table(
        pa.table(
            {
                "video_frame_index": range(count),
                "mp4_pts_ns": [index * 10 for index in range(count)],
            }
        ),
        path,
    )
    return sha256(path.read_bytes()).hexdigest()


def write_vts(path, sequences, timestamps):
    header = struct.pack("<8sIIqiHH", b"TRIVTS01", 4, 30000, 0, 0, 0, 0)
    rows = [
        struct.pack("<IQIQIII", frame, timestamp, sequence, frame * 10, 5000, 15, 26000)
        for frame, (sequence, timestamp) in enumerate(zip(sequences, timestamps, strict=True))
    ]
    return write_hash(path, header + b"".join(rows))


def stream(tmp_path, name, sequences, timestamps, video_count=None):
    vts = tmp_path / f"{name}.vts"
    video = tmp_path / f"{name}.parquet"
    return TimingStream(
        name,
        vts,
        write_vts(vts, sequences, timestamps),
        video,
        write_video(video, video_count or len(sequences)),
    )


def test_before_after_closest_coverage_and_outer_rows(tmp_path):
    imu = tmp_path / "imu.parquet"
    imu_hash = write_imu(imu)
    camera = stream(tmp_path, "single", range(4), (50, 150, 200, 350), 5)

    artifact = build_timing(imu, imu_hash, (camera,), tmp_path / "timing.parquet")
    rows = pq.read_table(tmp_path / "timing.parquet").to_pylist()

    assert (artifact.row_count, artifact.matched_rows, artifact.coverage_rows) == (5, 4, 2)
    assert [row["mapping_status"] for row in rows] == [
        "outside_imu_coverage",
        "mapped",
        "mapped",
        "outside_imu_coverage",
        "no_vts",
    ]
    assert (
        rows[0]["before_imu_index"],
        rows[0]["after_delta_ns"],
        rows[0]["closest_imu_index"],
    ) == (None, 50, 0)
    assert (
        rows[1]["before_delta_ns"],
        rows[1]["after_delta_ns"],
        rows[1]["closest_imu_index"],
    ) == (-50, 50, 0)
    assert (
        rows[2]["before_imu_index"],
        rows[2]["after_imu_index"],
        rows[2]["closest_delta_ns"],
    ) == (1, 1, 0)
    assert rows[4]["vts_match_status"] == "video_only"


def test_stereo_pairs_unique_encoder_sequence_not_frame_number(tmp_path):
    imu = tmp_path / "imu.parquet"
    imu_hash = write_imu(imu, (50, 100, 150, 200, 250))
    left = stream(tmp_path, "left", (69, 70, 71), (100, 150, 200))
    right = stream(tmp_path, "right", (70, 71), (151, 201))

    artifact = build_timing(imu, imu_hash, (left, right), tmp_path / "timing.parquet")
    rows = pq.read_table(tmp_path / "timing.parquet").to_pylist()
    left_rows = [row for row in rows if row["camera_stream_id"] == "left"]

    assert (artifact.stereo_pair_count, artifact.stereo_unmatched_rows) == (2, 1)
    assert left_rows[0]["stereo_pair_status"] == "unmatched"
    assert left_rows[1]["stereo_peer_video_frame_index"] == 0
    assert left_rows[1]["stereo_peer_vts_frame_number"] == 0
    assert left_rows[1]["vts_frame_number"] == 1


def test_missing_sof_vts_only_and_invalid_inputs_fail_without_fabrication(tmp_path):
    imu = tmp_path / "imu.parquet"
    imu_hash = write_imu(imu)
    camera = stream(tmp_path, "single", (1, 2), (0, 200), 1)
    output = tmp_path / "timing.parquet"

    build_timing(imu, imu_hash, (camera,), output)
    rows = pq.read_table(output).to_pylist()
    assert rows[0]["mapping_status"] == "missing_sof"
    assert rows[0]["closest_imu_index"] is None
    assert rows[1]["vts_match_status"] == "vts_only"

    with pytest.raises(TimingError, match="IMU Parquet SHA-256"):
        build_timing(imu, "0" * 64, (camera,), output)
    duplicate = stream(tmp_path, "left", (5, 5), (100, 200))
    right = stream(tmp_path, "right", (5, 6), (100, 200))
    with pytest.raises(TimingError, match="duplicate encoder sequence"):
        build_timing(imu, imu_hash, (duplicate, right), output)
