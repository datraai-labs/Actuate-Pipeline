from __future__ import annotations

import json

import numpy as np
import pytest

from actuate.ingest.imu import sync_imu

h5py = pytest.importorskip("h5py")


def _meta(path, frame_count=4, fps=30.0):
    meta = {
        "session_id": path.name,
        "frame_count": frame_count,
        "fps_nominal": fps,
    }
    (path / "session_meta.json").write_text(json.dumps(meta))
    return meta


def test_syncs_one_based_nearest_frame_records_and_ignores_trailing_frame(tmp_path):
    meta = _meta(tmp_path, frame_count=4)
    records = []
    # Two sensor readings per video frame, plus a capture-tool boundary record for frame N+1.
    for frame in range(1, 6):
        for sample in range(2):
            records.append(
                {
                    "timestamp_ns": frame * 1_000_000 + sample,
                    "nearest_video_frame": frame,
                    "video_frame_timestamp_ns": frame * 33_333_333,
                    "gyro": [frame + sample, frame * 2, frame * 3],
                    "accel": [0.0, 0.0, 9.81],
                    "mag": [1.0, 2.0, 3.0],
                    "temp_c": 30 + frame,
                }
            )
    (tmp_path / "imu.json").write_text(json.dumps(records))

    result = sync_imu(tmp_path, meta)

    assert result is not None
    assert result.raw_samples == 10
    assert result.frame_count == 4
    assert result.sync_method == "nearest_video_frame_mean_1based"
    with h5py.File(result.session_h5, "r") as h5:
        assert h5["imu/gyro"].shape == (4, 3)
        assert np.allclose(h5["imu/gyro"][:, 0], [1.5, 2.5, 3.5, 4.5])
        assert h5["imu/accel"].shape == (4, 3)
        assert h5["imu/timestamp_ns"].shape == (4,)
        assert h5["imu/video_timestamp_ns"].shape == (4,)

    updated = json.loads((tmp_path / "session_meta.json").read_text())
    assert updated["modalities"]["imu"] is True
    assert updated["imu"]["aligned_frames"] == 4

    cached = sync_imu(tmp_path, updated)
    assert cached is not None and cached.cached


def test_interpolates_timestamp_only_stream_onto_video_clock(tmp_path):
    meta = _meta(tmp_path, frame_count=3, fps=10.0)
    records = [
        {"timestamp_ns": 1_000_000_000, "gyro": [0, 0, 0], "accel": [0, 0, 9.8]},
        {"timestamp_ns": 1_200_000_000, "gyro": [2, 4, 6], "accel": [0, 0, 9.8]},
    ]
    (tmp_path / "imu.json").write_text(json.dumps({"samples": records}))

    result = sync_imu(tmp_path, meta)

    assert result is not None
    assert result.sync_method == "timestamp_interpolation"
    with h5py.File(result.session_h5, "r") as h5:
        assert np.allclose(h5["imu/gyro"][:], [[0, 0, 0], [1, 2, 3], [2, 4, 6]])


def test_requires_a_valid_gyroscope_stream(tmp_path):
    meta = _meta(tmp_path)
    (tmp_path / "imu.json").write_text(
        json.dumps([{"timestamp_ns": 1, "accel": [0, 0, 9.8]}])
    )
    with pytest.raises(ValueError, match="gyroscope"):
        sync_imu(tmp_path, meta)
