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


# ── Silent-data-loss guards (audit 2026-08-01, Group 1) ──────────────────────


def _two_sample_stream():
    return [
        {"timestamp_ns": 1_000_000_000, "gyro": [0, 0, 0], "accel": [0, 0, 9.8]},
        {"timestamp_ns": 1_200_000_000, "gyro": [2, 4, 6], "accel": [0, 0, 9.8]},
    ]


def test_multiple_imu_sidecars_fail_loudly_naming_every_file(tmp_path):
    """A session must NEVER ingest cleanly while a sensor stream is discarded.

    The old _find_source returned files[0]: a two-IMU session ingested one stream
    and silently dropped the other, producing a well-formed, confident, wrong
    session.h5.
    """
    meta = _meta(tmp_path, frame_count=3, fps=10.0)
    (tmp_path / "imu.json").write_text(json.dumps(_two_sample_stream()))
    (tmp_path / "imu_wrist.csv").write_text("timestamp_ns,gyro_x,gyro_y,gyro_z\n1,0,0,0\n")

    with pytest.raises(ValueError) as excinfo:
        sync_imu(tmp_path, meta)
    message = str(excinfo.value)
    assert "imu.json" in message
    assert "imu_wrist.csv" in message
    assert (tmp_path / "session.h5").exists() is False  # nothing half-written


def test_explicit_source_selects_among_multiple(tmp_path):
    """source= is the sanctioned way to disambiguate — explicit, never implicit."""
    meta = _meta(tmp_path, frame_count=3, fps=10.0)
    (tmp_path / "imu.json").write_text(json.dumps(_two_sample_stream()))
    (tmp_path / "imu_wrist.csv").write_text("not,even,parsed\n")

    result = sync_imu(tmp_path, meta, source=tmp_path / "imu.json")

    assert result is not None
    assert result.source_path.name == "imu.json"


def test_dropout_frames_are_flagged_and_raw_timestamps_preserved(tmp_path):
    """Frames synthesized across a raw-sample dropout must be distinguishable from
    measured ones at point of use, and the original sensor clock must survive sync."""
    meta = _meta(tmp_path, frame_count=10, fps=10.0)
    # 10 ms sample spacing over 0–990 ms, with 300–400 ms missing: a 120 ms gap
    # (12x median). Frames land every 100 ms; frames 3 (300 ms) and 4 (400 ms)
    # fall inside the gap, every other frame sits exactly on a sample.
    ts = [t * 10_000_000 for t in range(100) if not (29 < t < 41)]
    records = [
        {"timestamp_ns": t, "gyro": [1.0, 2.0, 3.0], "accel": [0, 0, 9.8]} for t in ts
    ]
    (tmp_path / "imu.json").write_text(json.dumps(records))

    result = sync_imu(tmp_path, meta)

    assert result is not None
    with h5py.File(result.session_h5, "r") as h5:
        flags = h5["imu/interpolated_over_dropout"][:]
        assert flags.dtype == np.bool_
        assert list(np.nonzero(flags)[0]) == [3, 4]
        assert not h5["imu/outside_source_range"][:].any()
        # the un-resampled sensor clock, exactly as recorded
        assert list(h5["imu/timestamp_ns_raw"][:]) == ts
        grp = h5["imu"]
        assert grp.attrs["raw_median_dt_ns"] == 10_000_000
        assert grp.attrs["raw_max_gap_ns"] == 120_000_000
        assert grp.attrs["raw_missing_sample_estimate"] == 11


def test_frame_mode_flags_frames_with_zero_samples(tmp_path):
    """nearest_video_frame mode: a frame no sample was assigned to got its value
    from interpolation, and sample_count is the per-frame evidence."""
    meta = _meta(tmp_path, frame_count=4)
    records = []
    for frame in (1, 2, 4, 5):  # frame 3 has NO samples; 5 is the boundary record
        records.append(
            {
                "timestamp_ns": frame * 1_000_000,
                "nearest_video_frame": frame,
                "gyro": [frame, 0, 0],
                "accel": [0, 0, 9.8],
            }
        )
    (tmp_path / "imu.json").write_text(json.dumps(records))

    result = sync_imu(tmp_path, meta)

    assert result is not None
    with h5py.File(result.session_h5, "r") as h5:
        counts = h5["imu/sample_count"][:]
        assert list(counts) == [1, 1, 0, 1]
        flags = h5["imu/interpolated_over_dropout"][:]
        assert list(np.nonzero(flags)[0]) == [2]
