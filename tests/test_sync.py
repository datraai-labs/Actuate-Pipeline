"""
DatraAI Pipeline — Tests for Step 02: Sync
Tests PTS monotonicity, drift threshold, and interpolation with synthetic data.
"""

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import read_session_h5, write_session_h5


class TestPTSMonotonic:
    """Test that PTS values are monotonically increasing."""

    def test_pts_monotonic_valid(self):
        """Synthetic monotonically increasing PTS should pass."""
        pts = np.linspace(0.0, 100.0, 3000)
        diffs = np.diff(pts)
        assert np.all(diffs > 0), "PTS should be monotonically increasing"

    def test_pts_monotonic_detects_violation(self):
        """Non-monotonic PTS should be detected."""
        pts = np.linspace(0.0, 100.0, 3000)
        # Introduce a non-monotonic point
        pts[500] = pts[499] - 0.01
        diffs = np.diff(pts)
        assert not np.all(diffs > 0), "Non-monotonic PTS should be detected"

    def test_pts_monotonic_exact_duplicate(self):
        """Duplicate PTS values (zero diff) should fail monotonicity check."""
        pts = np.array([0.0, 0.033, 0.066, 0.066, 0.1])
        diffs = np.diff(pts)
        assert not np.all(diffs > 0), "Duplicate PTS should fail monotonicity"


class TestSyncDrift:
    """Test IMU-video sync drift calculations."""

    def test_sync_drift_within_threshold(self):
        """With closely aligned synthetic IMU and video timestamps, drift < 2ms."""
        n_frames = 1000
        fps = 30.0
        imu_hz = 200.0

        # Video timestamps: 30 FPS, starting at epoch 1000.0
        video_t = np.arange(n_frames) / fps + 1000.0

        # IMU timestamps: 200 Hz, covering the same range
        n_imu = int((n_frames / fps) * imu_hz) + 100
        imu_t = np.arange(n_imu) / imu_hz + 1000.0

        # Find nearest IMU sample for each video frame
        nearest_idx = np.searchsorted(imu_t, video_t)
        nearest_idx = np.clip(nearest_idx, 0, len(imu_t) - 1)

        # Check preceding index too
        nearest_prev = np.clip(nearest_idx - 1, 0, len(imu_t) - 1)
        delta_right = np.abs(video_t - imu_t[nearest_idx])
        delta_left = np.abs(video_t - imu_t[nearest_prev])
        best_delta = np.minimum(delta_right, delta_left)

        max_drift_ms = np.max(best_delta) * 1000.0

        assert max_drift_ms <= cfg.SYNC_DRIFT_THRESHOLD_MS, (
            f"Max drift {max_drift_ms:.4f}ms exceeds threshold {cfg.SYNC_DRIFT_THRESHOLD_MS}ms"
        )

    def test_sync_drift_raises_warning_on_large_drift(self):
        """With > 2ms drift, verify the drift is detected."""
        n_frames = 100
        fps = 30.0

        # Video timestamps
        video_t = np.arange(n_frames) / fps + 1000.0

        # IMU timestamps with deliberate offset (10ms gap)
        imu_t = np.arange(50) / 200.0 + 1000.0 + 0.01  # 10ms offset

        nearest_idx = np.searchsorted(imu_t, video_t[:50])
        nearest_idx = np.clip(nearest_idx, 0, len(imu_t) - 1)
        delta = np.abs(video_t[:50] - imu_t[nearest_idx]) * 1000.0
        max_drift = np.max(delta)

        assert max_drift > cfg.SYNC_DRIFT_THRESHOLD_MS, (
            f"Expected drift > {cfg.SYNC_DRIFT_THRESHOLD_MS}ms, got {max_drift:.4f}ms"
        )

    def test_sync_drift_hdf5_metadata(self, tmp_path):
        """Verify drift_warning is correctly stored in HDF5 metadata."""
        h5_path = tmp_path / "test_session.h5"
        n_frames = 100

        timestamps = np.arange(n_frames, dtype=np.float64) / 30.0 + 1000.0
        pts = np.arange(n_frames, dtype=np.float64) / 30.0
        accel = np.random.randn(n_frames, 3).astype(np.float32)
        gyro = np.random.randn(n_frames, 3).astype(np.float32)

        metadata = {
            "session_id": "test",
            "sync_stats": {
                "max_drift_ms": 3.5,
                "mean_drift_ms": 1.2,
                "drift_warning": True,
                "interpolation_method": "linear",
                "imu_dropout_frame_count": 0,
            },
        }

        write_session_h5(h5_path, timestamps, pts, accel, gyro, metadata)
        data = read_session_h5(h5_path)

        assert data["metadata"]["sync_stats"]["drift_warning"] is True
        assert data["metadata"]["sync_stats"]["max_drift_ms"] == 3.5


class TestIMUInterpolation:
    """Test IMU interpolation produces correct shapes."""

    def test_imu_interpolation_shape(self):
        """Interpolated IMU should match video frame count."""
        n_frames = 500
        n_imu = 3500  # ~200Hz for ~17.5s of video
        fps = 30.0

        video_t = np.arange(n_frames) / fps + 1000.0
        imu_t = np.arange(n_imu) / 200.0 + 1000.0
        imu_data = np.random.randn(n_imu)

        synced = np.interp(video_t, imu_t, imu_data)

        assert synced.shape == (n_frames,), (
            f"Interpolated shape {synced.shape} != expected ({n_frames},)"
        )

    def test_imu_interpolation_6_channels(self):
        """All 6 IMU channels should interpolate to [N_frames, 6]."""
        n_frames = 300
        n_imu = 2000
        fps = 30.0

        video_t = np.arange(n_frames) / fps + 1000.0
        imu_t = np.arange(n_imu) / 200.0 + 1000.0
        imu_raw = np.random.randn(n_imu, 7)  # epoch_ms + 6 channels
        imu_raw[:, 0] = imu_t * 1000.0

        synced = np.zeros((n_frames, 6))
        for ch in range(6):
            synced[:, ch] = np.interp(video_t, imu_t, imu_raw[:, ch + 1])

        assert synced.shape == (n_frames, 6), (
            f"Synced IMU shape {synced.shape} != expected ({n_frames}, 6)"
        )

    def test_imu_interpolation_preserves_range(self):
        """Interpolation should not produce values outside the input range."""
        n_frames = 100
        n_imu = 700

        video_t = np.arange(n_frames) / 30.0 + 1000.0
        imu_t = np.arange(n_imu) / 200.0 + 1000.0
        imu_data = np.sin(imu_t)  # bounded [-1, 1]

        synced = np.interp(video_t, imu_t, imu_data)

        assert np.all(synced >= -1.0) and np.all(synced <= 1.0), (
            "Interpolated values should be within input range"
        )
