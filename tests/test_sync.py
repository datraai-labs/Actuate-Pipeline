"""
DatraAI Pipeline — Tests for Step 02: Sync.

Rewritten in the 2026-08-01 audit (Group 3): the previous version re-implemented the
drift/interpolation math inside the test bodies and never imported scripts/02_sync.py —
it passed with the production file deleted. Every test here executes PRODUCTION code:
scripts/02_sync.run(), utils/video_utils.extract_pts, or utils/hdf5_writer round-trips.
The load-bearing tests were confirmed to FAIL against mutated production code
(fabrication flags disabled; clean-metric gating reverted; monotonicity assert removed).
"""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils import video_utils
from utils.hdf5_writer import read_session_h5, write_session_h5

sync_mod = importlib.import_module("scripts.02_sync")

_EPOCH0_MS = 1_000_000.0


# ─── session builders (inputs only — all math is done by 02_sync.run) ────────


def _build_session(tmp_path, monkeypatch, t_ms, pts, meta_extra=None, with_mag=False):
    """Write pts.npy / imu_raw.npy / session_meta.json exactly as 01_ingest would."""
    proc_root = tmp_path / "processed"
    proc = proc_root / "sess"
    proc.mkdir(parents=True, exist_ok=True)

    n_cols = 11 if with_mag else 7
    imu = np.zeros((len(t_ms), n_cols))
    imu[:, 0] = _EPOCH0_MS + np.asarray(t_ms, dtype=np.float64)
    imu[:, 1:4] = 9.8
    # gyro-x is a linear ramp in time so interpolation correctness is checkable
    imu[:, 4] = np.asarray(t_ms, dtype=np.float64) / 1000.0
    imu[:, 5:7] = 0.1
    if with_mag:
        imu[:, 7:10] = 42.0
        imu[:, 10] = 40.0
    np.save(str(proc / "imu_raw.npy"), imu)

    np.save(str(proc / "pts.npy"), np.asarray(pts, dtype=np.float64))
    meta = {"session_id": "sess", "recording_start_epoch_ms": _EPOCH0_MS}
    meta.update(meta_extra or {})
    (proc / "session_meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(cfg, "PROCESSED_DIR", proc_root)
    return proc


def _build_gap_session(tmp_path, monkeypatch, n_frames=10):
    """10 ms IMU spacing over 0–990 ms with 300–400 ms missing (a 120 ms gap, 12x
    median). Frames land every 100 ms — frames 3 and 4 fall inside the gap; every
    other in-range frame sits exactly on a sample."""
    t_ms = np.array([t * 10.0 for t in range(100) if not (29 < t < 41)])
    proc = _build_session(tmp_path, monkeypatch, t_ms, np.arange(n_frames) * 0.1)
    return proc, t_ms


# ─── PTS extraction (production utils/video_utils.extract_pts) ───────────────


class TestExtractPTS:
    """Executes the real extract_pts with ffprobe's subprocess call mocked, so the
    production parsing + monotonicity assertion run without ffmpeg installed."""

    def _mock_ffprobe(self, monkeypatch, stdout):
        monkeypatch.setattr(video_utils, "_check_ffmpeg", lambda: None)
        monkeypatch.setattr(
            video_utils.subprocess, "run",
            lambda *args, **kwargs: SimpleNamespace(stdout=stdout),
        )

    def _video(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00" * 64)
        return video

    def test_parses_and_normalizes_to_zero_start(self, tmp_path, monkeypatch):
        self._mock_ffprobe(monkeypatch, "5.000000\n5.033333\n5.066667\n")
        pts = video_utils.extract_pts(self._video(tmp_path))
        assert pts[0] == 0.0
        assert len(pts) == 3
        assert pts[1] == pytest.approx(0.033333)

    def test_non_monotonic_pts_raises(self, tmp_path, monkeypatch):
        self._mock_ffprobe(monkeypatch, "0.0\n0.033\n0.020\n0.066\n")
        with pytest.raises(ValueError, match="not monotonically increasing"):
            video_utils.extract_pts(self._video(tmp_path))

    def test_duplicate_pts_raises(self, tmp_path, monkeypatch):
        self._mock_ffprobe(monkeypatch, "0.0\n0.033\n0.033\n")
        with pytest.raises(ValueError, match="not monotonically increasing"):
            video_utils.extract_pts(self._video(tmp_path))

    def test_empty_output_raises(self, tmp_path, monkeypatch):
        self._mock_ffprobe(monkeypatch, "\n")
        with pytest.raises(ValueError, match="No valid PTS"):
            video_utils.extract_pts(self._video(tmp_path))


# ─── drift metrics (production 02_sync.run) ──────────────────────────────────


class TestRunDriftMetrics:
    def test_on_grid_session_reports_near_zero_drift(self, tmp_path, monkeypatch):
        # 10 ms IMU spacing, frames every 100 ms exactly on samples
        _build_session(tmp_path, monkeypatch, np.arange(100) * 10.0, np.arange(10) * 0.1)

        stats = sync_mod.run("sess")

        assert stats["max_drift_clean_ms"] == pytest.approx(0.0, abs=1e-6)
        assert stats["drift_warning"] is False

    def test_constant_offset_beyond_threshold_is_detected(self, tmp_path, monkeypatch):
        # frames shifted 5 ms into the middle of every 10 ms sample gap
        _build_session(
            tmp_path, monkeypatch, np.arange(100) * 10.0, np.arange(10) * 0.1 + 0.005
        )

        stats = sync_mod.run("sess")

        assert stats["max_drift_clean_ms"] == pytest.approx(5.0)
        assert stats["drift_warning"] is True


class TestRunInterpolation:
    def test_shapes_values_and_h5_round_trip(self, tmp_path, monkeypatch):
        proc = _build_session(
            tmp_path, monkeypatch, np.arange(100) * 10.0, np.arange(10) * 0.1 + 0.005
        )

        sync_mod.run("sess")
        data = read_session_h5(proc / "session.h5")

        assert data["accel"].shape == (10, 3)
        assert data["gyro"].shape == (10, 3)
        # gyro-x ramps linearly with time (seconds since imu t0); a frame at
        # t0+105 ms must interpolate to 0.105 — executed by run(), not the test
        assert data["gyro"][1, 0] == pytest.approx(0.105, abs=1e-6)
        assert len(data["video_timestamps"]) == 10
        assert data["metadata"]["sync_stats"]["interpolation_method"] == "linear"


class TestOptionalChannels:
    def test_mag_and_temp_survive_to_h5_when_measured(self, tmp_path, monkeypatch):
        """audit Group 4: the legacy path used to coerce IMU to 7 columns, silently
        dropping magnetometer + temperature that the modern path keeps."""
        proc = _build_session(
            tmp_path, monkeypatch, np.arange(100) * 10.0, np.arange(10) * 0.1,
            with_mag=True,
        )

        sync_mod.run("sess")
        data = read_session_h5(proc / "session.h5")

        assert data["imu_mag"].shape == (10, 3)
        assert data["imu_mag"][0, 0] == pytest.approx(42.0)
        assert data["imu_temp_c"].shape == (10,)

    def test_unmeasured_channels_stay_absent_not_zero(self, tmp_path, monkeypatch):
        """A 7-column (pre-audit) imu_raw.npy has no mag/temp — the h5 must omit the
        datasets entirely. Absent means NOT MEASURED; zeros would claim otherwise."""
        proc = _build_session(
            tmp_path, monkeypatch, np.arange(100) * 10.0, np.arange(10) * 0.1
        )

        sync_mod.run("sess")
        data = read_session_h5(proc / "session.h5")

        assert "imu_mag" not in data
        assert "imu_temp_c" not in data


class TestLegacyModernH5Coexistence:
    """audit Group 4: write_session_h5 opened 'w' (truncate) while the modern
    actuate.ingest.imu writer opens 'a' with different key names — running legacy
    after modern DESTROYED the modern IMU group. Each writer now deletes and
    recreates only its own datasets. Confirmed RED against the 'w' writer."""

    def test_legacy_write_preserves_modern_imu_datasets(self, tmp_path):
        from actuate.ingest.imu import sync_imu

        session = tmp_path / "sess"
        session.mkdir()
        meta = {"session_id": "sess", "frame_count": 3, "fps_nominal": 10.0}
        (session / "session_meta.json").write_text(json.dumps(meta))
        (session / "imu.json").write_text(json.dumps([
            {"timestamp_ns": 1_000_000_000, "gyro": [0, 0, 0], "accel": [0, 0, 9.8],
             "mag": [1, 2, 3], "temp_c": 40.0},
            {"timestamp_ns": 1_200_000_000, "gyro": [2, 4, 6], "accel": [0, 0, 9.8],
             "mag": [1, 2, 3], "temp_c": 40.0},
        ]))
        result = sync_imu(session, meta)
        assert result is not None

        # legacy sync now writes to the SAME session.h5
        write_session_h5(
            path=session / "session.h5",
            video_timestamps_abs=np.arange(3) / 10.0,
            pts_relative=np.arange(3) / 10.0,
            accel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.ones((3, 3), dtype=np.float32),
            metadata_dict={"session_id": "sess"},
        )

        with h5py.File(session / "session.h5", "r") as f:
            # modern-only evidence keys survived the legacy write
            assert "timestamp_ns" in f["imu"]
            assert "timestamp_ns_raw" in f["imu"]
            assert int(f["imu"].attrs["schema_version"]) >= 3
            # legacy keys present too
            assert "timestamps" in f["imu"]
            assert "pts_relative" in f["video"]
            # shared-contract keys (accel/gyro: frame-aligned IMU) are last-writer-
            # wins by design — here the legacy values
            assert np.allclose(f["imu/gyro"][:], 1.0)

    def test_modern_write_preserves_legacy_video_group(self, tmp_path):
        from actuate.ingest.imu import sync_imu

        session = tmp_path / "sess"
        session.mkdir()
        write_session_h5(
            path=session / "session.h5",
            video_timestamps_abs=np.arange(3) / 10.0,
            pts_relative=np.arange(3) / 10.0,
            accel=np.zeros((3, 3), dtype=np.float32),
            gyro=np.ones((3, 3), dtype=np.float32),
            metadata_dict={"session_id": "sess"},
        )
        meta = {"session_id": "sess", "frame_count": 3, "fps_nominal": 10.0}
        (session / "session_meta.json").write_text(json.dumps(meta))
        (session / "imu.json").write_text(json.dumps([
            {"timestamp_ns": 1_000_000_000, "gyro": [0, 0, 0], "accel": [0, 0, 9.8]},
            {"timestamp_ns": 1_200_000_000, "gyro": [2, 4, 6], "accel": [0, 0, 9.8]},
        ]))

        sync_imu(session, meta)

        with h5py.File(session / "session.h5", "r") as f:
            assert "pts_relative" in f["video"]
            assert "timestamps" in f["imu"]
            assert "timestamp_ns" in f["imu"]


class TestSyncDriftHDF5Metadata:
    def test_drift_warning_round_trips_through_metadata(self, tmp_path):
        h5_path = tmp_path / "test_session.h5"
        write_session_h5(
            path=h5_path,
            video_timestamps_abs=np.arange(10) / 30.0,
            pts_relative=np.arange(10) / 30.0,
            accel=np.zeros((10, 3), dtype=np.float32),
            gyro=np.zeros((10, 3), dtype=np.float32),
            metadata_dict={"sync_stats": {"max_drift_ms": 7.5, "drift_warning": True}},
        )
        data = read_session_h5(h5_path)
        assert data["metadata"]["sync_stats"]["drift_warning"] is True
        assert data["metadata"]["sync_stats"]["max_drift_ms"] == 7.5


# ─── fabrication accounting (audit Group 1) ──────────────────────────────────


class TestRunFabricationAccounting:
    def test_dropout_frames_flagged_and_raw_axis_preserved(self, tmp_path, monkeypatch):
        proc, t_ms = _build_gap_session(tmp_path, monkeypatch)

        stats = sync_mod.run("sess")

        assert stats["frames_interpolated_over_dropout"] == 2
        assert stats["frames_outside_imu_range"] == 0
        assert stats["raw_median_dt_ms"] == pytest.approx(10.0)
        assert stats["raw_max_gap_ms"] == pytest.approx(120.0)
        assert stats["raw_missing_sample_estimate"] == 11

        data = read_session_h5(proc / "session.h5")
        flags = data["imu_interpolated_over_dropout"]
        assert list(np.nonzero(flags)[0]) == [3, 4]
        # the original sensor clock survives sync, exactly as recorded
        assert len(data["imu_timestamps_raw"]) == len(t_ms)
        assert np.allclose(data["imu_timestamps_raw"], (_EPOCH0_MS + t_ms) / 1000.0)

    def test_frames_beyond_imu_range_flagged_separately(self, tmp_path, monkeypatch):
        proc, _ = _build_gap_session(tmp_path, monkeypatch, n_frames=12)

        stats = sync_mod.run("sess")

        # frames 10 (1.0 s) and 11 (1.1 s) lie past the last raw sample (0.99 s):
        # np.interp clamps them to the edge value — held, not measured.
        assert stats["frames_outside_imu_range"] == 2
        assert stats["frames_interpolated_over_dropout"] == 2

        data = read_session_h5(proc / "session.h5")
        assert list(np.nonzero(data["imu_outside_imu_range"])[0]) == [10, 11]


# ─── drift-vs-jitter separation + anchor honesty (audit Group 2) ─────────────


class TestDriftMetricSeparation:
    """max_drift_ms on a dropout-y stream measures transport jitter (it is bounded
    by the gap structure), not clock alignment. The clean metric excludes
    fabricated frames and is what gets gated."""

    def test_clean_drift_excludes_dropout_spanning_frames(self, tmp_path, monkeypatch):
        _build_gap_session(tmp_path, monkeypatch)

        stats = sync_mod.run("sess")

        # frames 3/4 sit 10 ms from their nearest sample inside the 120 ms gap —
        # that 10 ms is the gap's geometry, not misalignment. Every healthy frame
        # sits exactly on a sample.
        assert stats["max_drift_ms"] == pytest.approx(10.0)
        assert stats["max_drift_clean_ms"] == pytest.approx(0.0, abs=1e-6)
        assert stats["drift_warning"] is False  # gated on the CLEAN metric

    def test_declared_camera_latency_is_applied_and_recorded(self, tmp_path, monkeypatch):
        proc, _ = _build_gap_session(tmp_path, monkeypatch)
        meta = json.loads((proc / "session_meta.json").read_text())
        meta["timestamp_semantics"] = {"camera_to_imu_latency_ms": 5.0}
        (proc / "session_meta.json").write_text(json.dumps(meta))

        stats = sync_mod.run("sess")

        # shifting the video axis +5 ms puts every healthy frame exactly mid-gap
        # between 10 ms-spaced samples -> clean drift becomes 5 ms.
        assert stats["camera_to_imu_latency_ms_applied"] == 5.0
        assert stats["max_drift_clean_ms"] == pytest.approx(5.0)

    def test_undeclared_latency_records_the_zero_assumption(self, tmp_path, monkeypatch):
        _build_gap_session(tmp_path, monkeypatch)

        stats = sync_mod.run("sess")

        assert stats["camera_to_imu_latency_ms_applied"] is None
        assert "assumed 0" in stats["camera_to_imu_latency_note"]

    def test_fallback_anchor_marked_unvalidated_in_sync_stats(self, tmp_path, monkeypatch):
        """The meta written by _build_session predates the anchor-validation
        fields — 02_sync must treat that as UNVALIDATED, never as clean."""
        _build_gap_session(tmp_path, monkeypatch)

        stats = sync_mod.run("sess")

        assert stats["temporal_alignment_validated"] is False
        assert "UNVALIDATED" in stats["temporal_alignment_note"]

    def test_validated_anchor_fields_propagate_from_meta(self, tmp_path, monkeypatch):
        _build_session(
            tmp_path, monkeypatch, np.arange(100) * 10.0, np.arange(10) * 0.1,
            meta_extra={
                "temporal_alignment_validated": True,
                "temporal_alignment_note": "anchor from container creation_time; cross-validated",
                "imu_clock_domain": "epoch",
            },
        )

        stats = sync_mod.run("sess")

        assert stats["temporal_alignment_validated"] is True
        assert stats["imu_clock_domain"] == "epoch"
