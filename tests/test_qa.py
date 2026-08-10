"""
DatraAI Pipeline — Tests for Step 03: Quality Assurance (audit 2026-08-01, Group 3).

First test coverage this stage has ever had. All four checks execute the PRODUCTION
functions in scripts/03_qa.py. The fps tests encode the two structural bugs the audit
found (first-300-frames blindness; global TARGET_FPS instead of the session's own
cadence) plus the Master Spec §L0 fps-bug gate — they were written RED against the
pre-fix code and confirmed to fail.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

cv2 = pytest.importorskip("cv2")

qa_mod = importlib.import_module("scripts.03_qa")


# ─── video fixtures ──────────────────────────────────────────────────────────


def _write_video(path: Path, frames):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )
    assert writer.isOpened(), "cv2 VideoWriter could not open — codec missing?"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def sharp_video(tmp_path_factory):
    """High-frequency noise -> large Laplacian variance on every sampled frame."""
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(91)]
    return _write_video(tmp_path_factory.mktemp("qa") / "sharp.mp4", frames)


@pytest.fixture(scope="module")
def blurred_video(tmp_path_factory):
    """A flat gray scene -> near-zero Laplacian variance."""
    frames = [np.full((64, 64, 3), 127, dtype=np.uint8) for _ in range(91)]
    return _write_video(tmp_path_factory.mktemp("qa") / "flat.mp4", frames)


@pytest.fixture(scope="module")
def moving_video(tmp_path_factory):
    """A pattern translating between sampled frames -> real optical flow."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    frames = []
    for i in range(91):
        shifted = np.roll(base, shift=(i // 30) * 16, axis=1)
        frames.append(cv2.cvtColor(shifted[:, :64], cv2.COLOR_GRAY2BGR))
    return _write_video(tmp_path_factory.mktemp("qa") / "moving.mp4", frames)


# ─── CHECK 1: blur ───────────────────────────────────────────────────────────


class TestBlurCheck:
    def test_sharp_video_passes(self, sharp_video):
        result = qa_mod._check_blur(sharp_video)
        assert result["passed"] is True
        assert result["score"] > cfg.BLUR_THRESHOLD

    def test_flat_video_fails_for_the_right_reason(self, blurred_video):
        result = qa_mod._check_blur(blurred_video)
        assert result["passed"] is False
        assert result["score"] < cfg.BLUR_THRESHOLD
        assert result["details"]["frames_below_threshold_count"] == result["details"]["frames_sampled"]


# ─── CHECK 2: coverage ───────────────────────────────────────────────────────


class TestCoverageCheck:
    def test_moving_camera_passes(self, moving_video):
        result = qa_mod._check_coverage(moving_video)
        assert result["passed"] is True

    def test_static_scene_fails(self, blurred_video):
        result = qa_mod._check_coverage(blurred_video)
        assert result["passed"] is False
        assert result["details"]["static_segment_count"] == result["details"]["pairs_sampled"]


# ─── CHECK 3: fps consistency ────────────────────────────────────────────────


def _h5(timestamps, fps_nominal=None, sync_stats=None, raw_ts=None):
    data = {
        "video_timestamps": np.asarray(timestamps, dtype=np.float64),
        "metadata": {},
    }
    if fps_nominal is not None:
        data["metadata"]["fps_nominal"] = fps_nominal
    if sync_stats is not None:
        data["metadata"]["sync_stats"] = sync_stats
    if raw_ts is not None:
        data["imu_timestamps_raw"] = np.asarray(raw_ts, dtype=np.float64)
    return data


class TestFpsConsistency:
    def test_clean_30fps_session_passes(self):
        result = qa_mod._check_fps_consistency(_h5(np.arange(600) / 30.0, fps_nominal=30.0))
        assert result["passed"] is True
        assert result["details"]["dropped_frame_estimate"] == 0

    def test_drops_after_the_first_ten_seconds_are_detected(self):
        """AUDIT BUG 1: the old check read timestamps[:300] — a session dropping
        frames from t=20s onward scored perfectly. Confirmed RED pre-fix."""
        clean = np.arange(600) / 30.0                     # 0..20 s clean
        dropping = 20.0 + np.arange(150) / 10.0           # then 10 fps: 2/3 dropped
        result = qa_mod._check_fps_consistency(_h5(np.concatenate([clean, dropping]), fps_nominal=30.0))
        assert result["passed"] is False
        assert result["details"]["dropped_frame_estimate"] >= 140

    def test_120fps_rig_scored_against_its_own_cadence(self):
        """AUDIT BUG 2: dropped-frame detection compared against global
        TARGET_FPS=30 — a 120 fps rig (already in the corpus: 87f178b6) dropping
        every other frame produced 16.7 ms intervals, invisible under the 50 ms
        bar. Confirmed RED pre-fix."""
        t = np.arange(1200) / 120.0
        halved = t[::2]                                    # every 2nd frame dropped
        result = qa_mod._check_fps_consistency(_h5(halved, fps_nominal=120.0))
        assert result["details"]["measured_fps"] == pytest.approx(60.0, rel=0.02)
        assert result["details"]["fps_metadata_mismatch"] is True
        assert result["passed"] is False

    def test_master_spec_fps_bug_gate_metadata_not_silently_trusted(self):
        """Master Spec §L0 verification gate: a fixture where source fps != metadata
        fps must be CAUGHT. Absent everywhere pre-fix (confirmed RED)."""
        result = qa_mod._check_fps_consistency(_h5(np.arange(1200) / 120.0, fps_nominal=30.0))
        assert result["passed"] is False
        assert result["details"]["fps_metadata_mismatch"] is True
        assert result["details"]["measured_fps"] == pytest.approx(120.0, rel=0.02)
        assert result["details"]["metadata_fps"] == 30.0

    def test_matching_metadata_at_120fps_passes(self):
        result = qa_mod._check_fps_consistency(_h5(np.arange(1200) / 120.0, fps_nominal=120.0))
        assert result["passed"] is True
        assert result["details"]["fps_metadata_mismatch"] is False


# ─── CHECK 4: sync drift ─────────────────────────────────────────────────────


class TestSyncDriftCheck:
    def test_gates_on_clean_metric_not_transport_jitter(self):
        """The real session's case: raw max_drift 3.94 ms (half a dropout gap) but
        clean 1.05 ms — the session passes, and for the right reason."""
        result = qa_mod._check_sync_drift(_h5(
            [0.0], sync_stats={
                "max_drift_ms": 3.9447, "max_drift_clean_ms": 1.0479,
                "raw_median_dt_ms": 1.7328, "raw_missing_sample_estimate": 1053,
                "temporal_alignment_validated": False,
            },
            raw_ts=np.arange(0, 95.0, 1.7328e-3),
        ))
        assert result["passed"] is True
        assert result["score"] == pytest.approx(1.0479)
        assert result["details"]["gated_on"] == "max_drift_clean_ms"

    def test_legacy_h5_without_clean_metric_gates_on_conflated_number(self):
        result = qa_mod._check_sync_drift(_h5([0.0], sync_stats={"max_drift_ms": 3.9447}))
        assert result["passed"] is False           # 3.94 > 2.0 — fail-closed on old files
        assert "legacy" in result["details"]["gated_on"]

    def test_unhealthy_stream_fails_on_missing_fraction_despite_clean_drift(self):
        """A heavily-dropping sensor can still interpolate to a tiny nearest-sample
        distance; the missing-sample budget catches what drift cannot."""
        raw = np.arange(0, 95.0, 1.7328e-3)
        result = qa_mod._check_sync_drift(_h5(
            [0.0], sync_stats={
                "max_drift_ms": 0.5, "max_drift_clean_ms": 0.5,
                "raw_median_dt_ms": 1.7328,
                "raw_missing_sample_estimate": int(0.10 * len(raw)),   # 10% missing
                "temporal_alignment_validated": False,
            },
            raw_ts=raw,
        ))
        assert result["passed"] is False
        assert result["details"]["stream_health_ok"] is False

    def test_unvalidated_anchor_is_surfaced_not_hidden(self):
        result = qa_mod._check_sync_drift(_h5(
            [0.0], sync_stats={"max_drift_ms": 1.0, "max_drift_clean_ms": 1.0,
                               "temporal_alignment_validated": False}))
        assert result["temporal_alignment_validated"] is False
        assert "UNVALIDATED" in result["details"]["temporal_alignment"]

    def test_missing_sync_stats_fail_closed(self):
        result = qa_mod._check_sync_drift(_h5([0.0]))
        assert result["passed"] is False
