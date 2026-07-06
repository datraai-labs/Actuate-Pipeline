"""
DatraAI Pipeline — Tests for Step 00: Per-Worker Calibration (v2 addendum §5)
Tests _compute_worker_thresholds' pure derivation logic and run()'s
consent-gated orchestration. No MediaPipe/video dependency — synthetic
hand_pose.json-shaped fixtures only (see 00_calibration.py's module
docstring for what remains unverified against real footage).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.worker_profile_store import profile_path

_spec = importlib.util.spec_from_file_location(
    "calibration",
    str(Path(__file__).resolve().parent.parent / "scripts" / "00_calibration.py"),
)
_cal_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal_mod)

_compute_worker_thresholds = _cal_mod._compute_worker_thresholds
run = _cal_mod.run


def _frame(idx, fingertip_dists=None, thumb_index_dist=None, hands_detected=True):
    derived = None
    if hands_detected:
        derived = {}
        if fingertip_dists is not None:
            derived["fingertip_dists"] = fingertip_dists
        if thumb_index_dist is not None:
            derived["thumb_index_dist"] = thumb_index_dist
    return {"frame_idx": idx, "hands_detected": hands_detected, "derived": derived}


def _full_calibration_clip(open_dist=0.30, closed_dist=0.02, pinch_open=0.15, pinch_closed=0.01, n_each=15):
    """A clean synthetic clip: n_each frames fully open, n_each fully closed, n_each pinching."""
    frames = []
    idx = 0
    for _ in range(n_each):
        frames.append(_frame(idx, fingertip_dists=[open_dist] * 5, thumb_index_dist=pinch_open))
        idx += 1
    for _ in range(n_each):
        frames.append(_frame(idx, fingertip_dists=[closed_dist] * 5, thumb_index_dist=pinch_open))
        idx += 1
    for _ in range(n_each):
        frames.append(_frame(idx, fingertip_dists=[open_dist] * 5, thumb_index_dist=pinch_closed))
        idx += 1
    return frames


class TestComputeWorkerThresholds:
    def test_midpoint_between_open_and_closed(self):
        frames = _full_calibration_clip(open_dist=0.30, closed_dist=0.02)
        result = _compute_worker_thresholds(frames)
        assert result["power_grasp_dist"] == pytest.approx((0.30 + 0.02) / 2)
        assert result["calibration_open_dist_max"] == pytest.approx(0.30)
        assert result["calibration_closed_dist_min"] == pytest.approx(0.02)

    def test_lateral_pinch_midpoint(self):
        frames = _full_calibration_clip(pinch_open=0.15, pinch_closed=0.01)
        result = _compute_worker_thresholds(frames)
        assert result["lateral_pinch_dist"] == pytest.approx((0.15 + 0.01) / 2)

    def test_different_workers_get_different_thresholds(self):
        """The whole point of §5: two workers with genuinely different hand geometry must get genuinely different personalized thresholds, not a constant."""
        small_hand = _full_calibration_clip(open_dist=0.22, closed_dist=0.015)
        large_hand = _full_calibration_clip(open_dist=0.38, closed_dist=0.03)
        small_result = _compute_worker_thresholds(small_hand)
        large_result = _compute_worker_thresholds(large_hand)
        assert small_result["power_grasp_dist"] != large_result["power_grasp_dist"]
        assert small_result["power_grasp_dist"] < large_result["power_grasp_dist"]

    def test_ignores_frames_without_hands_detected(self):
        frames = _full_calibration_clip()
        frames.append(_frame(999, hands_detected=False))
        result = _compute_worker_thresholds(frames)
        # Should not raise, and the extra no-hands frame shouldn't skew the result.
        assert result["n_fingertip_samples"] == 45 * 5  # 45 detected frames × 5 fingertips each

    def test_too_few_frames_raises(self):
        frames = _full_calibration_clip(n_each=2)  # well under CALIBRATION_MIN_OPEN/CLOSED_FRAMES
        with pytest.raises(ValueError):
            _compute_worker_thresholds(frames)

    def test_too_few_pinch_samples_raises(self):
        """Enough fingertip samples but not enough distinct pinch frames — must still fail closed, not derive from partial data."""
        frames = []
        idx = 0
        for _ in range(30):
            frames.append(_frame(idx, fingertip_dists=[0.3] * 5, thumb_index_dist=None))
            idx += 1
        # Only a couple of frames actually carry thumb_index_dist.
        for _ in range(2):
            frames.append(_frame(idx, fingertip_dists=[0.02] * 5, thumb_index_dist=0.02))
            idx += 1
        with pytest.raises(ValueError):
            _compute_worker_thresholds(frames)


class TestRunConsentGate:
    def test_run_refuses_without_consent_no_profile_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)

        def _fake_extract(video_path):
            return _full_calibration_clip()
        monkeypatch.setattr(_cal_mod, "_extract_calibration_hand_pose", _fake_extract)

        fake_video = tmp_path / "calibration.mp4"
        fake_video.write_bytes(b"not a real video")

        with pytest.raises(PermissionError):
            run("worker_010", fake_video, consent_granted=False)
        assert not profile_path("worker_010").exists()

    def test_run_saves_profile_with_consent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)

        def _fake_extract(video_path):
            return _full_calibration_clip(open_dist=0.28, closed_dist=0.02)
        monkeypatch.setattr(_cal_mod, "_extract_calibration_hand_pose", _fake_extract)

        fake_video = tmp_path / "calibration.mp4"
        fake_video.write_bytes(b"not a real video")

        profile = run("worker_011", fake_video, consent_granted=True)
        assert profile["worker_id"] == "worker_011"
        assert profile["power_grasp_dist"] == pytest.approx((0.28 + 0.02) / 2)
        assert profile_path("worker_011").exists()

    def test_run_missing_video_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            run("worker_012", tmp_path / "does_not_exist.mp4", consent_granted=True)
