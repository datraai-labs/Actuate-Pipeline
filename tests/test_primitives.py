"""
DatraAI Pipeline — Tests for Step 05: Primitives
Tests individual primitive detectors and the temporal smoothing filter with synthetic data.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import write_session_h5

# We need to import the primitive functions.
# Since 05_primitives.py has a numeric prefix, use importlib.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "primitives",
    str(Path(__file__).resolve().parent.parent / "scripts" / "05_primitives.py"),
)
_primitives_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_primitives_mod)

detect_wrist_pronate = _primitives_mod.detect_wrist_pronate
detect_wrist_supinate = _primitives_mod.detect_wrist_supinate
detect_wrist_flex = _primitives_mod.detect_wrist_flex
detect_idle = _primitives_mod.detect_idle
detect_contact_onset = _primitives_mod.detect_contact_onset
detect_power_grasp = _primitives_mod.detect_power_grasp
detect_lateral_pinch = _primitives_mod.detect_lateral_pinch
detect_reach_onset = _primitives_mod.detect_reach_onset
detect_finger_curl = _primitives_mod.detect_finger_curl
detect_finger_extend = _primitives_mod.detect_finger_extend
apply_minimum_duration_filter = _primitives_mod.apply_minimum_duration_filter
run = _primitives_mod.run


class TestWristPronate:
    """Test wrist pronation detection."""

    def test_wrist_pronate_fires_above_threshold(self):
        """Gyro Z > threshold (in rad/s) should detect pronation."""
        # 20 deg/s in radians
        gyro_z = np.full(5, np.radians(20.0))
        assert detect_wrist_pronate(gyro_z) is True

    def test_wrist_pronate_does_not_fire_below_threshold(self):
        """Gyro Z below threshold should not detect pronation."""
        gyro_z = np.full(5, np.radians(5.0))
        assert detect_wrist_pronate(gyro_z) is False

    def test_wrist_pronate_empty_window(self):
        """Empty window should not fire."""
        assert detect_wrist_pronate(np.array([])) is False

    def test_wrist_pronate_boundary_value(self):
        """At exactly the threshold, should not fire (need to exceed)."""
        gyro_z = np.full(5, np.radians(cfg.PRONATE_GYRO_Z_DEG_S))
        # At exactly threshold, mean == threshold, not >, so False
        assert detect_wrist_pronate(gyro_z) is False


class TestWristSupinate:
    """Test wrist supination detection."""

    def test_wrist_supinate_fires(self):
        """Negative gyro Z exceeding threshold should detect supination."""
        gyro_z = np.full(5, np.radians(-20.0))
        assert detect_wrist_supinate(gyro_z) is True

    def test_wrist_supinate_does_not_fire(self):
        """Positive gyro Z should not detect supination."""
        gyro_z = np.full(5, np.radians(5.0))
        assert detect_wrist_supinate(gyro_z) is False


class TestWristPeakDetection:
    """Peak-based detection should catch brief fast snaps a window mean would miss."""

    def test_brief_snap_fires_pronate(self):
        """2 of 5 frames spiking above threshold, rest near zero — mean would miss this."""
        gyro_z = np.radians(np.array([0.0, 0.0, 60.0, 60.0, 0.0]))
        assert detect_wrist_pronate(gyro_z) is True

    def test_single_noise_spike_does_not_fire(self):
        """A single noisy sample above threshold should not trigger a false positive."""
        gyro_z = np.radians(np.array([0.0, 0.0, 60.0, 0.0, 0.0]))
        assert detect_wrist_pronate(gyro_z) is False

    def test_brief_snap_fires_supinate(self):
        gyro_z = np.radians(np.array([0.0, -60.0, -60.0, 0.0, 0.0]))
        assert detect_wrist_supinate(gyro_z) is True

    def test_single_noise_spike_does_not_fire_supinate(self):
        gyro_z = np.radians(np.array([0.0, 0.0, 0.0, -60.0, 0.0]))
        assert detect_wrist_supinate(gyro_z) is False


class TestFingerCurlHandSwitch:
    """A dominant-hand switch between frames must not compare unrelated hands."""

    def _hand(self, curl_amount: float):
        """Landmarks with fingers curled by `curl_amount` (bigger = more curled)."""
        landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
        for mcp, pip, dip in [(5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19), (1, 2, 3)]:
            landmarks[mcp] = [0.0, 0.0, 0.0]
            landmarks[pip] = [0.05, 0.0, 0.0]
            landmarks[dip] = [0.05 + 0.05 * math.cos(curl_amount), 0.05 * math.sin(curl_amount), 0.0]
        return {"landmarks": landmarks}

    def test_no_switch_detects_curl(self):
        """Same dominant hand across frames: real curl should be detected."""
        prev = {
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": self._hand(curl_amount=0.0),
        }
        curr = {
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": self._hand(curl_amount=1.4),
        }
        assert detect_finger_curl(curr, prev) is True

    def test_hand_switch_does_not_compare_unrelated_hands(self):
        """If dominant hand flips between frames, don't fabricate curl from the other hand."""
        prev = {
            "hands_detected": True,
            "dominant_hand": "left",
            "left_hand": self._hand(curl_amount=1.4),
        }
        curr = {
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": self._hand(curl_amount=1.4),
        }
        # No prior data for the "right" hand exists in prev frame, so nothing
        # should be comparable — must not fall back to comparing prev's
        # "left" hand against curr's "right" hand.
        assert detect_finger_curl(curr, prev) is False
        assert detect_finger_extend(curr, prev) is False


class TestIdle:
    """Test idle detection."""

    def test_idle_fires_zero_motion(self):
        """Near-zero accel and velocity should detect idle."""
        accel = np.full((5, 3), 0.01)  # very low accel
        pose_frame = {
            "hands_detected": True,
            "derived": {
                "wrist_velocity_magnitude": 0.01,
            },
        }
        assert detect_idle(accel, pose_frame) is True

    def test_idle_does_not_fire_high_accel(self):
        """High accel magnitude should not be idle."""
        accel = np.full((5, 3), 1.0)  # accel_mag ≈ 1.73
        pose_frame = {
            "hands_detected": True,
            "derived": {
                "wrist_velocity_magnitude": 0.01,
            },
        }
        assert detect_idle(accel, pose_frame) is False

    def test_idle_does_not_fire_high_velocity(self):
        """High wrist velocity should not be idle."""
        accel = np.full((5, 3), 0.01)
        pose_frame = {
            "hands_detected": True,
            "derived": {
                "wrist_velocity_magnitude": 0.5,
            },
        }
        assert detect_idle(accel, pose_frame) is False

    def test_idle_no_pose(self):
        """With no pose data, only check accel."""
        accel = np.full((5, 3), 0.01)
        assert detect_idle(accel, None) is True


class TestContactOnset:
    """Test contact onset detection."""

    def test_contact_onset_fires_spike_with_decel(self):
        """Accel spike > threshold followed by deceleration should fire."""
        # Create accel window with spike and deceleration
        accel = np.zeros((5, 3))
        # Build spike pattern: ramp up then down
        accel[0] = [0.3, 0.3, 0.3]  # baseline
        accel[1] = [0.5, 0.5, 0.5]  # rising
        accel[2] = [0.8, 0.8, 0.8]  # peak — mag ≈ 1.39 > 1.2
        accel[3] = [0.4, 0.4, 0.4]  # decel
        accel[4] = [0.2, 0.2, 0.2]  # baseline

        assert detect_contact_onset(accel) is True

    def test_contact_onset_does_not_fire_low_accel(self):
        """Accel below threshold should not fire."""
        accel = np.full((5, 3), 0.1)  # mag ≈ 0.17
        assert detect_contact_onset(accel) is False

    def test_contact_onset_short_window(self):
        """Too-short window should not fire."""
        accel = np.array([[1.0, 1.0, 1.0]])  # only 1 sample
        assert detect_contact_onset(accel) is False


class TestPowerGrasp:
    """Test power grasp detection."""

    def test_power_grasp_fires_close_fingers(self):
        """All fingertips close to palm should fire."""
        pose = {
            "hands_detected": True,
            "derived": {
                "fingertip_dists": [0.10, 0.11, 0.12, 0.13, 0.14],
            },
        }
        assert detect_power_grasp(pose) is True

    def test_power_grasp_does_not_fire_open_hand(self):
        """Open hand (large distances) should not fire."""
        pose = {
            "hands_detected": True,
            "derived": {
                "fingertip_dists": [0.20, 0.22, 0.25, 0.23, 0.21],
            },
        }
        assert detect_power_grasp(pose) is False

    def test_power_grasp_no_hands(self):
        """No hands detected should not fire."""
        pose = {"hands_detected": False}
        assert detect_power_grasp(pose) is False

    def test_power_grasp_threshold_override_v2_addendum_2_5(self):
        """
        v2 addendum §2/§5 — a glove-adjusted or per-worker-calibrated
        threshold must actually change the detection outcome: fingertip
        distances that fail the bare-hand default should fire once a
        larger (glove-scaled) threshold is passed explicitly.
        """
        pose = {
            "hands_detected": True,
            "derived": {
                "fingertip_dists": [0.16, 0.17, 0.18, 0.19, 0.20],
            },
        }
        assert detect_power_grasp(pose) is False  # fails the bare-hand default (0.15)
        assert detect_power_grasp(pose, threshold=0.25) is True  # passes a glove-scaled threshold
        assert detect_power_grasp(pose, threshold=0.10) is False  # a tighter threshold still correctly rejects it


class TestLateralPinch:
    """Test lateral pinch detection."""

    def test_lateral_pinch_fires(self):
        """Thumb-index distance below threshold should fire."""
        pose = {
            "hands_detected": True,
            "derived": {
                "thumb_index_dist": 0.05,
            },
        }
        assert detect_lateral_pinch(pose) is True

    def test_lateral_pinch_does_not_fire(self):
        """Large thumb-index distance should not fire."""
        pose = {
            "hands_detected": True,
            "derived": {
                "thumb_index_dist": 0.20,
            },
        }
        assert detect_lateral_pinch(pose) is False

    def test_lateral_pinch_threshold_override_v2_addendum_2_5(self):
        pose = {
            "hands_detected": True,
            "derived": {
                "thumb_index_dist": 0.10,
            },
        }
        assert detect_lateral_pinch(pose) is False  # fails the bare-hand default (0.08)
        assert detect_lateral_pinch(pose, threshold=0.12) is True  # passes a glove-scaled threshold


class TestMinimumDurationFilter:
    """Test temporal smoothing filter."""

    def test_short_detection_suppressed(self):
        """2-frame detection below MIN_PRIMITIVE_FRAMES (3) should be suppressed."""
        raw_flags = {
            "test_prim": [False, False, True, True, False, False, False],
        }
        result = apply_minimum_duration_filter(raw_flags, min_frames=3)
        # 2-frame run should be suppressed
        assert result["test_prim"] == [False, False, False, False, False, False, False]

    def test_long_detection_preserved(self):
        """Detection >= MIN_PRIMITIVE_FRAMES should be preserved."""
        raw_flags = {
            "test_prim": [False, True, True, True, True, False, False],
        }
        result = apply_minimum_duration_filter(raw_flags, min_frames=3)
        # 4-frame run should be preserved
        assert result["test_prim"] == [False, True, True, True, True, False, False]

    def test_exact_minimum_duration(self):
        """Detection of exactly MIN_PRIMITIVE_FRAMES should be preserved."""
        raw_flags = {
            "test_prim": [False, True, True, True, False],
        }
        result = apply_minimum_duration_filter(raw_flags, min_frames=3)
        assert result["test_prim"] == [False, True, True, True, False]

    def test_multiple_primitives_independent(self):
        """Filter should apply independently per primitive type."""
        raw_flags = {
            "prim_a": [True, True, False, False],  # 2 frames — suppressed
            "prim_b": [True, True, True, False],    # 3 frames — preserved
        }
        result = apply_minimum_duration_filter(raw_flags, min_frames=3)
        assert result["prim_a"] == [False, False, False, False]
        assert result["prim_b"] == [True, True, True, False]

    def test_all_true(self):
        """All-true flags should be fully preserved."""
        raw_flags = {
            "test_prim": [True, True, True, True, True],
        }
        result = apply_minimum_duration_filter(raw_flags, min_frames=3)
        assert result["test_prim"] == [True, True, True, True, True]

    def test_empty_flags(self):
        """Empty flag list should return empty."""
        raw_flags = {"test_prim": []}
        result = apply_minimum_duration_filter(raw_flags, min_frames=3)
        assert result["test_prim"] == []


class TestRunContactEventExemptFromSustainedSmoothing:
    """
    Regression test for a real bug found via real session_001 data:
    contact_onset/contact_release are instantaneous transition markers (an
    edge, not a sustained state) — under VisionPrimaryStrategy's
    no-object-track fallback, a real event is ALWAYS exactly 1 frame wide,
    so applying the shared MIN_PRIMITIVE_FRAMES=3 filter to them silently
    erased every real contact event, which in turn broke
    bolt_tightening's task-signature score (requires contact_onset>=3).
    config.CONTACT_EVENT_MIN_FRAMES=1 exempts these two primitives from
    that filter; confirm run() actually applies it, and that OTHER
    primitives are still smoothed normally (the exemption is scoped, not a
    blanket smoothing disablement).
    """

    def test_single_frame_contact_onset_survives_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "IMU_SOURCE_MODE", "head_mounted")  # VisionPrimaryStrategy
        session_id = "sess_contact_event_smoothing"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        n = 10
        video_ts = np.arange(n, dtype=np.float64) / 30.0
        write_session_h5(
            proc_dir / "session.h5",
            video_timestamps_abs=video_ts,
            pts_relative=video_ts,
            accel=np.zeros((n, 3), dtype=np.float32),
            gyro=np.zeros((n, 3), dtype=np.float32),
            metadata_dict={},
        )

        # Grasp for exactly ONE frame (frame 4), open otherwise -> a single
        # False->True->False transition -> onset at frame 4, release at
        # frame 5, each exactly 1 frame wide (mirrors the real session_001
        # finding: every genuine raw contact_onset detection was 1 frame).
        pose_data = []
        for i in range(n):
            closed = (i == 4)
            dists = [0.02] * 5 if closed else [0.5] * 5
            pose_data.append({
                "frame_idx": i,
                "hands_detected": True,
                "dominant_hand": "right",
                "right_hand": {"confidence": 0.9, "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)]},
                "derived": {
                    "thumb_index_dist": 0.5,
                    "fingertip_dists": dists,
                    "wrist_velocity_magnitude": 0.0,
                    "wrist_velocity": [0.0, 0.0],
                },
            })
        with open(proc_dir / "hand_pose.json", "w") as f:
            json.dump(pose_data, f)

        result = run(session_id)
        contact_onset_frames = [f["frame_idx"] for f in result if f["raw_flags"]["contact_onset"]]
        assert contact_onset_frames == [4], (
            f"expected the single real contact_onset event at frame 4 to survive "
            f"smoothing, got {contact_onset_frames}"
        )

    def test_other_primitives_still_smoothed_normally(self, tmp_path, monkeypatch):
        """Sanity check: the exemption must be scoped to contact_onset/contact_release, not a blanket disablement."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "IMU_SOURCE_MODE", "head_mounted")
        session_id = "sess_other_prims_still_smoothed"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        n = 10
        video_ts = np.arange(n, dtype=np.float64) / 30.0
        write_session_h5(
            proc_dir / "session.h5",
            video_timestamps_abs=video_ts,
            pts_relative=video_ts,
            accel=np.zeros((n, 3), dtype=np.float32),
            gyro=np.zeros((n, 3), dtype=np.float32),
            metadata_dict={},
        )

        # power_grasp True for exactly 1 frame (frame 4) -> a SUSTAINED-type
        # primitive's short blip must still be suppressed by the default
        # MIN_PRIMITIVE_FRAMES=3 filter.
        pose_data = []
        for i in range(n):
            closed = (i == 4)
            dists = [0.02] * 5 if closed else [0.5] * 5
            pose_data.append({
                "frame_idx": i,
                "hands_detected": True,
                "dominant_hand": "right",
                "right_hand": {"confidence": 0.9, "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)]},
                "derived": {
                    "thumb_index_dist": 0.5,
                    "fingertip_dists": dists,
                    "wrist_velocity_magnitude": 0.0,
                    "wrist_velocity": [0.0, 0.0],
                },
            })
        with open(proc_dir / "hand_pose.json", "w") as f:
            json.dump(pose_data, f)

        result = run(session_id)
        power_grasp_frames = [f["frame_idx"] for f in result if f["raw_flags"]["power_grasp"]]
        assert power_grasp_frames == [], (
            f"expected the 1-frame power_grasp blip to be suppressed by the "
            f"default MIN_PRIMITIVE_FRAMES filter, got {power_grasp_frames}"
        )


class TestRunGloveAndCalibrationThresholdWiring:
    """
    v2 addendum §2/§5 end-to-end wiring: run() must actually read
    glove_type/worker_id from session_meta.json (not just accept the
    parameter in isolated unit tests) and use it to shift real detection
    outcomes.
    """

    def _session_with_borderline_grasp(self, tmp_path, monkeypatch, session_id, session_meta_extra=None):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "IMU_SOURCE_MODE", "head_mounted")  # VisionPrimaryStrategy
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        n = 10
        video_ts = np.arange(n, dtype=np.float64) / 30.0
        write_session_h5(
            proc_dir / "session.h5",
            video_timestamps_abs=video_ts,
            pts_relative=video_ts,
            accel=np.zeros((n, 3), dtype=np.float32),
            gyro=np.zeros((n, 3), dtype=np.float32),
            metadata_dict={},
        )

        # Fingertip distance 0.18 for every frame: fails the bare default
        # (0.15) but would pass a sufficiently glove-scaled or
        # worker-calibrated threshold. Held for the full clip so it
        # survives MIN_PRIMITIVE_FRAMES smoothing either way.
        pose_data = [{
            "frame_idx": i,
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": {"confidence": 0.9, "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)]},
            "derived": {
                "thumb_index_dist": 0.5,
                "fingertip_dists": [0.18] * 5,
                "wrist_velocity_magnitude": 0.0,
                "wrist_velocity": [0.0, 0.0],
            },
        } for i in range(n)]
        with open(proc_dir / "hand_pose.json", "w") as f:
            json.dump(pose_data, f)

        session_meta = {"session_id": session_id}
        if session_meta_extra:
            session_meta.update(session_meta_extra)
        with open(proc_dir / "session_meta.json", "w") as f:
            json.dump(session_meta, f)

        return proc_dir

    def test_bare_hand_default_does_not_fire(self, tmp_path, monkeypatch):
        session_id = "sess_glove_none"
        self._session_with_borderline_grasp(tmp_path, monkeypatch, session_id, {"glove_type": "none"})
        result = run(session_id)
        assert all(not f["raw_flags"]["power_grasp"] for f in result)

    def test_thick_glove_type_fires_the_same_landmarks(self, tmp_path, monkeypatch):
        """0.18 fails bare (0.15) but 0.15*1.35=0.2025 > 0.18 — the glove-scaled threshold should fire."""
        session_id = "sess_glove_thick"
        self._session_with_borderline_grasp(tmp_path, monkeypatch, session_id, {"glove_type": "thick"})
        result = run(session_id)
        assert any(f["raw_flags"]["power_grasp"] for f in result)

    def test_missing_session_meta_falls_back_to_bare_default(self, tmp_path, monkeypatch):
        """No session_meta.json at all (pre-§2/§5 processed dirs) must behave exactly as before — no crash, no glove assumption."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "IMU_SOURCE_MODE", "head_mounted")
        session_id = "sess_no_meta"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)
        n = 10
        video_ts = np.arange(n, dtype=np.float64) / 30.0
        write_session_h5(
            proc_dir / "session.h5", video_timestamps_abs=video_ts, pts_relative=video_ts,
            accel=np.zeros((n, 3), dtype=np.float32), gyro=np.zeros((n, 3), dtype=np.float32),
            metadata_dict={},
        )
        pose_data = [{
            "frame_idx": i, "hands_detected": True, "dominant_hand": "right",
            "right_hand": {"confidence": 0.9, "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)]},
            "derived": {"thumb_index_dist": 0.5, "fingertip_dists": [0.18] * 5,
                        "wrist_velocity_magnitude": 0.0, "wrist_velocity": [0.0, 0.0]},
        } for i in range(n)]
        with open(proc_dir / "hand_pose.json", "w") as f:
            json.dump(pose_data, f)
        # No session_meta.json written at all.

        result = run(session_id)  # must not raise
        assert all(not f["raw_flags"]["power_grasp"] for f in result)

    def test_worker_calibration_profile_overrides_glove_default(self, tmp_path, monkeypatch):
        """A worker's own calibration profile (once loaded) must take priority over the glove-type default."""
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path / "calibration")
        from utils.worker_profile_store import save_profile
        save_profile("worker_099", {"power_grasp_dist": 0.25, "lateral_pinch_dist": 0.05}, consent_granted=True)

        session_id = "sess_worker_calibrated"
        self._session_with_borderline_grasp(
            tmp_path, monkeypatch, session_id,
            {"glove_type": "none", "worker_id": "worker_099"},  # glove says "none" (0.15) but worker profile says 0.25
        )
        result = run(session_id)
        assert any(f["raw_flags"]["power_grasp"] for f in result), (
            "worker_099's calibrated 0.25 threshold should fire on 0.18 fingertip "
            "distances even though glove_type='none' alone would not"
        )
