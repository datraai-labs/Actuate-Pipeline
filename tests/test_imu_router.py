"""
DatraAI Pipeline — Tests for utils/imu_source_router.py and the
vision-derived wrist-rotation helpers in utils/video_utils.py (v2 addendum §1).
All synthetic data — no real video/API calls.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.imu_source_router import (
    ALL_PRIMITIVES,
    FusionStrategy,
    VisionPrimaryStrategy,
    WristPrimaryStrategy,
    check_imu_mount_plausibility,
    detect_contact_onset,
    detect_wrist_pronate,
    get_primitive_strategy,
)
from utils.video_utils import (
    compute_wrist_flexion_from_landmarks,
    compute_wrist_rotation_from_landmarks,
)


def _blank_landmarks():
    return [[0.0, 0.0, 0.0] for _ in range(21)]


def _pose_frame(wrist_mcp_vector, dominant_hand="right", hands_detected=True, derived=None):
    """Build a hand_pose.json-style frame with a given wrist(0)->MCP(9) vector."""
    landmarks = _blank_landmarks()
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[9] = list(wrist_mcp_vector)
    return {
        "hands_detected": hands_detected,
        "dominant_hand": dominant_hand if hands_detected else None,
        f"{dominant_hand}_hand": {"landmarks": landmarks, "confidence": 0.9} if hands_detected else None,
        "derived": derived,
    }


class TestComputeWristRotationFromLandmarks:
    """Vision-derived pronation/supination proxy."""

    def test_known_rotation_rate(self):
        fps = 30.0
        prev = _pose_frame([1.0, 0.0, 0.0])
        # Rotate the wrist->MCP vector by +30 degrees in the xy-plane.
        theta = np.radians(30.0)
        curr = _pose_frame([np.cos(theta), np.sin(theta), 0.0])

        result = compute_wrist_rotation_from_landmarks(curr, prev, fps)

        expected = 30.0 / (1.0 / fps)  # deg/s
        assert result is not None
        assert abs(result - expected) < 1.0

    def test_negative_rotation_is_supination_direction(self):
        fps = 30.0
        prev = _pose_frame([1.0, 0.0, 0.0])
        theta = np.radians(-30.0)
        curr = _pose_frame([np.cos(theta), np.sin(theta), 0.0])

        result = compute_wrist_rotation_from_landmarks(curr, prev, fps)
        assert result is not None
        assert result < 0

    def test_no_hands_returns_none(self):
        fps = 30.0
        prev = _pose_frame([1.0, 0.0, 0.0])
        curr = _pose_frame([1.0, 0.0, 0.0], hands_detected=False)
        assert compute_wrist_rotation_from_landmarks(curr, prev, fps) is None

    def test_zero_fps_returns_none(self):
        prev = _pose_frame([1.0, 0.0, 0.0])
        curr = _pose_frame([0.0, 1.0, 0.0])
        assert compute_wrist_rotation_from_landmarks(curr, prev, 0.0) is None


class TestComputeWristFlexionFromLandmarks:
    """Vision-derived flexion/extension proxy from landmark z."""

    def test_known_elevation_change(self):
        fps = 30.0
        prev = _pose_frame([1.0, 0.0, 0.0])
        # elevation = atan2(-z, planar); choose z so elevation ~= 30 deg
        z = -np.tan(np.radians(30.0))
        curr = _pose_frame([1.0, 0.0, z])

        result = compute_wrist_flexion_from_landmarks(curr, prev, fps)
        expected = 30.0 / (1.0 / fps)
        assert result is not None
        assert abs(result - expected) < 2.0


class TestWristPrimaryStrategy:
    """Strategy wiring should match the underlying gyro/accel detector functions."""

    def test_pronate_detected_via_gyro(self):
        strategy = WristPrimaryStrategy()
        gyro_z_deg_s = 30.0
        gyro_window = np.zeros((5, 3))
        gyro_window[:, 2] = np.radians(gyro_z_deg_s)
        accel_window = np.zeros((5, 3))

        imu_window = {
            "accel_window": accel_window,
            "gyro_window": gyro_window,
            "dt": 1.0 / 30.0,
            "fps": 30.0,
            "pose_window": [None],
            "contact_onset_history": [],
        }

        result = strategy.detect_primitives(0, None, None, imu_window, None)
        assert result["wrist_pronate"] is True
        assert result["wrist_supinate"] is False
        assert set(result.keys()) == set(ALL_PRIMITIVES)


class TestVisionPrimaryStrategy:
    """Head-mounted-IMU strategy: fine-motor primitives from vision only."""

    def test_idle_from_head_imu_stillness(self):
        strategy = VisionPrimaryStrategy()
        still_accel = np.full((5, 3), 0.01)
        still_gyro = np.zeros((5, 3))

        imu_window = {
            "head_accel_window": still_accel,
            "head_gyro_window": still_gyro,
            "fps": 30.0,
            "pose_window": [None],
        }

        result = strategy.detect_primitives(0, None, None, imu_window, None)
        assert result["idle"] is True
        # The head-mounted-IMU path must produce ONLY an idle/activity
        # signal from IMU data — every hand/wrist-derived primitive must be
        # False here, not merely unchecked. With no pose data, that False
        # comes from lacking landmarks (correct: no vision signal either),
        # not from any IMU-based fallback for these primitives.
        for prim in ALL_PRIMITIVES:
            if prim == "idle":
                continue
            assert result[prim] is False, f"{prim} should be False (got {result[prim]!r}) with no pose data"

    def test_wrist_role_imu_keys_are_ignored_even_when_present(self):
        """
        Regression guard against a future refactor accidentally wiring the
        wrist-role accel_window/gyro_window into VisionPrimaryStrategy.
        Stuff those keys with an extreme signal that WOULD fire
        wrist_pronate/contact_onset under WristPrimaryStrategy, while the
        actual hand landmarks show zero rotation/no grasp change — confirm
        the vision strategy's output is unaffected by that IMU data.
        """
        strategy = VisionPrimaryStrategy()
        pose = _pose_frame([1.0, 0.0, 0.0], derived={"fingertip_dists": [0.3] * 5, "thumb_index_dist": 0.3})

        extreme_gyro = np.zeros((5, 3))
        extreme_gyro[:, 2] = np.radians(500.0)
        assert detect_wrist_pronate(extreme_gyro[:, 2]) is True  # sanity-check the trap is genuine

        # A flat/uniform accel array would NOT trigger detect_contact_onset
        # even under WristPrimaryStrategy (it requires a spike-then-decel
        # shape), which would make that assertion pass vacuously regardless
        # of whether VisionPrimaryStrategy reads this key. Use a genuine
        # spike+deceleration pattern instead, matching
        # TestContactOnset.test_contact_onset_fires_spike_with_decel in
        # test_primitives.py, so the trap is real.
        extreme_accel = np.array([
            [0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5],
            [0.8, 0.8, 0.8],  # peak, mag ~1.39 > CONTACT_ACCEL_G (1.2)
            [0.4, 0.4, 0.4],
            [0.2, 0.2, 0.2],
        ])
        assert detect_contact_onset(extreme_accel) is True  # sanity-check the trap is genuine

        imu_window = {
            "accel_window": extreme_accel,
            "gyro_window": extreme_gyro,
            "head_accel_window": np.full((5, 3), 0.01),  # head itself is still
            "head_gyro_window": np.zeros((5, 3)),
            "fps": 30.0,
            "pose_window": [pose, pose],
            "contact_onset_history": [],
        }

        result = strategy.detect_primitives(1, pose, pose, imu_window, None)

        assert result["wrist_pronate"] is False
        assert result["wrist_supinate"] is False
        assert result["wrist_flex"] is False
        assert result["contact_onset"] is False
        # idle should still be driven by the (still) head-role keys, proving
        # those two roles are read independently rather than one clobbering
        # the other.
        assert result["idle"] is True

    def test_not_idle_with_head_motion(self):
        strategy = VisionPrimaryStrategy()
        moving_accel = np.full((5, 3), 1.0)
        moving_gyro = np.zeros((5, 3))

        imu_window = {
            "head_accel_window": moving_accel,
            "head_gyro_window": moving_gyro,
            "fps": 30.0,
            "pose_window": [None],
        }

        result = strategy.detect_primitives(0, None, None, imu_window, None)
        assert result["idle"] is False

    def test_contact_onset_fallback_on_grasp_transition(self):
        """No object tracking available: onset fires on the grasp-shape transition edge."""
        strategy = VisionPrimaryStrategy()
        open_hand = _pose_frame(
            [1.0, 0.0, 0.0],
            derived={"fingertip_dists": [0.3, 0.3, 0.3, 0.3, 0.3], "thumb_index_dist": 0.3},
        )
        closed_hand = _pose_frame(
            [1.0, 0.0, 0.0],
            derived={"fingertip_dists": [0.05] * 5, "thumb_index_dist": 0.3},
        )
        imu_window = {
            "head_accel_window": np.zeros((0, 3)),
            "head_gyro_window": np.zeros((0, 3)),
            "fps": 30.0,
            "pose_window": [open_hand, closed_hand],
        }

        result = strategy.detect_primitives(1, closed_hand, open_hand, imu_window, None)
        assert result["contact_onset"] is True
        assert result["contact_release"] is False

    def test_wrist_pronate_from_vision_rotation(self):
        strategy = VisionPrimaryStrategy()
        prev = _pose_frame([1.0, 0.0, 0.0])
        theta = np.radians(30.0)
        curr = _pose_frame([np.cos(theta), np.sin(theta), 0.0])
        imu_window = {
            "head_accel_window": np.zeros((0, 3)),
            "head_gyro_window": np.zeros((0, 3)),
            "fps": 30.0,
            "pose_window": [prev, curr],
        }

        result = strategy.detect_primitives(1, curr, prev, imu_window, None)
        assert result["wrist_pronate"] is True
        assert result["wrist_supinate"] is False

    def test_marginal_rotation_above_threshold_fires(self):
        """A modest, realistic per-frame rotation should still cross the 15 deg/s threshold."""
        strategy = VisionPrimaryStrategy()
        fps = 30.0
        # 1.0 degree of rotation within one 1/30s frame == 30 deg/s,
        # comfortably above PRONATE_GYRO_Z_DEG_S (15) without needing an
        # extreme multi-degree swing per frame.
        prev = _pose_frame([1.0, 0.0, 0.0])
        theta = np.radians(1.0)
        curr = _pose_frame([np.cos(theta), np.sin(theta), 0.0])
        imu_window = {
            "head_accel_window": np.zeros((0, 3)),
            "head_gyro_window": np.zeros((0, 3)),
            "fps": fps,
            "pose_window": [prev, curr],
        }
        result = strategy.detect_primitives(1, curr, prev, imu_window, None)
        assert result["wrist_pronate"] is True

    def test_sub_threshold_rotation_does_not_fire(self):
        """Rotation below the deg/s threshold must not trigger pronation."""
        strategy = VisionPrimaryStrategy()
        fps = 30.0
        # 0.3 degree of rotation within one 1/30s frame == 9 deg/s, below
        # the 15 deg/s threshold.
        prev = _pose_frame([1.0, 0.0, 0.0])
        theta = np.radians(0.3)
        curr = _pose_frame([np.cos(theta), np.sin(theta), 0.0])
        imu_window = {
            "head_accel_window": np.zeros((0, 3)),
            "head_gyro_window": np.zeros((0, 3)),
            "fps": fps,
            "pose_window": [prev, curr],
        }
        result = strategy.detect_primitives(1, curr, prev, imu_window, None)
        assert result["wrist_pronate"] is False
        assert result["wrist_supinate"] is False


class TestFusionStrategy:
    """Confidence-weighted vote between vision and wrist-IMU derivations."""

    def test_agreement_no_disagreement_flag(self):
        strategy = FusionStrategy()
        # Both sources see no rotation and no motion.
        pose = _pose_frame([1.0, 0.0, 0.0], derived={"fingertip_dists": [0.3] * 5, "thumb_index_dist": 0.3})
        imu_window = {
            "accel_window": np.zeros((5, 3)),
            "gyro_window": np.zeros((5, 3)),
            "head_accel_window": np.zeros((5, 3)),
            "head_gyro_window": np.zeros((5, 3)),
            "dt": 1.0 / 30.0,
            "fps": 30.0,
            "pose_window": [pose, pose],
            "contact_onset_history": [],
        }

        result = strategy.detect_primitives(1, pose, pose, imu_window, None)
        assert result["disagreement"] is False
        assert result["wrist_pronate"] is False

    def test_disagreement_flagged_when_sources_conflict(self):
        strategy = FusionStrategy()
        # Wrist IMU sees a strong pronation spike...
        gyro_window = np.zeros((5, 3))
        gyro_window[:, 2] = np.radians(30.0)
        # ...but vision sees no wrist rotation at all (identical landmarks).
        pose = _pose_frame([1.0, 0.0, 0.0], derived={"fingertip_dists": [0.3] * 5, "thumb_index_dist": 0.3})

        imu_window = {
            "accel_window": np.zeros((5, 3)),
            "gyro_window": gyro_window,
            "head_accel_window": np.zeros((5, 3)),
            "head_gyro_window": np.zeros((5, 3)),
            "dt": 1.0 / 30.0,
            "fps": 30.0,
            "pose_window": [pose, pose],
            "contact_onset_history": [],
        }

        result = strategy.detect_primitives(1, pose, pose, imu_window, None)
        # imu-only signal (weight 0.3) shouldn't outvote vision (weight 0.7, sees nothing)
        assert result["wrist_pronate"] is False
        assert result["disagreement"] is True


class TestGetPrimitiveStrategy:
    """Factory should map config modes to the documented strategy classes."""

    def test_head_mounted_maps_to_vision(self):
        assert isinstance(get_primitive_strategy("head_mounted"), VisionPrimaryStrategy)

    def test_none_maps_to_vision(self):
        assert isinstance(get_primitive_strategy("none"), VisionPrimaryStrategy)

    def test_wrist_mounted_maps_to_wrist(self):
        assert isinstance(get_primitive_strategy("wrist_mounted"), WristPrimaryStrategy)

    def test_dual_maps_to_fusion(self):
        assert isinstance(get_primitive_strategy("dual"), FusionStrategy)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            get_primitive_strategy("bogus_mode")


class TestCheckImuMountPlausibility:
    """
    Advisory heuristic: does the single ingested IMU stream's gross motion
    look plausible for the declared IMU_SOURCE_MODE mount location? Never
    blocks the pipeline — only flags for human review.
    """

    def test_plausible_head_mounted_smooth_motion(self):
        gyro = np.radians(np.full((100, 3), 2.0))  # smooth, low-magnitude head motion
        accel = np.zeros((100, 3))
        result = check_imu_mount_plausibility("head_mounted", gyro, accel)
        assert result["checked"] is True
        assert result["plausible"] is True
        assert result["reason"] is None

    def test_implausible_head_mounted_wrist_like_spikes(self):
        gyro = np.zeros((100, 3))
        # 5% of frames show a wrist-snap-like spike (300 deg/s), far above
        # what's plausible for whole-head rotation.
        gyro[:5, 2] = np.radians(300.0)
        accel = np.zeros((100, 3))
        result = check_imu_mount_plausibility("head_mounted", gyro, accel)
        assert result["checked"] is True
        assert result["plausible"] is False
        assert "wrist" in result["reason"].lower()
        assert result["spike_fraction"] == pytest.approx(0.05)

    def test_plausible_wrist_mounted_with_real_motion(self):
        gyro = np.zeros((100, 3))
        gyro[:10, 2] = np.radians(200.0)  # plenty of real wrist-snap-like motion
        accel = np.zeros((100, 3))
        result = check_imu_mount_plausibility("wrist_mounted", gyro, accel)
        assert result["checked"] is True
        assert result["plausible"] is True

    def test_implausible_wrist_mounted_implausibly_still(self):
        # A wrist IMU during active manipulation should never be this still
        # across an entire session.
        gyro = np.radians(np.full((100, 3), 0.1))
        accel = np.zeros((100, 3))
        result = check_imu_mount_plausibility("wrist_mounted", gyro, accel)
        assert result["checked"] is True
        assert result["plausible"] is False
        assert "head" in result["reason"].lower()

    def test_none_mode_skips_check(self):
        gyro = np.zeros((100, 3))
        gyro[:5, 2] = np.radians(300.0)  # would be flagged if this were "head_mounted"
        accel = np.zeros((100, 3))
        result = check_imu_mount_plausibility("none", gyro, accel)
        assert result["checked"] is False
        assert result["plausible"] is True

    def test_empty_gyro_array_is_safe(self):
        result = check_imu_mount_plausibility("head_mounted", np.zeros((0, 3)), np.zeros((0, 3)))
        assert result["checked"] is False
        assert result["plausible"] is True

    def test_disabled_via_config(self):
        original = cfg.IMU_MOUNT_PLAUSIBILITY_CHECK_ENABLED
        cfg.IMU_MOUNT_PLAUSIBILITY_CHECK_ENABLED = False
        try:
            gyro = np.zeros((100, 3))
            gyro[:5, 2] = np.radians(300.0)
            result = check_imu_mount_plausibility("head_mounted", gyro, np.zeros((100, 3)))
            assert result["checked"] is False
            assert result["plausible"] is True
        finally:
            cfg.IMU_MOUNT_PLAUSIBILITY_CHECK_ENABLED = original
