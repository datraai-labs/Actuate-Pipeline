"""
DatraAI Pipeline — Tests for utils/confidence.py (v2 addendum §9)

Core requirement being tested throughout: confidence must actually VARY
meaningfully with how marginal or unambiguous a detection is — a value
sitting near its decision threshold must score visibly lower than the
same boolean outcome reached with a wide margin. A function that just
returns a constant (or near-constant) value passes a naive "field exists"
check but fails every test in this file.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.confidence import (
    _landmark_confidence,
    _margin_confidence,
    _nearest_object,
    _snr_factor,
    confidence_contact_onset_imu,
    confidence_contact_release_imu,
    confidence_contact_vision,
    confidence_finger_curl,
    confidence_finger_extend,
    confidence_idle_head,
    confidence_idle_wrist,
    confidence_lateral_pinch,
    confidence_power_grasp,
    confidence_reach_onset,
    confidence_transport,
    confidence_wrist_flex_imu,
    confidence_wrist_pronate_imu,
    confidence_wrist_rotation_vision,
    confidence_wrist_supinate_imu,
)
from utils.imu_source_router import (
    ALL_PRIMITIVES,
    FusionStrategy,
    VisionPrimaryStrategy,
    WristPrimaryStrategy,
)


def _pose_with_derived(derived: dict, hands_detected=True, dominant_hand="right", confidence=0.9):
    return {
        "hands_detected": hands_detected,
        "dominant_hand": dominant_hand if hands_detected else None,
        f"{dominant_hand}_hand": {"confidence": confidence} if hands_detected else None,
        "derived": derived if hands_detected else None,
    }


def _finger_hand(curl_amount: float):
    """Landmarks with fingers curled by `curl_amount` (bigger = more curled) — mirrors test_primitives.py's fixture."""
    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    for mcp, pip, dip in [(5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19), (1, 2, 3)]:
        landmarks[mcp] = [0.0, 0.0, 0.0]
        landmarks[pip] = [0.05, 0.0, 0.0]
        landmarks[dip] = [0.05 + 0.05 * math.cos(curl_amount), 0.05 * math.sin(curl_amount), 0.0]
    return {"landmarks": landmarks}


def _finger_pose(curl_amount: float, confidence=0.9):
    return {
        "hands_detected": True,
        "dominant_hand": "right",
        "right_hand": {**_finger_hand(curl_amount), "confidence": confidence},
    }


class TestMarginConfidence:
    def test_zero_at_threshold(self):
        assert _margin_confidence(15.0, 15.0) == 0.0

    def test_approaches_one_far_from_threshold(self):
        # margin == threshold magnitude -> full confidence (default scale)
        assert _margin_confidence(30.0, 15.0) == 1.0

    def test_clips_beyond_one(self):
        assert _margin_confidence(1000.0, 15.0) == 1.0

    def test_symmetric_around_threshold(self):
        below = _margin_confidence(10.0, 15.0)
        above = _margin_confidence(20.0, 15.0)
        assert below == pytest.approx(above)

    def test_monotonic_increasing_with_distance(self):
        near = _margin_confidence(16.0, 15.0)
        far = _margin_confidence(25.0, 15.0)
        assert far > near

    def test_zero_threshold_uses_scale_one(self):
        # threshold=0 -> default scale falls back to 1.0, not division by zero
        assert _margin_confidence(0.5, 0.0) == 0.5


class TestSnrFactor:
    def test_floor_on_short_window(self):
        assert _snr_factor(np.array([1.0])) == cfg.IMU_SNR_CONFIDENCE_FLOOR

    def test_floor_on_all_zero_window(self):
        assert _snr_factor(np.zeros(5)) == cfg.IMU_SNR_CONFIDENCE_FLOOR

    def test_clean_signal_scores_high(self):
        """A constant (zero-variance) peak signal should hit the max multiplier."""
        assert _snr_factor(np.full(5, 20.0)) == 1.0

    def test_noisy_signal_scores_lower_than_clean(self):
        clean = np.full(10, 20.0)
        noisy = np.array([20.0, -18.0, 22.0, -19.0, 21.0, -20.0, 19.0, -21.0, 20.0, -18.0])
        assert _snr_factor(noisy) < _snr_factor(clean)


class TestLandmarkConfidence:
    def test_no_hands_is_neutral_not_full(self):
        assert _landmark_confidence({"hands_detected": False}) == 0.5
        assert _landmark_confidence(None) == 0.5

    def test_reads_dominant_hand_confidence(self):
        pose = {"hands_detected": True, "dominant_hand": "left", "left_hand": {"confidence": 0.73}}
        assert _landmark_confidence(pose) == 0.73


class TestNearestObject:
    def test_none_without_tracked_objects(self):
        pose = _pose_with_derived({"thumb_index_dist": 0.05})
        assert _nearest_object(pose, {"tracked_objects": []}) is None

    def test_returns_distance_and_confidence_of_nearest(self):
        landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
        pose = {
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": {"landmarks": landmarks, "confidence": 0.9},
        }
        object_track_frame = {
            "tracked_objects": [
                {"centroid_norm": [0.9, 0.9], "confidence": 0.3},
                {"centroid_norm": [0.51, 0.51], "confidence": 0.8},  # nearer
            ]
        }
        dist, obj_conf = _nearest_object(pose, object_track_frame)
        assert dist < 0.1
        assert obj_conf == 0.8


class TestWristRotationImuConfidence:
    """IMU gyro-derived rotation: a marginal reading must score lower than an unambiguous one."""

    def test_marginal_pronate_scores_lower_than_unambiguous(self):
        threshold = cfg.WRIST_IMU_THRESHOLDS["PRONATE_GYRO_Z_DEG_S"]
        marginal = np.radians(np.full(5, threshold + 1.0))
        unambiguous = np.radians(np.full(5, threshold + 40.0))
        assert confidence_wrist_pronate_imu(marginal) < confidence_wrist_pronate_imu(unambiguous)

    def test_below_threshold_also_scores_by_margin(self):
        """A confidently-FALSE reading far below threshold should score higher than one just below it."""
        threshold = cfg.WRIST_IMU_THRESHOLDS["PRONATE_GYRO_Z_DEG_S"]
        just_below = np.radians(np.full(5, threshold - 1.0))
        far_below = np.radians(np.full(5, 0.0))
        assert confidence_wrist_pronate_imu(far_below) > confidence_wrist_pronate_imu(just_below)

    def test_empty_window_zero_confidence(self):
        assert confidence_wrist_pronate_imu(np.array([])) == 0.0

    def test_supinate_marginal_vs_unambiguous(self):
        threshold = cfg.WRIST_IMU_THRESHOLDS["SUPINATE_GYRO_Z_DEG_S"]
        marginal = np.radians(np.full(5, threshold - 1.0))
        unambiguous = np.radians(np.full(5, threshold - 40.0))
        assert confidence_wrist_supinate_imu(marginal) < confidence_wrist_supinate_imu(unambiguous)

    def test_flex_marginal_vs_unambiguous(self):
        threshold = cfg.WRIST_IMU_THRESHOLDS["FLEX_GYRO_X_DEG_S"]
        marginal = np.radians(np.full(5, threshold + 1.0))
        unambiguous = np.radians(np.full(5, threshold + 40.0))
        assert confidence_wrist_flex_imu(marginal) < confidence_wrist_flex_imu(unambiguous)

    def test_noisy_window_discounts_confidence(self):
        """Same strong peak, but jittery window -> lower confidence than a clean window."""
        threshold = cfg.WRIST_IMU_THRESHOLDS["PRONATE_GYRO_Z_DEG_S"]
        clean = np.radians(np.full(5, threshold + 40.0))
        jittery = np.radians(np.array([threshold + 40.0, -30.0, threshold + 45.0, -25.0, threshold + 42.0]))
        assert confidence_wrist_pronate_imu(jittery) < confidence_wrist_pronate_imu(clean)


class TestContactImuConfidence:
    def test_marginal_spike_scores_lower_than_strong_spike(self):
        marginal = np.zeros((5, 3))
        marginal[2] = [cfg.CONTACT_ACCEL_G / 3 + 0.05, cfg.CONTACT_ACCEL_G / 3 + 0.05, cfg.CONTACT_ACCEL_G / 3 + 0.05]
        strong = np.zeros((5, 3))
        strong[2] = [2.0, 2.0, 2.0]  # mag >> CONTACT_ACCEL_G

        assert confidence_contact_onset_imu(marginal) < confidence_contact_onset_imu(strong)

    def test_short_window_zero(self):
        assert confidence_contact_onset_imu(np.zeros((2, 3))) == 0.0

    def test_release_confidence_varies_with_margin(self):
        near_threshold = np.full((5, 3), 0.6)  # mag ~1.04, close to the 1.0g release threshold
        far_below = np.full((5, 3), 0.01)      # mag ~0.017, clearly released
        assert confidence_contact_release_imu(far_below) > confidence_contact_release_imu(near_threshold)


class TestIdleWristConfidence:
    def test_deep_stillness_scores_higher_than_marginal_stillness(self):
        deep_still_accel = np.full((5, 3), 0.001)
        marginal_accel = np.full((5, 3), 0.08)  # close to the 0.15g idle ceiling (mag ~0.14)
        pose = _pose_with_derived({"wrist_velocity_magnitude": 0.001})

        deep = confidence_idle_wrist(deep_still_accel, pose)
        marginal = confidence_idle_wrist(marginal_accel, pose)
        assert deep > marginal

    def test_empty_window_zero(self):
        assert confidence_idle_wrist(np.zeros((0, 3)), None) == 0.0


class TestVisionRotationConfidence:
    def test_none_value_is_zero(self):
        assert confidence_wrist_rotation_vision(None, 15.0, None) == 0.0

    def test_marginal_vs_unambiguous(self):
        pose = _pose_with_derived({}, confidence=0.9)
        marginal = confidence_wrist_rotation_vision(16.0, 15.0, pose)
        unambiguous = confidence_wrist_rotation_vision(55.0, 15.0, pose)
        assert marginal < unambiguous

    def test_low_landmark_confidence_discounts_score(self):
        strong_tracking = _pose_with_derived({}, confidence=0.95)
        weak_tracking = _pose_with_derived({}, confidence=0.3)
        assert confidence_wrist_rotation_vision(55.0, 15.0, weak_tracking) < confidence_wrist_rotation_vision(
            55.0, 15.0, strong_tracking
        )


class TestPowerGraspConfidence:
    def test_marginal_grasp_scores_lower_than_unambiguous(self):
        """This is the exact scenario named in the spec: a marginal grasp near threshold must score visibly lower than an unambiguous one."""
        marginal = _pose_with_derived({"fingertip_dists": [0.14, 0.145, 0.148, 0.149, 0.1499]})
        unambiguous = _pose_with_derived({"fingertip_dists": [0.02, 0.03, 0.01, 0.025, 0.015]})
        assert confidence_power_grasp(marginal) < confidence_power_grasp(unambiguous)

    def test_no_hands_zero(self):
        assert confidence_power_grasp({"hands_detected": False}) == 0.0

    def test_open_hand_far_above_threshold_scores_high_too(self):
        """A confidently-open hand (far above threshold) should score high, same as confidently-closed."""
        open_hand = _pose_with_derived({"fingertip_dists": [0.5, 0.5, 0.5, 0.5, 0.5]})
        marginal = _pose_with_derived({"fingertip_dists": [0.14, 0.15, 0.16, 0.14, 0.15]})
        assert confidence_power_grasp(open_hand) > confidence_power_grasp(marginal)


class TestLateralPinchConfidence:
    def test_marginal_vs_unambiguous(self):
        marginal = _pose_with_derived({"thumb_index_dist": 0.078})
        unambiguous = _pose_with_derived({"thumb_index_dist": 0.005})
        assert confidence_lateral_pinch(marginal) < confidence_lateral_pinch(unambiguous)


class TestReachOnsetConfidence:
    def test_marginal_vs_unambiguous(self):
        prev = _pose_with_derived({"wrist_velocity_magnitude": 0.0})
        marginal = _pose_with_derived({"wrist_velocity_magnitude": cfg.REACH_WRIST_VEL + 0.01})
        unambiguous = _pose_with_derived({"wrist_velocity_magnitude": cfg.REACH_WRIST_VEL + 1.0})
        assert confidence_reach_onset(marginal, prev) < confidence_reach_onset(unambiguous, prev)

    def test_missing_frames_zero(self):
        assert confidence_reach_onset(None, None) == 0.0


class TestFingerVoteConfidence:
    def test_marginal_curl_scores_lower_than_unambiguous(self):
        prev = _finger_pose(curl_amount=0.0)
        marginal_curl = _finger_pose(curl_amount=0.02)   # tiny curl — most fingers likely still tied/near-zero delta
        strong_curl = _finger_pose(curl_amount=1.4)      # unambiguous curl, matches test_primitives.py's fixture

        marginal_conf = confidence_finger_curl(marginal_curl, prev)
        strong_conf = confidence_finger_curl(strong_curl, prev)
        assert strong_conf > marginal_conf

    def test_extend_direction_scores_symmetric_to_curl(self):
        curled_prev = _finger_pose(curl_amount=1.4)
        extended = _finger_pose(curl_amount=0.0)
        assert confidence_finger_extend(extended, curled_prev) > 0.5

    def test_missing_frames_zero(self):
        assert confidence_finger_curl(None, None) == 0.0
        assert confidence_finger_extend(None, None) == 0.0


class TestTransportConfidence:
    def _window(self, vel_mag, direction_deg=0.0, n=10, confidence=0.9):
        rad = math.radians(direction_deg)
        vel = [vel_mag * math.cos(rad), vel_mag * math.sin(rad)]
        return [
            _pose_with_derived({"wrist_velocity_magnitude": vel_mag, "wrist_velocity": vel}, confidence=confidence)
            for _ in range(n)
        ]

    def test_sustained_consistent_transport_scores_higher_than_marginal(self):
        marginal = self._window(cfg.REACH_WRIST_VEL + 0.01, direction_deg=0.0)
        unambiguous = self._window(cfg.REACH_WRIST_VEL + 1.0, direction_deg=0.0)
        assert confidence_transport(marginal) < confidence_transport(unambiguous)

    def test_short_window_zero(self):
        assert confidence_transport([None, None]) == 0.0

    def test_marginally_consistent_direction_scores_lower_than_strongly_consistent(self):
        """
        Both windows are above the 0.707 direction-consistency threshold
        (both would detect as transport=True), but one sits just above it
        (marginal) and the other is a unanimous single direction (strongly
        confident) — confidence must reflect that margin, not saturate
        identically just because both cross the boolean threshold. Speed
        is held constant across both so this isolates the direction term.
        """
        strongly_consistent = self._window(1.0, direction_deg=0.0)  # resultant_length = 1.0
        # 6 frames at 0 deg, 4 at 60 deg -> resultant_length ~= 0.87, still
        # above the 0.707 threshold but with a much smaller margin.
        marginally_consistent = self._window(1.0, direction_deg=0.0, n=6) + self._window(1.0, direction_deg=60.0, n=4)
        assert confidence_transport(marginally_consistent) < confidence_transport(strongly_consistent)


class TestIdleHeadConfidence:
    def test_marginal_vs_deep_stillness(self):
        deep_still = np.full((5, 3), 0.001)
        marginal = np.full((5, 3), 0.08)  # close to HEAD_IMU_THRESHOLDS ceiling
        gyro = np.zeros((5, 3))
        assert confidence_idle_head(marginal, gyro) < confidence_idle_head(deep_still, gyro)

    def test_empty_window_zero(self):
        assert confidence_idle_head(np.zeros((0, 3)), np.zeros((0, 3))) == 0.0


class TestContactVisionConfidence:
    def test_low_confidence_object_detection_suppresses_score(self):
        """A low-confidence real object detection (e.g. Grounding DINO near its box_threshold) should visibly suppress contact confidence even at a strong distance margin, vs. a confidently-detected object."""
        landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
        pose = {"hands_detected": True, "dominant_hand": "right", "right_hand": {"landmarks": landmarks, "confidence": 0.9}}
        low_conf_track = {"tracked_objects": [{"centroid_norm": [0.5, 0.5], "confidence": 0.10}]}
        real_track = {"tracked_objects": [{"centroid_norm": [0.5, 0.5], "confidence": 0.95}]}

        low_conf = confidence_contact_vision(pose, low_conf_track, dist_thresh=0.06)
        real_conf = confidence_contact_vision(pose, real_track, dist_thresh=0.06)
        assert low_conf < real_conf

    def test_no_object_track_falls_back_to_moderate_baseline(self):
        pose = _pose_with_derived({}, confidence=0.9)
        conf = confidence_contact_vision(pose, None, dist_thresh=0.06)
        assert 0.0 < conf < 1.0


class TestStrategyComputeConfidencesShape:
    """compute_confidences() must cover every primitive with a value in [0,1], for every strategy."""

    def _imu_window(self):
        return {
            "accel_window": np.random.RandomState(0).normal(0, 0.3, (5, 3)),
            "gyro_window": np.radians(np.random.RandomState(1).normal(0, 10, (5, 3))),
            "head_accel_window": np.full((5, 3), 0.01),
            "head_gyro_window": np.zeros((5, 3)),
            "dt": 1.0 / 30.0,
            "fps": 30.0,
            "pose_window": [None] * 5,
            "contact_onset_history": [False] * 5,
        }

    def test_wrist_primary_covers_all_primitives_in_range(self):
        strategy = WristPrimaryStrategy()
        result = strategy.compute_confidences(0, None, None, self._imu_window(), None)
        assert set(result.keys()) == set(ALL_PRIMITIVES)
        for prim, val in result.items():
            assert 0.0 <= val <= 1.0, f"{prim}={val} out of [0,1]"

    def test_vision_primary_covers_all_primitives_in_range(self):
        strategy = VisionPrimaryStrategy()
        result = strategy.compute_confidences(0, None, None, self._imu_window(), None)
        assert set(result.keys()) == set(ALL_PRIMITIVES)
        for prim, val in result.items():
            assert 0.0 <= val <= 1.0, f"{prim}={val} out of [0,1]"

    def test_fusion_covers_all_primitives_in_range(self):
        strategy = FusionStrategy()
        result = strategy.compute_confidences(0, None, None, self._imu_window(), None)
        assert set(result.keys()) == set(ALL_PRIMITIVES)
        for prim, val in result.items():
            assert 0.0 <= val <= 1.0, f"{prim}={val} out of [0,1]"


class TestFusionConfidenceWeighting:
    """Fusion's voted-primitive confidence must genuinely combine both strategies' scores, not just pick one."""

    def test_voted_primitive_is_weighted_average(self):
        strategy = FusionStrategy()
        gyro_window = np.radians(np.full((5, 3), 0.0))
        gyro_window[:, 2] = np.radians(50.0)  # strong IMU pronate signal
        imu_window = {
            "accel_window": np.zeros((5, 3)),
            "gyro_window": gyro_window,
            "head_accel_window": np.zeros((5, 3)),
            "head_gyro_window": np.zeros((5, 3)),
            "dt": 1.0 / 30.0,
            "fps": 30.0,
            "pose_window": [None],
            "contact_onset_history": [],
        }
        # No pose data at all -> vision-side pronate confidence is 0.0
        # (value_deg_s is None), IMU-side should be high.
        result = strategy.compute_confidences(0, None, None, imu_window, None)
        vision_conf = strategy._vision.compute_confidences(0, None, None, imu_window, None)
        wrist_conf = strategy._wrist.compute_confidences(0, None, None, imu_window, None)

        expected = cfg.IMU_FUSION_WEIGHT_VISION * vision_conf["wrist_pronate"] + cfg.IMU_FUSION_WEIGHT_IMU * wrist_conf["wrist_pronate"]
        assert result["wrist_pronate"] == pytest.approx(expected, abs=1e-6)
        # Sanity: this must NOT just equal the wrist-only value (i.e. the
        # weighting is actually applied, not a silent pass-through).
        assert result["wrist_pronate"] != pytest.approx(wrist_conf["wrist_pronate"], abs=1e-9)
