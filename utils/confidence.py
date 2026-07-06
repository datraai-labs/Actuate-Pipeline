"""
DatraAI Pipeline — Per-Primitive Confidence (v2 addendum §9)

Every boolean primitive detector in utils/imu_source_router.py has a
companion confidence function here that scores CERTAINTY in that boolean
call (whichever way it landed), not "how likely is this True". A
measurement sitting exactly at its decision threshold is a coin flip
regardless of which side it fell on and scores near 0; a measurement far
from the threshold — in either direction — scores near 1.

Three signals feed into a primitive's confidence, combined multiplicatively
where more than one applies:

  1. Margin over/under threshold, normalized (`_margin_confidence`) — the
     primary signal for every primitive.
  2. Landmark/tracking confidence (`_landmark_confidence`,
     `_nearest_object`) — MediaPipe's own per-hand classification score
     from hand_pose.json, or a tracked object's detection confidence from
     object_tracks.json, discounting a strong geometric margin computed
     from landmarks/tracks the upstream model itself wasn't sure about.
  3. IMU signal-to-noise (`_snr_factor`) — a peak reading inside a
     high-variance (jittery) window is less trustworthy than the same peak
     in a calm one; applies only to gyro/accel-window-derived primitives.

These are pure functions (no I/O) so they're unit-testable without a real
session — scripts/05_primitives.py's orchestration loop calls them once
per frame alongside the existing detect_* boolean calls.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg


def _margin_confidence(actual: float, threshold: float, scale: Optional[float] = None) -> float:
    """
    Confidence in a threshold-based boolean decision: 0.0 exactly at
    `threshold` (a coin flip), approaching 1.0 as `actual` moves away from
    it in either direction. `scale` is the distance from threshold that
    counts as "fully confident" — defaults to abs(threshold) itself, so a
    margin equal to the threshold's own magnitude reaches full confidence.
    """
    if scale is None:
        scale = abs(threshold)
    if scale <= 1e-9:
        scale = 1.0
    margin = abs(actual - threshold)
    return float(np.clip(margin / scale, 0.0, 1.0))


def _snr_factor(values: np.ndarray, floor: Optional[float] = None) -> float:
    """
    Multiplicative confidence discount from a 1D window's peak-to-noise
    ratio: peak absolute value vs. the RAW (signed) window's own standard
    deviation. Std is computed on signed values, not magnitudes — a window
    oscillating between +20 and -20 is genuinely jittery even though every
    sample has the same magnitude, and using abs() before computing std
    would hide exactly that sign-flipping noise. Fewer than 2 samples, or
    an all-zero window, can't establish a noise floor — returns `floor`
    (genuine uncertainty, not full trust).
    """
    if floor is None:
        floor = cfg.IMU_SNR_CONFIDENCE_FLOOR
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2:
        return floor
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-9:
        return floor
    std = float(np.std(values))
    snr = peak / (std + 1e-9)
    return float(np.clip(snr / cfg.IMU_SNR_CONFIDENCE_REFERENCE, floor, 1.0))


def _landmark_confidence(pose_frame: Optional[dict]) -> float:
    """
    MediaPipe's own per-hand classification confidence (0-1) for the
    dominant hand in this frame. No hand tracked at all is genuine
    uncertainty about whether landmarks would even be reliable here, not
    full confidence — returns a neutral 0.5, not 1.0.
    """
    if not pose_frame or not pose_frame.get("hands_detected"):
        return 0.5
    dom = pose_frame.get("dominant_hand")
    if not dom:
        return 0.5
    hand = pose_frame.get(f"{dom}_hand")
    if not hand:
        return 0.5
    return float(hand.get("confidence", 0.5))


def _nearest_object(pose_frame: Optional[dict], object_track_frame: Optional[dict]) -> Optional[Tuple[float, float]]:
    """
    (distance, confidence) of the nearest tracked object to the dominant
    hand's mean fingertip position, or None if unavailable. Mirrors
    utils.imu_source_router._nearest_object_distance's fingertip/centroid
    math but also surfaces the winning object's own detection confidence,
    which _nearest_object_distance doesn't need for its boolean-only caller.
    """
    if not object_track_frame:
        return None
    tracked = object_track_frame.get("tracked_objects", [])
    if not tracked:
        return None
    if not pose_frame or not pose_frame.get("hands_detected"):
        return None
    dom = pose_frame.get("dominant_hand")
    if not dom:
        return None
    hand = pose_frame.get(f"{dom}_hand")
    if not hand:
        return None
    landmarks = hand.get("landmarks")
    if not landmarks:
        return None

    tip_indices = [4, 8, 12, 16, 20]
    tips = [landmarks[i] for i in tip_indices if i < len(landmarks)]
    if not tips:
        return None
    hand_x = sum(t[0] for t in tips) / len(tips)
    hand_y = sum(t[1] for t in tips) / len(tips)

    best_dist = None
    best_conf = 0.0
    for obj in tracked:
        centroid = obj.get("centroid_norm")
        if not centroid or len(centroid) < 2:
            continue
        dist = math.hypot(hand_x - centroid[0], hand_y - centroid[1])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_conf = float(obj.get("confidence", 0.0))
    if best_dist is None:
        return None
    return best_dist, best_conf


# ═══════════════════════════════════════════════════════════════
# WRIST-IMU-DERIVED CONFIDENCE (WristPrimaryStrategy / FusionStrategy)
# ═══════════════════════════════════════════════════════════════


def confidence_wrist_pronate_imu(gyro_z_window: np.ndarray) -> float:
    if len(gyro_z_window) == 0:
        return 0.0
    deg_s = np.degrees(gyro_z_window)
    peak = float(np.max(deg_s))
    margin_conf = _margin_confidence(peak, cfg.WRIST_IMU_THRESHOLDS["PRONATE_GYRO_Z_DEG_S"])
    return margin_conf * _snr_factor(deg_s)


def confidence_wrist_supinate_imu(gyro_z_window: np.ndarray) -> float:
    if len(gyro_z_window) == 0:
        return 0.0
    deg_s = np.degrees(gyro_z_window)
    trough = float(np.min(deg_s))
    margin_conf = _margin_confidence(trough, cfg.WRIST_IMU_THRESHOLDS["SUPINATE_GYRO_Z_DEG_S"])
    return margin_conf * _snr_factor(deg_s)


def confidence_wrist_flex_imu(gyro_x_window: np.ndarray) -> float:
    if len(gyro_x_window) == 0:
        return 0.0
    deg_s = np.degrees(gyro_x_window)
    peak = float(np.max(deg_s))
    margin_conf = _margin_confidence(peak, cfg.WRIST_IMU_THRESHOLDS["FLEX_GYRO_X_DEG_S"])
    return margin_conf * _snr_factor(deg_s)


def confidence_contact_onset_imu(accel_window: np.ndarray) -> float:
    if len(accel_window) < 3:
        return 0.0
    accel_mag = np.linalg.norm(accel_window, axis=1)
    peak = float(np.max(accel_mag))
    margin_conf = _margin_confidence(peak, cfg.CONTACT_ACCEL_G)
    return margin_conf * _snr_factor(accel_mag)


# The 1.0g release threshold is a literal in detect_contact_release (not a
# named config constant) — mirrored here rather than introduced as a new
# config knob to avoid changing detect_contact_release's tuned behavior.
_CONTACT_RELEASE_ACCEL_G = 1.0


def confidence_contact_release_imu(accel_window: np.ndarray) -> float:
    if len(accel_window) == 0:
        return 0.0
    accel_mag = np.linalg.norm(accel_window, axis=1)
    mean_accel = float(np.mean(accel_mag))
    margin_conf = _margin_confidence(mean_accel, _CONTACT_RELEASE_ACCEL_G)
    return margin_conf * _snr_factor(accel_mag)


# Idle's own thresholds (0.15g accel, 0.05 wrist-vel) are literals inside
# detect_idle, not named config constants — mirrored here for the same
# reason as _CONTACT_RELEASE_ACCEL_G above.
_IDLE_ACCEL_MAG_MAX = 0.15
_IDLE_WRIST_VEL_MAX = 0.05


def confidence_idle_wrist(accel_window: np.ndarray, pose_frame: Optional[dict]) -> float:
    if len(accel_window) == 0:
        return 0.0
    accel_mag_mean = float(np.mean(np.linalg.norm(accel_window, axis=1)))
    wrist_vel = 0.0
    if pose_frame and pose_frame.get("hands_detected") and pose_frame.get("derived"):
        wrist_vel = pose_frame["derived"].get("wrist_velocity_magnitude", 0.0)

    # AND logic (both must hold) — overall confidence is bottlenecked by
    # whichever constraint sits closest to its own threshold.
    margin_accel = _margin_confidence(accel_mag_mean, _IDLE_ACCEL_MAG_MAX)
    margin_vel = _margin_confidence(wrist_vel, _IDLE_WRIST_VEL_MAX)
    return min(margin_accel, margin_vel)


# ═══════════════════════════════════════════════════════════════
# VISION-DERIVED CONFIDENCE (VisionPrimaryStrategy / FusionStrategy)
# ═══════════════════════════════════════════════════════════════


def confidence_wrist_rotation_vision(value_deg_s: Optional[float], threshold: float, pose_frame: Optional[dict]) -> float:
    """Shared by both wrist_pronate and wrist_supinate/flex vision paths — `value_deg_s` is None when landmarks were unavailable to compute it at all."""
    if value_deg_s is None:
        return 0.0
    return _margin_confidence(value_deg_s, threshold) * _landmark_confidence(pose_frame)


def confidence_power_grasp(pose_frame: Optional[dict], threshold: Optional[float] = None) -> float:
    """`threshold` mirrors detect_power_grasp's glove-adjusted/per-worker-calibrated override (v2 addendum §2/§5) — confidence must be scored against the SAME effective threshold the boolean call used, or the margin would be measured from the wrong baseline."""
    if not pose_frame or not pose_frame.get("hands_detected"):
        return 0.0
    derived = pose_frame.get("derived")
    if derived is None:
        return 0.0
    dists = derived.get("fingertip_dists", [])
    if len(dists) < 5:
        return 0.0
    if threshold is None:
        threshold = cfg.POWER_GRASP_DIST
    # All 5 must be below threshold (AND logic) — the least-closed
    # fingertip is the bottleneck that determines the boolean outcome.
    bottleneck = max(dists)
    margin_conf = _margin_confidence(bottleneck, threshold)
    return margin_conf * _landmark_confidence(pose_frame)


def confidence_lateral_pinch(pose_frame: Optional[dict], threshold: Optional[float] = None) -> float:
    """`threshold` mirrors detect_lateral_pinch's glove-adjusted/per-worker-calibrated override (v2 addendum §2/§5)."""
    if not pose_frame or not pose_frame.get("hands_detected"):
        return 0.0
    derived = pose_frame.get("derived")
    if derived is None:
        return 0.0
    dist = derived.get("thumb_index_dist", None)
    if dist is None:
        return 0.0
    if threshold is None:
        threshold = cfg.LATERAL_PINCH_DIST
    margin_conf = _margin_confidence(dist, threshold)
    return margin_conf * _landmark_confidence(pose_frame)


def confidence_reach_onset(pose_frame: Optional[dict], prev_pose_frame: Optional[dict]) -> float:
    if pose_frame is None or prev_pose_frame is None:
        return 0.0
    if not pose_frame.get("hands_detected") or not prev_pose_frame.get("hands_detected"):
        return 0.0
    derived = pose_frame.get("derived")
    if derived is None:
        return 0.0
    vel_mag = derived.get("wrist_velocity_magnitude", 0.0)
    margin_conf = _margin_confidence(vel_mag, cfg.REACH_WRIST_VEL)
    return margin_conf * _landmark_confidence(pose_frame)


# Finger curl/extend vote across up to 5 fingers with a >=3 majority
# decision boundary — margin is expressed in "votes away from the tie
# line", with a half-width of 2 votes spanning from the 3-vote boundary to
# a unanimous 5 (or down to 0), so a 3/5 split scores low and a 5/5 or 0/5
# split scores full confidence.
_FINGER_VOTE_THRESHOLD = 3.0
_FINGER_VOTE_SCALE = 2.0
# The vote count alone can't tell a decisive curl from a barely-perceptible
# one — 5 fingers each moving by 0.01 rad "wins" the vote just as
# unanimously as 5 fingers moving by 1.0 rad. This is the per-finger angle
# delta (radians) treated as "a fully confident curl/extend motion",
# discounting the vote-count margin by how large the underlying motion
# actually was.
_FINGER_DELTA_SCALE = 0.3


def confidence_finger_curl(pose_frame: Optional[dict], prev_pose_frame: Optional[dict]) -> float:
    return _confidence_finger_vote(pose_frame, prev_pose_frame, direction=-1)


def confidence_finger_extend(pose_frame: Optional[dict], prev_pose_frame: Optional[dict]) -> float:
    return _confidence_finger_vote(pose_frame, prev_pose_frame, direction=1)


def _confidence_finger_vote(pose_frame: Optional[dict], prev_pose_frame: Optional[dict], direction: int) -> float:
    from utils.imu_source_router import _get_finger_angles  # local import avoids a circular import at module load

    if pose_frame is None or prev_pose_frame is None:
        return 0.0
    if not pose_frame.get("hands_detected") or not prev_pose_frame.get("hands_detected"):
        return 0.0
    dom = pose_frame.get("dominant_hand")
    if not dom:
        return 0.0
    curr_hand = pose_frame.get(f"{dom}_hand")
    prev_hand = prev_pose_frame.get(f"{dom}_hand")
    if not curr_hand or not prev_hand:
        return 0.0

    curr_angles = _get_finger_angles(curr_hand["landmarks"])
    prev_angles = _get_finger_angles(prev_hand["landmarks"])

    count = 0
    matching_deltas = []
    for finger in curr_angles:
        if finger in prev_angles:
            delta = curr_angles[finger] - prev_angles[finger]
            if direction < 0 and delta < 0:
                count += 1
                matching_deltas.append(abs(delta))
            elif direction > 0 and delta > 0:
                count += 1
                matching_deltas.append(abs(delta))

    vote_margin = _margin_confidence(float(count), _FINGER_VOTE_THRESHOLD, scale=_FINGER_VOTE_SCALE)
    if matching_deltas:
        magnitude_conf = float(np.clip(np.mean(matching_deltas) / _FINGER_DELTA_SCALE, 0.0, 1.0))
        geometric_conf = vote_margin * magnitude_conf
    else:
        # Zero fingers moved this direction at all — a confidently-False
        # vote count with no motion magnitude to weigh it against.
        geometric_conf = vote_margin
    return geometric_conf * _landmark_confidence(pose_frame)


_TRANSPORT_VEL_COUNT_THRESHOLD = 6.0
_TRANSPORT_VEL_COUNT_SCALE = 4.0
_TRANSPORT_DIRECTION_THRESHOLD = 0.707
_TRANSPORT_DIRECTION_SCALE = 0.3


def confidence_transport(pose_frames_window: List[Optional[dict]]) -> float:
    if len(pose_frames_window) < 5:
        return 0.0

    velocities = []
    directions = []
    landmark_confs = []

    for pf in pose_frames_window:
        if pf and pf.get("hands_detected") and pf.get("derived"):
            vel_mag = pf["derived"].get("wrist_velocity_magnitude", 0.0)
            vel = pf["derived"].get("wrist_velocity", [0.0, 0.0])
            velocities.append(vel_mag)
            landmark_confs.append(_landmark_confidence(pf))
            if vel_mag > 0.01:
                directions.append(math.atan2(vel[1], vel[0]))
        else:
            velocities.append(0.0)

    high_vel_count = sum(1 for v in velocities if v > cfg.REACH_WRIST_VEL)
    margin_count = _margin_confidence(
        float(high_vel_count), _TRANSPORT_VEL_COUNT_THRESHOLD, scale=_TRANSPORT_VEL_COUNT_SCALE
    )
    # Count-of-frames-over-threshold alone can't distinguish "barely over
    # REACH_WRIST_VEL" from "far over it" when every frame in the window
    # has the same speed — average each frame's own margin so a faster,
    # more unambiguous transport genuinely scores higher.
    margin_vel_mag = float(np.mean([_margin_confidence(v, cfg.REACH_WRIST_VEL) for v in velocities])) if velocities else 0.0

    if len(directions) < 3:
        # detect_transport itself returns False outright here — no
        # direction signal to be confident about either way, so the
        # confidence in that False mirrors detect_idle's "no data" case.
        margin_dir = 0.0
    else:
        angles = np.array(directions)
        mean_x = np.mean(np.cos(angles))
        mean_y = np.mean(np.sin(angles))
        resultant_length = math.sqrt(mean_x ** 2 + mean_y ** 2)
        margin_dir = _margin_confidence(
            resultant_length, _TRANSPORT_DIRECTION_THRESHOLD, scale=_TRANSPORT_DIRECTION_SCALE
        )

    # AND logic (sustained speed AND consistent direction) — bottlenecked
    # by whichever constraint is closer to its own threshold. margin_vel_mag
    # additionally distinguishes "barely fast enough" from "unambiguously
    # fast" within a window where every frame already clears the count
    # threshold.
    geometric_conf = min(margin_count, margin_dir, margin_vel_mag)
    mean_landmark_conf = float(np.mean(landmark_confs)) if landmark_confs else 0.5
    return geometric_conf * mean_landmark_conf


_HEAD_IDLE_ACCEL_SCALE = None  # use threshold-magnitude default via _margin_confidence


def confidence_idle_head(head_accel_window: np.ndarray, head_gyro_window: np.ndarray) -> float:
    if head_accel_window.shape[0] == 0:
        return 0.0
    accel_mag = np.linalg.norm(head_accel_window, axis=1)
    accel_mag_mean = float(np.mean(accel_mag))
    gyro_mag_mean = 0.0
    if head_gyro_window.shape[0] > 0:
        gyro_mag_mean = float(np.mean(np.degrees(np.linalg.norm(head_gyro_window, axis=1))))

    margin_accel = _margin_confidence(accel_mag_mean, cfg.HEAD_IMU_THRESHOLDS["IDLE_ACCEL_MAG_MAX"])
    margin_gyro = _margin_confidence(gyro_mag_mean, cfg.HEAD_IMU_THRESHOLDS["IDLE_GYRO_MAG_MAX"])
    geometric_conf = min(margin_accel, margin_gyro)
    return geometric_conf * _snr_factor(accel_mag)


def confidence_contact_vision(
    pose_frame: Optional[dict],
    object_track_frame: Optional[dict],
    dist_thresh: float,
) -> float:
    """
    Vision-path contact confidence (VisionPrimaryStrategy._detect_contact_vision).
    With real object tracking: margin from the fingertip-to-object distance,
    discounted by the winning object's own detection confidence — a §3 STUB
    detection (fixed confidence ~0.10) correctly suppresses this almost
    entirely, since the underlying object identity isn't real yet. Without
    object tracking (current default, no 04c output wired in): there's no
    continuous geometric quantity to score, only the grasp-shape transition
    edge itself — a fixed moderate baseline discounted by landmark
    confidence, documented as a known simplification pending §3.
    """
    if object_track_frame is not None:
        nearest = _nearest_object(pose_frame, object_track_frame)
        if nearest is None:
            return 0.0
        dist, obj_conf = nearest
        margin_conf = _margin_confidence(dist, dist_thresh)
        return margin_conf * obj_conf

    return 0.6 * _landmark_confidence(pose_frame)
