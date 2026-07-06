"""
DatraAI Pipeline — IMU Source Router (v2 addendum §1)

A head-mounted IMU measures head/mount motion, not hand or finger motion —
it cannot physically capture finger curl, grasp aperture, or wrist rotation.
A wrist-mounted IMU (strap/glove) measures those directly, as in v1. This
module provides the primitive-detection strategy that scripts/05_primitives.py
delegates to, selected by config.IMU_SOURCE_MODE:

  - VisionPrimaryStrategy ("head_mounted" / "none"): fine-motor primitives
    come from hand_pose.json (+ object_tracks.json once available) only.
    Head IMU is only used for whole-body idle detection.
  - WristPrimaryStrategy ("wrist_mounted"): v1 behavior — gyro/accel-based
    detectors.
  - FusionStrategy ("dual"): confidence-weighted vote between the two,
    flagging genuinely unresolved disagreements for review.

The low-level per-primitive detector functions below were moved here from
scripts/05_primitives.py, which now re-exports them (via `from
utils.imu_source_router import ...`) so existing call sites and tests are
unaffected.
"""

import math
from typing import Dict, List, Optional

import numpy as np

import config as cfg

from utils.video_utils import (
    compute_wrist_rotation_from_landmarks,
    compute_wrist_flexion_from_landmarks,
)
from utils import confidence as conf

ALL_PRIMITIVES = [
    "wrist_pronate",
    "wrist_supinate",
    "wrist_flex",
    "reach_onset",
    "power_grasp",
    "lateral_pinch",
    "contact_onset",
    "contact_release",
    "finger_curl",
    "finger_extend",
    "idle",
    "transport",
]


# ═══════════════════════════════════════════════════════════════
# LOW-LEVEL PRIMITIVE DETECTOR FUNCTIONS
# (moved from scripts/05_primitives.py — logic unchanged except where noted)
# ═══════════════════════════════════════════════════════════════


def detect_wrist_pronate(gyro_z_window: np.ndarray) -> bool:
    """
    Detect wrist pronation from gyro Z-axis.
    gyro_z_window in rad/s → convert to deg/s.

    Peak-based rather than mean-based: a brief, fast wrist snap (e.g. bolt
    tightening) can spike well above threshold for 2 of a 5-frame window and
    still average below it. Requiring >=2 samples over threshold (rather
    than a bare peak check) keeps this robust to a single noisy sample.
    """
    if len(gyro_z_window) == 0:
        return False
    deg_s = np.degrees(gyro_z_window)
    above = deg_s > cfg.WRIST_IMU_THRESHOLDS["PRONATE_GYRO_Z_DEG_S"]
    return int(np.sum(above)) >= min(2, len(deg_s))


def detect_wrist_supinate(gyro_z_window: np.ndarray) -> bool:
    """
    Detect wrist supination from gyro Z-axis.
    Peak-based (see detect_wrist_pronate) — requires >=2 samples below
    SUPINATE_GYRO_Z_DEG_S rather than a window mean.
    """
    if len(gyro_z_window) == 0:
        return False
    deg_s = np.degrees(gyro_z_window)
    below = deg_s < cfg.WRIST_IMU_THRESHOLDS["SUPINATE_GYRO_Z_DEG_S"]
    return int(np.sum(below)) >= min(2, len(deg_s))


def detect_wrist_flex(gyro_x_window: np.ndarray) -> bool:
    """
    Detect wrist flexion from gyro X-axis.
    Peak-based (see detect_wrist_pronate) — requires >=2 samples above
    FLEX_GYRO_X_DEG_S rather than a window mean.
    """
    if len(gyro_x_window) == 0:
        return False
    deg_s = np.degrees(gyro_x_window)
    above = deg_s > cfg.WRIST_IMU_THRESHOLDS["FLEX_GYRO_X_DEG_S"]
    return int(np.sum(above)) >= min(2, len(deg_s))


def detect_reach_onset(pose_frame: Optional[dict], prev_pose_frame: Optional[dict], dt: float) -> bool:
    """
    Detect reach onset: wrist moving outward with velocity > REACH_WRIST_VEL.
    Outward = wrist x increasing (moving away from body center).
    """
    if pose_frame is None or prev_pose_frame is None:
        return False
    if not pose_frame.get("hands_detected") or not prev_pose_frame.get("hands_detected"):
        return False

    derived = pose_frame.get("derived")
    if derived is None:
        return False

    vel_mag = derived.get("wrist_velocity_magnitude", 0.0)
    vel = derived.get("wrist_velocity", [0.0, 0.0])

    # Outward: x velocity is positive (moving right = away from body in egocentric)
    return vel_mag > cfg.REACH_WRIST_VEL and vel[0] > 0


def detect_power_grasp(pose_frame: Optional[dict], threshold: Optional[float] = None) -> bool:
    """
    Power grasp: all 5 fingertip distances from palm < threshold
    (config.POWER_GRASP_DIST by default). `threshold` lets callers pass a
    glove-adjusted or per-worker-calibrated value (v2 addendum §2/§5) — see
    utils/glove_profile.py and utils/worker_profile_store.py.
    """
    if pose_frame is None or not pose_frame.get("hands_detected"):
        return False
    derived = pose_frame.get("derived")
    if derived is None:
        return False

    dists = derived.get("fingertip_dists", [])
    if len(dists) < 5:
        return False

    if threshold is None:
        threshold = cfg.POWER_GRASP_DIST
    return all(d < threshold for d in dists)


def detect_lateral_pinch(pose_frame: Optional[dict], threshold: Optional[float] = None) -> bool:
    """
    Lateral pinch: thumb-index distance < threshold (config.LATERAL_PINCH_DIST
    by default). `threshold` lets callers pass a glove-adjusted or
    per-worker-calibrated value (v2 addendum §2/§5).
    """
    if pose_frame is None or not pose_frame.get("hands_detected"):
        return False
    derived = pose_frame.get("derived")
    if derived is None:
        return False

    if threshold is None:
        threshold = cfg.LATERAL_PINCH_DIST
    return derived.get("thumb_index_dist", 999.0) < threshold


def detect_contact_onset(accel_window: np.ndarray) -> bool:
    """
    Contact onset: accel spike > CONTACT_ACCEL_G followed by deceleration.
    """
    if len(accel_window) < 3:
        return False

    accel_mag = np.linalg.norm(accel_window, axis=1)
    peak_idx = int(np.argmax(accel_mag))
    peak_val = accel_mag[peak_idx]

    if peak_val <= cfg.CONTACT_ACCEL_G:
        return False

    # Check deceleration after peak
    if peak_idx < len(accel_mag) - 1:
        after_peak = accel_mag[peak_idx + 1:]
        if len(after_peak) > 0:
            # Derivative after peak should be negative (decelerating)
            return float(after_peak[-1]) < float(peak_val)

    return False


def detect_contact_release(
    accel_window: np.ndarray,
    contact_history: List[bool],
    frame_idx: int,
) -> bool:
    """
    Contact release: contact_onset was True 1-10 frames ago AND current accel < 1.0g.
    """
    if len(accel_window) == 0:
        return False

    accel_mag = float(np.mean(np.linalg.norm(accel_window, axis=1)))

    # Check if contact_onset fired recently (1-10 frames ago)
    recent_contact = False
    look_back = min(10, len(contact_history))
    for i in range(1, look_back + 1):
        idx = frame_idx - i
        if 0 <= idx < len(contact_history) and contact_history[idx]:
            recent_contact = True
            break

    return recent_contact and accel_mag < 1.0


def _compute_joint_angle(a, b, c) -> float:
    """Compute angle at joint b between segments a-b and b-c (in radians)."""
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    dot = sum(ba[i] * bc[i] for i in range(3))
    mag_ba = math.sqrt(sum(x ** 2 for x in ba))
    mag_bc = math.sqrt(sum(x ** 2 for x in bc))

    if mag_ba * mag_bc == 0:
        return math.pi

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.acos(cos_angle)


# Finger joint indices: (MCP, PIP, DIP) for each finger
FINGER_JOINTS = {
    "index":  (5, 6, 7),
    "middle": (9, 10, 11),
    "ring":   (13, 14, 15),
    "pinky":  (17, 18, 19),
    "thumb":  (1, 2, 3),
}


def _get_finger_angles(landmarks: list) -> dict:
    """Compute MCP-PIP-DIP angle for each finger."""
    angles = {}
    for finger, (mcp, pip, dip) in FINGER_JOINTS.items():
        if len(landmarks) > max(mcp, pip, dip):
            angles[finger] = _compute_joint_angle(
                landmarks[mcp], landmarks[pip], landmarks[dip]
            )
    return angles


def detect_finger_curl(pose_frame: Optional[dict], prev_pose_frame: Optional[dict]) -> bool:
    """
    Finger curl: MCP→PIP→DIP angle decreasing for >= 3 fingers vs previous frame.
    """
    if pose_frame is None or prev_pose_frame is None:
        return False
    if not pose_frame.get("hands_detected") or not prev_pose_frame.get("hands_detected"):
        return False

    dom = pose_frame.get("dominant_hand")
    if not dom:
        return False

    # Compare the SAME hand label (dom) across both frames — not whichever
    # hand was dominant in prev_pose_frame, which can differ from dom if
    # handedness flipped between frames and would otherwise compare
    # unrelated hands' joint angles.
    curr_hand = pose_frame.get(f"{dom}_hand")
    prev_hand = prev_pose_frame.get(f"{dom}_hand")

    if not curr_hand or not prev_hand:
        return False

    curr_angles = _get_finger_angles(curr_hand["landmarks"])
    prev_angles = _get_finger_angles(prev_hand["landmarks"])

    curling_count = 0
    for finger in curr_angles:
        if finger in prev_angles:
            if curr_angles[finger] < prev_angles[finger]:
                curling_count += 1

    return curling_count >= 3


def detect_finger_extend(pose_frame: Optional[dict], prev_pose_frame: Optional[dict]) -> bool:
    """
    Finger extend: MCP→PIP→DIP angle increasing for >= 3 fingers vs previous frame.
    """
    if pose_frame is None or prev_pose_frame is None:
        return False
    if not pose_frame.get("hands_detected") or not prev_pose_frame.get("hands_detected"):
        return False

    dom = pose_frame.get("dominant_hand")
    if not dom:
        return False

    # Same-hand comparison — see detect_finger_curl for why this must use
    # `dom` on both frames rather than prev_pose_frame's own dominant hand.
    curr_hand = pose_frame.get(f"{dom}_hand")
    prev_hand = prev_pose_frame.get(f"{dom}_hand")

    if not curr_hand or not prev_hand:
        return False

    curr_angles = _get_finger_angles(curr_hand["landmarks"])
    prev_angles = _get_finger_angles(prev_hand["landmarks"])

    extending_count = 0
    for finger in curr_angles:
        if finger in prev_angles:
            if curr_angles[finger] > prev_angles[finger]:
                extending_count += 1

    return extending_count >= 3


def detect_idle(accel_window: np.ndarray, pose_frame: Optional[dict]) -> bool:
    """
    Idle: low accel magnitude AND low wrist velocity. (Wrist-mounted-IMU
    variant — see VisionPrimaryStrategy._detect_idle_head for the
    head-mounted-IMU equivalent.)
    """
    if len(accel_window) == 0:
        return False

    accel_mag = float(np.mean(np.linalg.norm(accel_window, axis=1)))

    wrist_vel = 0.0
    if pose_frame and pose_frame.get("hands_detected") and pose_frame.get("derived"):
        wrist_vel = pose_frame["derived"].get("wrist_velocity_magnitude", 0.0)

    return accel_mag < 0.15 and wrist_vel < 0.05


def detect_transport(
    pose_frames_window: List[Optional[dict]],
) -> bool:
    """
    Transport: sustained wrist velocity > REACH_WRIST_VEL for >= 6/10 frames
    with consistent direction (< 45° variance).
    """
    if len(pose_frames_window) < 5:
        return False

    velocities = []
    directions = []

    for pf in pose_frames_window:
        if pf and pf.get("hands_detected") and pf.get("derived"):
            vel_mag = pf["derived"].get("wrist_velocity_magnitude", 0.0)
            vel = pf["derived"].get("wrist_velocity", [0.0, 0.0])
            velocities.append(vel_mag)
            if vel_mag > 0.01:
                angle = math.atan2(vel[1], vel[0])
                directions.append(angle)
        else:
            velocities.append(0.0)

    # Check sustained velocity
    high_vel_count = sum(1 for v in velocities if v > cfg.REACH_WRIST_VEL)
    if high_vel_count < 6:
        return False

    # Check direction consistency
    if len(directions) < 3:
        return False

    angles = np.array(directions)
    # Circular variance: 1 - |mean of unit vectors|
    mean_x = np.mean(np.cos(angles))
    mean_y = np.mean(np.sin(angles))
    resultant_length = math.sqrt(mean_x ** 2 + mean_y ** 2)

    # If resultant length > cos(45°) ≈ 0.707, directions are consistent
    return resultant_length > 0.707


def check_imu_mount_plausibility(imu_source_mode: str, gyro: np.ndarray, accel: np.ndarray) -> dict:
    """
    Advisory heuristic on whether the single ingested IMU stream's gross
    motion characteristics look plausible for the declared
    config.IMU_SOURCE_MODE mount location.

    This exists because the pipeline currently ingests only one physical
    IMU stream and TRUSTS config.IMU_SOURCE_MODE to say where it's mounted
    — there is no second, independent stream to cross-check against, so a
    real hardware misconfiguration (e.g. a wrist strap feeding a session
    labeled "head_mounted") produces no error anywhere in ingest/sync. This
    check looks for gross statistical signatures that are implausible for
    the declared mount:

      - "head_mounted": frequent gyro spikes far exceeding plausible head
        rotation but squarely in wrist-snap territory suggest the stream is
        actually a wrist IMU.
      - "wrist_mounted": a gyro stream that is implausibly still across the
        whole session (never approaching even a modest rotation) suggests
        the stream is actually a head IMU during otherwise-active work.

    This is a best-effort heuristic, not a ground-truth check — it can miss
    real mismatches and can false-positive on legitimate motion (e.g. a
    worker who rapidly turns their head). It never blocks the pipeline or
    changes primitive-detection behavior; it only flags for human review.

    Returns:
        {"checked": bool, "plausible": bool, "reason": str|None,
         "peak_gyro_deg_s": float, "spike_fraction": float}
    """
    if not cfg.IMU_MOUNT_PLAUSIBILITY_CHECK_ENABLED or imu_source_mode not in (
        "head_mounted",
        "wrist_mounted",
        "dual",
    ):
        return {
            "checked": False,
            "plausible": True,
            "reason": None,
            "peak_gyro_deg_s": 0.0,
            "spike_fraction": 0.0,
        }

    if gyro is None or len(gyro) == 0:
        return {
            "checked": False,
            "plausible": True,
            "reason": None,
            "peak_gyro_deg_s": 0.0,
            "spike_fraction": 0.0,
        }

    gyro_mag_deg = np.degrees(np.linalg.norm(gyro, axis=1))
    peak = float(np.max(gyro_mag_deg))
    spike_fraction = float(np.mean(gyro_mag_deg > cfg.IMU_MOUNT_WRIST_SNAP_LIKE_DEG_S))

    if imu_source_mode in ("head_mounted", "dual") and spike_fraction > cfg.IMU_MOUNT_WRIST_SNAP_FRACTION_THRESHOLD:
        return {
            "checked": True,
            "plausible": False,
            "reason": (
                f"{spike_fraction:.1%} of frames show gyro spikes above "
                f"{cfg.IMU_MOUNT_WRIST_SNAP_LIKE_DEG_S:.0f} deg/s, which is far more "
                f"consistent with wrist-snap motion than head motion — the physically "
                f"mounted device may actually be wrist-mounted, not head-mounted as "
                f"configured (IMU_SOURCE_MODE={imu_source_mode!r})."
            ),
            "peak_gyro_deg_s": peak,
            "spike_fraction": spike_fraction,
        }

    if (
        imu_source_mode == "wrist_mounted"
        and spike_fraction < cfg.IMU_MOUNT_HEAD_STILLNESS_SPIKE_FRACTION_MAX
        and peak < cfg.IMU_MOUNT_HEAD_STILLNESS_PEAK_DEG_S
    ):
        return {
            "checked": True,
            "plausible": False,
            "reason": (
                f"Gyro stream never exceeds {peak:.1f} deg/s across the whole session — "
                f"implausibly still for a wrist IMU during active manipulation. The "
                f"physically mounted device may actually be head-mounted, not "
                f"wrist-mounted as configured."
            ),
            "peak_gyro_deg_s": peak,
            "spike_fraction": spike_fraction,
        }

    return {
        "checked": True,
        "plausible": True,
        "reason": None,
        "peak_gyro_deg_s": peak,
        "spike_fraction": spike_fraction,
    }


def apply_minimum_duration_filter(
    raw_flags: Dict[str, List[bool]],
    min_frames: int,
) -> Dict[str, List[bool]]:
    """
    Suppress primitive detections shorter than min_frames consecutive frames.
    """
    smoothed = {}

    for prim_name, flags in raw_flags.items():
        n = len(flags)
        result = list(flags)

        # Find runs of True
        i = 0
        while i < n:
            if result[i]:
                # Find end of this run
                j = i
                while j < n and result[j]:
                    j += 1
                run_len = j - i
                if run_len < min_frames:
                    # Suppress this short run
                    for k in range(i, j):
                        result[k] = False
                i = j
            else:
                i += 1

        smoothed[prim_name] = result

    return smoothed


def _nearest_object_distance(pose_frame: Optional[dict], object_track_frame: Optional[dict]) -> Optional[float]:
    """
    Normalized distance from the dominant hand's mean fingertip position to
    the nearest tracked object's centroid, for one frame. None if either
    input is unavailable (v2 addendum §3 — object_tracks.json).
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

    best = None
    for obj in tracked:
        centroid = obj.get("centroid_norm")
        if not centroid or len(centroid) < 2:
            continue
        dist = math.hypot(hand_x - centroid[0], hand_y - centroid[1])
        if best is None or dist < best:
            best = dist
    return best


# ═══════════════════════════════════════════════════════════════
# PRIMITIVE STRATEGIES
# ═══════════════════════════════════════════════════════════════


class PrimitiveStrategy:
    """
    Base interface for per-frame primitive detection. `imu_window` is a
    dict bundling whatever windowed context the strategy needs — built by
    scripts/05_primitives.py's orchestration loop each frame:

      accel_window, gyro_window : np.ndarray[N,3]  — wrist/forearm IMU window (± a few frames)
      head_accel_window, head_gyro_window : np.ndarray[N,3] — head-mount IMU window
      dt, fps                   : float             — frame timing
      pose_window                : list[dict|None]  — last ~10 pose frames (for detect_transport)
      contact_onset_history       : list[bool]       — per-frame contact_onset history so far
    """

    name = "base"

    def detect_primitives(
        self,
        frame_idx: int,
        pose_frame: Optional[dict],
        prev_pose_frame: Optional[dict],
        imu_window: dict,
        object_track_frame: Optional[dict],
    ) -> Dict[str, bool]:
        raise NotImplementedError

    def compute_confidences(
        self,
        frame_idx: int,
        pose_frame: Optional[dict],
        prev_pose_frame: Optional[dict],
        imu_window: dict,
        object_track_frame: Optional[dict],
    ) -> Dict[str, float]:
        """
        Per-primitive confidence float (0-1) for this same frame — a
        separate call from detect_primitives() (not merged into its return
        dict) so existing callers/tests asserting `set(result.keys()) ==
        set(ALL_PRIMITIVES)` on detect_primitives() are unaffected. See
        utils/confidence.py for the derivation (v2 addendum §9).
        """
        raise NotImplementedError


class WristPrimaryStrategy(PrimitiveStrategy):
    """
    v1 behavior — used when IMU_SOURCE_MODE == "wrist_mounted". Fine-motor
    primitives come from wrist/forearm IMU gyro/accel; grasp shape and
    finger articulation come from hand pose (unchanged either way, since
    those were always vision-derived).
    """

    name = "wrist_primary"

    def detect_primitives(self, frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame):
        accel_window = imu_window.get("accel_window", np.zeros((0, 3)))
        gyro_window = imu_window.get("gyro_window", np.zeros((0, 3)))
        gyro_z_window = gyro_window[:, 2] if gyro_window.shape[0] else np.array([])
        gyro_x_window = gyro_window[:, 0] if gyro_window.shape[0] else np.array([])
        dt = imu_window.get("dt", 1.0 / cfg.TARGET_FPS)
        pose_window = imu_window.get("pose_window", [pose_frame])
        contact_history = imu_window.get("contact_onset_history", [])
        # Glove-adjusted or per-worker-calibrated overrides (v2 addendum
        # §2/§5) — None falls back to config defaults inside the detectors.
        power_grasp_dist = imu_window.get("power_grasp_dist")
        lateral_pinch_dist = imu_window.get("lateral_pinch_dist")

        return {
            "wrist_pronate": detect_wrist_pronate(gyro_z_window),
            "wrist_supinate": detect_wrist_supinate(gyro_z_window),
            "wrist_flex": detect_wrist_flex(gyro_x_window),
            "reach_onset": detect_reach_onset(pose_frame, prev_pose_frame, dt),
            "power_grasp": detect_power_grasp(pose_frame, power_grasp_dist),
            "lateral_pinch": detect_lateral_pinch(pose_frame, lateral_pinch_dist),
            "contact_onset": detect_contact_onset(accel_window),
            "contact_release": detect_contact_release(accel_window, contact_history, frame_idx),
            "finger_curl": detect_finger_curl(pose_frame, prev_pose_frame),
            "finger_extend": detect_finger_extend(pose_frame, prev_pose_frame),
            "idle": detect_idle(accel_window, pose_frame),
            "transport": detect_transport(pose_window),
        }

    def compute_confidences(self, frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame):
        accel_window = imu_window.get("accel_window", np.zeros((0, 3)))
        gyro_window = imu_window.get("gyro_window", np.zeros((0, 3)))
        gyro_z_window = gyro_window[:, 2] if gyro_window.shape[0] else np.array([])
        gyro_x_window = gyro_window[:, 0] if gyro_window.shape[0] else np.array([])
        pose_window = imu_window.get("pose_window", [pose_frame])
        power_grasp_dist = imu_window.get("power_grasp_dist")
        lateral_pinch_dist = imu_window.get("lateral_pinch_dist")

        return {
            "wrist_pronate": conf.confidence_wrist_pronate_imu(gyro_z_window),
            "wrist_supinate": conf.confidence_wrist_supinate_imu(gyro_z_window),
            "wrist_flex": conf.confidence_wrist_flex_imu(gyro_x_window),
            "reach_onset": conf.confidence_reach_onset(pose_frame, prev_pose_frame),
            "power_grasp": conf.confidence_power_grasp(pose_frame, power_grasp_dist),
            "lateral_pinch": conf.confidence_lateral_pinch(pose_frame, lateral_pinch_dist),
            "contact_onset": conf.confidence_contact_onset_imu(accel_window),
            "contact_release": conf.confidence_contact_release_imu(accel_window),
            "finger_curl": conf.confidence_finger_curl(pose_frame, prev_pose_frame),
            "finger_extend": conf.confidence_finger_extend(pose_frame, prev_pose_frame),
            "idle": conf.confidence_idle_wrist(accel_window, pose_frame),
            "transport": conf.confidence_transport(pose_window),
        }


class VisionPrimaryStrategy(PrimitiveStrategy):
    """
    Used when IMU_SOURCE_MODE is "head_mounted" or "none". Fine-motor
    primitives come entirely from hand_pose.json (+ object_tracks.json once
    scripts/04c_object_track.py exists — v2 addendum §3) since a
    head-mounted IMU cannot see hand/finger motion. Head IMU is only
    consulted here for whole-body idle detection.
    """

    name = "vision_primary"

    def __init__(self):
        # Only needed for the object-track contact path, where we don't
        # receive the previous frame's object_track_frame and so must carry
        # the "was in contact" state ourselves across sequential calls.
        self._prev_object_contact = False

    def detect_primitives(self, frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame):
        fps = imu_window.get("fps", cfg.TARGET_FPS)
        dt = 1.0 / fps if fps > 0 else 0.0
        pose_window = imu_window.get("pose_window", [pose_frame])
        thresholds = cfg.WRIST_IMU_THRESHOLDS
        # Glove-adjusted or per-worker-calibrated overrides (v2 addendum
        # §2/§5) — None falls back to config defaults inside the detectors.
        power_grasp_dist = imu_window.get("power_grasp_dist")
        lateral_pinch_dist = imu_window.get("lateral_pinch_dist")

        pronation_deg_s = compute_wrist_rotation_from_landmarks(pose_frame, prev_pose_frame, fps)
        flexion_deg_s = compute_wrist_flexion_from_landmarks(pose_frame, prev_pose_frame, fps)

        wrist_pronate = pronation_deg_s is not None and pronation_deg_s > thresholds["PRONATE_GYRO_Z_DEG_S"]
        wrist_supinate = pronation_deg_s is not None and pronation_deg_s < thresholds["SUPINATE_GYRO_Z_DEG_S"]
        wrist_flex = flexion_deg_s is not None and abs(flexion_deg_s) > thresholds["FLEX_GYRO_X_DEG_S"]

        power_grasp = detect_power_grasp(pose_frame, power_grasp_dist)
        lateral_pinch = detect_lateral_pinch(pose_frame, lateral_pinch_dist)
        grasping = power_grasp or lateral_pinch
        prev_grasping = (
            detect_power_grasp(prev_pose_frame, power_grasp_dist)
            or detect_lateral_pinch(prev_pose_frame, lateral_pinch_dist)
        )

        contact_onset, contact_release = self._detect_contact_vision(
            pose_frame, object_track_frame, grasping, prev_grasping
        )

        head_accel_window = imu_window.get("head_accel_window", np.zeros((0, 3)))
        head_gyro_window = imu_window.get("head_gyro_window", np.zeros((0, 3)))

        return {
            "wrist_pronate": wrist_pronate,
            "wrist_supinate": wrist_supinate,
            "wrist_flex": wrist_flex,
            "reach_onset": detect_reach_onset(pose_frame, prev_pose_frame, dt),
            "power_grasp": power_grasp,
            "lateral_pinch": lateral_pinch,
            "contact_onset": contact_onset,
            "contact_release": contact_release,
            "finger_curl": detect_finger_curl(pose_frame, prev_pose_frame),
            "finger_extend": detect_finger_extend(pose_frame, prev_pose_frame),
            "idle": self._detect_idle_head(head_accel_window, head_gyro_window),
            "transport": detect_transport(pose_window),
        }

    def _detect_contact_vision(self, pose_frame, object_track_frame, grasping, prev_grasping):
        """
        Returns (contact_onset, contact_release).

        With object tracking (v2 addendum §3) available: contact requires
        fingertip-to-nearest-tracked-object proximity below
        HAND_OBJECT_CONTACT_DIST_NORM AND a grasp shape; onset/release are
        the transition edges of that combined state (tracked via instance
        state since only the current frame's object_track_frame is given).

        Without object tracking (current default until 04c_object_track.py
        is wired into scripts/05_primitives.py): falls back to the
        grasp-shape transition edge alone.
        """
        if object_track_frame is not None:
            dist_thresh = getattr(cfg, "HAND_OBJECT_CONTACT_DIST_NORM", 0.06)
            nearest_dist = _nearest_object_distance(pose_frame, object_track_frame)
            in_contact = bool(grasping and nearest_dist is not None and nearest_dist < dist_thresh)
            onset = in_contact and not self._prev_object_contact
            release = self._prev_object_contact and not in_contact
            self._prev_object_contact = in_contact
            return onset, release

        onset = grasping and not prev_grasping
        release = prev_grasping and not grasping
        return onset, release

    @staticmethod
    def _detect_idle_head(head_accel_window: np.ndarray, head_gyro_window: np.ndarray) -> bool:
        """Whole-body stillness from the head-mounted IMU (config.HEAD_IMU_THRESHOLDS)."""
        if head_accel_window.shape[0] == 0:
            return False
        accel_mag = float(np.mean(np.linalg.norm(head_accel_window, axis=1)))
        gyro_mag_deg = 0.0
        if head_gyro_window.shape[0] > 0:
            gyro_mag_deg = float(np.mean(np.degrees(np.linalg.norm(head_gyro_window, axis=1))))
        return (
            accel_mag < cfg.HEAD_IMU_THRESHOLDS["IDLE_ACCEL_MAG_MAX"]
            and gyro_mag_deg < cfg.HEAD_IMU_THRESHOLDS["IDLE_GYRO_MAG_MAX"]
        )

    def compute_confidences(self, frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame):
        fps = imu_window.get("fps", cfg.TARGET_FPS)
        pose_window = imu_window.get("pose_window", [pose_frame])
        thresholds = cfg.WRIST_IMU_THRESHOLDS
        dist_thresh = getattr(cfg, "HAND_OBJECT_CONTACT_DIST_NORM", 0.06)
        power_grasp_dist = imu_window.get("power_grasp_dist")
        lateral_pinch_dist = imu_window.get("lateral_pinch_dist")

        pronation_deg_s = compute_wrist_rotation_from_landmarks(pose_frame, prev_pose_frame, fps)
        flexion_deg_s = compute_wrist_flexion_from_landmarks(pose_frame, prev_pose_frame, fps)
        abs_flexion_deg_s = abs(flexion_deg_s) if flexion_deg_s is not None else None

        head_accel_window = imu_window.get("head_accel_window", np.zeros((0, 3)))
        head_gyro_window = imu_window.get("head_gyro_window", np.zeros((0, 3)))

        # contact_onset/contact_release share one in_contact state's edges
        # (see _detect_contact_vision) — same confidence value for both,
        # mirroring how the boolean detection itself treats them as one
        # continuous state's transitions rather than independent signals.
        contact_conf = conf.confidence_contact_vision(pose_frame, object_track_frame, dist_thresh)

        return {
            "wrist_pronate": conf.confidence_wrist_rotation_vision(
                pronation_deg_s, thresholds["PRONATE_GYRO_Z_DEG_S"], pose_frame
            ),
            "wrist_supinate": conf.confidence_wrist_rotation_vision(
                pronation_deg_s, thresholds["SUPINATE_GYRO_Z_DEG_S"], pose_frame
            ),
            "wrist_flex": conf.confidence_wrist_rotation_vision(
                abs_flexion_deg_s, thresholds["FLEX_GYRO_X_DEG_S"], pose_frame
            ),
            "reach_onset": conf.confidence_reach_onset(pose_frame, prev_pose_frame),
            "power_grasp": conf.confidence_power_grasp(pose_frame, power_grasp_dist),
            "lateral_pinch": conf.confidence_lateral_pinch(pose_frame, lateral_pinch_dist),
            "contact_onset": contact_conf,
            "contact_release": contact_conf,
            "finger_curl": conf.confidence_finger_curl(pose_frame, prev_pose_frame),
            "finger_extend": conf.confidence_finger_extend(pose_frame, prev_pose_frame),
            "idle": conf.confidence_idle_head(head_accel_window, head_gyro_window),
            "transport": conf.confidence_transport(pose_window),
        }


class FusionStrategy(PrimitiveStrategy):
    """
    Used when IMU_SOURCE_MODE == "dual" (both wrist IMU and vision
    available). Confidence-weighted vote between VisionPrimaryStrategy and
    WristPrimaryStrategy, but only for primitives with genuinely
    independent derivations from the two modalities (wrist rotation +
    contact). Pose-derived primitives (grasp/pinch/finger articulation/
    reach/transport) are computed identically by both strategies — there's
    nothing to vote on, so they're taken directly from one.
    """

    name = "fusion"

    _VOTED_PRIMITIVES = (
        "wrist_pronate",
        "wrist_supinate",
        "wrist_flex",
        "contact_onset",
        "contact_release",
    )

    def __init__(self):
        self._vision = VisionPrimaryStrategy()
        self._wrist = WristPrimaryStrategy()

    def detect_primitives(self, frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame):
        vision_result = self._vision.detect_primitives(
            frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame
        )
        wrist_result = self._wrist.detect_primitives(
            frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame
        )

        merged = dict(wrist_result)  # pose-derived fields are identical either way
        disagreement = False

        for prim in self._VOTED_PRIMITIVES:
            vision_flag = bool(vision_result.get(prim, False))
            imu_flag = bool(wrist_result.get(prim, False))
            weighted_score = (
                cfg.IMU_FUSION_WEIGHT_VISION * vision_flag
                + cfg.IMU_FUSION_WEIGHT_IMU * imu_flag
            )
            merged[prim] = weighted_score >= 0.5

            if vision_flag != imu_flag and abs(weighted_score - 0.5) < cfg.IMU_FUSION_DISAGREEMENT_MARGIN:
                disagreement = True

        merged["disagreement"] = disagreement
        return merged

    def compute_confidences(self, frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame):
        vision_conf = self._vision.compute_confidences(
            frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame
        )
        wrist_conf = self._wrist.compute_confidences(
            frame_idx, pose_frame, prev_pose_frame, imu_window, object_track_frame
        )

        # Pose-derived confidences (identical either way, same as the
        # boolean merge above) come from wrist_conf; voted primitives are
        # weighted the same way as the boolean vote itself.
        merged = dict(wrist_conf)
        for prim in self._VOTED_PRIMITIVES:
            v = vision_conf.get(prim, 0.0)
            w = wrist_conf.get(prim, 0.0)
            merged[prim] = float(np.clip(
                cfg.IMU_FUSION_WEIGHT_VISION * v + cfg.IMU_FUSION_WEIGHT_IMU * w, 0.0, 1.0
            ))
        return merged


def get_primitive_strategy(imu_source_mode: str) -> PrimitiveStrategy:
    """
    Factory for the primitive-detection strategy matching
    config.IMU_SOURCE_MODE.
    """
    if imu_source_mode in ("head_mounted", "none"):
        return VisionPrimaryStrategy()
    if imu_source_mode == "wrist_mounted":
        return WristPrimaryStrategy()
    if imu_source_mode == "dual":
        return FusionStrategy()
    raise ValueError(
        f"[imu_source_router] Unknown IMU_SOURCE_MODE: {imu_source_mode!r}. "
        f"Expected one of: head_mounted, wrist_mounted, dual, none."
    )
