"""
DatraAI Pipeline — Step 04: Hand Pose Estimation
MediaPipe Hands on the perception source video (raw or compressed, per
config.PERCEPTION_SOURCE — v2 addendum §8) → hand_pose.json.

Input:  processed/{session_id}/compressed.mp4 or raw/{session_id}/raw.mp4
        (resolved by utils.video_utils.resolve_perception_source)
Output: processed/{session_id}/hand_pose.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.video_utils import resolve_perception_source

STEP = "04_hand_pose"


class NoHandsError(Exception):
    """Raised when hand presence rate is critically low."""
    pass


# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Palm center approximated as midpoint of wrist(0) and middle finger MCP(9)
PALM_CENTER_LANDMARKS = [0, 9]


def _landmark_distance(lm1, lm2) -> float:
    """Euclidean distance between two landmarks (x, y, z)."""
    return math.sqrt(
        (lm1[0] - lm2[0]) ** 2
        + (lm1[1] - lm2[1]) ** 2
        + (lm1[2] - lm2[2]) ** 2
    )


def _compute_palm_center(landmarks):
    """Compute palm center as midpoint of wrist and middle MCP."""
    w = landmarks[PALM_CENTER_LANDMARKS[0]]
    m = landmarks[PALM_CENTER_LANDMARKS[1]]
    return [(w[0] + m[0]) / 2, (w[1] + m[1]) / 2, (w[2] + m[2]) / 2]


def run(session_id: str) -> list:
    """
    Run MediaPipe hand pose on every frame and write hand_pose.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    # Import MediaPipe lazily (heavy dependency)
    import mediapipe as mp

    proc_dir = cfg.PROCESSED_DIR / session_id
    video_path = resolve_perception_source(session_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.TARGET_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{STEP}] Processing {total_frames} frames at {fps:.1f} FPS...")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    results_list = []
    prev_wrist = None
    prev_timestamp = None
    hands_detected_count = 0
    frame_idx = 0
    report_interval = max(1, total_frames // 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "hands_detected": False,
            "dominant_hand": None,
            "right_hand": None,
            "left_hand": None,
            "derived": None,
        }

        if result.multi_hand_landmarks and result.multi_handedness:
            frame_data["hands_detected"] = True
            hands_detected_count += 1

            hand_entries = {}
            for hand_landmarks, hand_class in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                # Determine hand label
                label = hand_class.classification[0].label.lower()  # "left" or "right"
                confidence = hand_class.classification[0].score

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([
                        round(lm.x, 6),
                        round(lm.y, 6),
                        round(lm.z, 6),
                    ])

                hand_entries[label] = {
                    "landmarks": landmarks,
                    "confidence": round(confidence, 4),
                }

            frame_data["right_hand"] = hand_entries.get("right")
            frame_data["left_hand"] = hand_entries.get("left")

            # Dominant hand: higher mean confidence
            if len(hand_entries) > 0:
                dominant = max(hand_entries.items(), key=lambda x: x[1]["confidence"])
                frame_data["dominant_hand"] = dominant[0]

            # Compute derived features from dominant hand
            dom_label = frame_data["dominant_hand"]
            dom_hand = hand_entries.get(dom_label) if dom_label else None

            if dom_hand:
                lms = dom_hand["landmarks"]
                palm_center = _compute_palm_center(lms)

                # Thumb-index distance
                thumb_index_dist = _landmark_distance(lms[THUMB_TIP], lms[INDEX_TIP])

                # Fingertip distances from palm center
                fingertip_dists = [
                    round(_landmark_distance(palm_center, lms[ft]), 6)
                    for ft in FINGERTIPS
                ]

                # Wrist velocity
                wrist = lms[WRIST]
                wrist_vel = [0.0, 0.0]
                wrist_vel_mag = 0.0

                if prev_wrist is not None and prev_timestamp is not None:
                    dt = timestamp_sec - prev_timestamp
                    if dt > 0:
                        wrist_vel = [
                            round((wrist[0] - prev_wrist[0]) / dt, 6),
                            round((wrist[1] - prev_wrist[1]) / dt, 6),
                        ]
                        wrist_vel_mag = math.sqrt(wrist_vel[0] ** 2 + wrist_vel[1] ** 2)

                prev_wrist = wrist
                prev_timestamp = timestamp_sec

                frame_data["derived"] = {
                    "thumb_index_dist": round(thumb_index_dist, 6),
                    "fingertip_dists": fingertip_dists,
                    "wrist_velocity": [round(v, 6) for v in wrist_vel],
                    "wrist_velocity_magnitude": round(wrist_vel_mag, 6),
                }
        else:
            # No hands — reset tracking state
            prev_wrist = None
            prev_timestamp = None

        results_list.append(frame_data)
        frame_idx += 1

        if frame_idx % report_interval == 0:
            pct = frame_idx / total_frames * 100
            print(f"[{STEP}] Progress: {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    hands.close()

    # Compute hand presence rate
    hand_presence_rate = hands_detected_count / max(1, frame_idx)
    print(f"[{STEP}] Hand presence rate: {hand_presence_rate:.1%} ({hands_detected_count}/{frame_idx} frames)")

    if hand_presence_rate < cfg.HAND_PRESENCE_RATE_MIN:
        raise NoHandsError(
            f"[{STEP}] Hand presence rate {hand_presence_rate:.1%} "
            f"below minimum {cfg.HAND_PRESENCE_RATE_MIN:.0%} "
            f"({hands_detected_count}/{frame_idx} frames) — session unusable for training"
        )

    # Write output
    output_path = proc_dir / "hand_pose.json"
    with open(output_path, "w") as f:
        json.dump(results_list, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"[{STEP}] Wrote {len(results_list)} frames to hand_pose.json ({size_mb:.1f}MB)")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return results_list


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 04: Hand Pose Estimation")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
