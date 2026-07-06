"""
DatraAI Pipeline — Step 00: Per-Worker Calibration (v2 addendum §5)

A short, explicit calibration clip — worker spreads their hand fully open,
closes into a full fist, then does a lateral (thumb-index) pinch — lets
grasp/pinch detection thresholds be personalized to that worker's actual
hand geometry instead of one global default tuned for an unknown "typical"
hand. The derived thresholds are stored via utils/worker_profile_store.py
(consent-gated, retention-limited) and looked up by
scripts/05_primitives.py at runtime for any session that declares a
matching worker_id (v2 addendum §2's session_config.json).

This stage is optional per worker — a session with no worker_id, or a
worker with no calibration profile on file, simply falls back to the
glove-adjusted or raw global default (see utils/glove_profile.py).

NEEDS REAL VALIDATION: the frame-derivation logic below (_compute_worker_thresholds)
is pure and unit-tested with synthetic hand_pose-shaped fixtures, but this
has NOT been run against a real calibration recording in this dev
environment — no such footage exists yet (see docs/PIPELINE_STATUS.md's
business-blocker note on real footage generally). The MediaPipe extraction
path (_extract_calibration_hand_pose) reuses the same, already-verified
MediaPipe Hands call as scripts/04_hand_pose.py, so that part carries the
same confidence as §1/§10's MediaPipe usage — it's specifically the
calibration-clip-shape assumption (clear open/closed/pinch phases) that's
unverified against a real clip.

Input:  raw/{worker_id}/calibration.mp4 (worker: open hand -> fist -> pinch)
Output: calibration/workers/{worker_id}_profile.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.worker_profile_store import save_profile

STEP = "00_calibration"


def _extract_calibration_hand_pose(video_path: Path) -> list:
    """
    Run MediaPipe Hands over a calibration clip and return hand_pose.json-shaped
    frame dicts (reuses the same landmark/derived-feature extraction as
    scripts/04_hand_pose.py — NEEDS REAL GPU/webcam validation, see module
    docstring). Kept separate from _compute_worker_thresholds so the
    threshold-derivation math is testable with plain synthetic fixtures,
    with no MediaPipe/video dependency at all.
    """
    import cv2
    import mediapipe as mp

    # Local import + inline landmark math mirrors scripts/04_hand_pose.py's
    # approach exactly (same fingertip/palm-center/thumb-index derivation)
    # so a calibration clip and a real session frame produce directly
    # comparable "derived" values — see that script for the reference
    # implementation this mirrors.
    from importlib import util as _importlib_util

    hand_pose_spec = _importlib_util.spec_from_file_location(
        "hand_pose_ref", str(Path(__file__).resolve().parent / "04_hand_pose.py")
    )
    hand_pose_mod = _importlib_util.module_from_spec(hand_pose_spec)
    hand_pose_spec.loader.exec_module(hand_pose_mod)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        frame_data = {"frame_idx": frame_idx, "hands_detected": False, "derived": None}
        if result.multi_hand_landmarks:
            landmarks = [[round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)] for lm in result.multi_hand_landmarks[0].landmark]
            palm_center = hand_pose_mod._compute_palm_center(landmarks)
            thumb_index_dist = hand_pose_mod._landmark_distance(
                landmarks[hand_pose_mod.THUMB_TIP], landmarks[hand_pose_mod.INDEX_TIP]
            )
            fingertip_dists = [
                hand_pose_mod._landmark_distance(palm_center, landmarks[ft])
                for ft in hand_pose_mod.FINGERTIPS
            ]
            frame_data["hands_detected"] = True
            frame_data["derived"] = {
                "thumb_index_dist": round(thumb_index_dist, 6),
                "fingertip_dists": [round(d, 6) for d in fingertip_dists],
            }
        frames.append(frame_data)
        frame_idx += 1

    cap.release()
    hands.close()
    return frames


def _compute_worker_thresholds(hand_pose_frames: list) -> dict:
    """
    Pure function: given hand_pose.json-shaped frames from a calibration
    clip (open hand -> fist -> pinch), derive personalized
    power_grasp_dist / lateral_pinch_dist. Takes the widest fingertip
    spread seen (most-open configuration) and the tightest spread seen
    (most-closed / full-fist configuration) across the WHOLE clip and sets
    the threshold at their midpoint — same logic for thumb-index distance
    -> lateral_pinch_dist. No assumption about which portion of the clip
    is "open" vs "closed" — the max/min over the whole clip finds both
    extremes regardless of ordering, as long as the worker actually
    performs both a full open and a full close somewhere in the clip.
    """
    detected_frames = [f for f in hand_pose_frames if f.get("hands_detected") and f.get("derived")]

    all_fingertip_vals = [d for f in detected_frames for d in f["derived"].get("fingertip_dists", [])]
    thumb_index_vals = [f["derived"]["thumb_index_dist"] for f in detected_frames if "thumb_index_dist" in f["derived"]]

    if len(all_fingertip_vals) < (cfg.CALIBRATION_MIN_OPEN_FRAMES + cfg.CALIBRATION_MIN_CLOSED_FRAMES):
        raise ValueError(
            f"[{STEP}] Not enough hand-detected calibration frames "
            f"({len(detected_frames)} frames with hands detected) to trust a "
            f"derived threshold — need at least "
            f"{cfg.CALIBRATION_MIN_OPEN_FRAMES + cfg.CALIBRATION_MIN_CLOSED_FRAMES} "
            f"fingertip-distance samples total."
        )
    if len(thumb_index_vals) < cfg.CALIBRATION_MIN_PINCH_FRAMES:
        raise ValueError(
            f"[{STEP}] Not enough thumb-index samples ({len(thumb_index_vals)}) "
            f"to trust a derived lateral_pinch_dist — need at least "
            f"{cfg.CALIBRATION_MIN_PINCH_FRAMES}."
        )

    max_open = max(all_fingertip_vals)
    min_closed = min(all_fingertip_vals)
    power_grasp_dist = round((max_open + min_closed) / 2.0, 6)

    max_pinch_open = max(thumb_index_vals)
    min_pinch_closed = min(thumb_index_vals)
    lateral_pinch_dist = round((max_pinch_open + min_pinch_closed) / 2.0, 6)

    return {
        "power_grasp_dist": power_grasp_dist,
        "lateral_pinch_dist": lateral_pinch_dist,
        "calibration_open_dist_max": round(max_open, 6),
        "calibration_closed_dist_min": round(min_closed, 6),
        "calibration_pinch_open_max": round(max_pinch_open, 6),
        "calibration_pinch_closed_min": round(min_pinch_closed, 6),
        "n_fingertip_samples": len(all_fingertip_vals),
        "n_pinch_samples": len(thumb_index_vals),
    }


def run(worker_id: str, calibration_video_path: Path, consent_granted: bool) -> dict:
    """
    Process a worker's calibration clip and store their personalized
    grasp/pinch thresholds. Fail-closed on consent — see
    utils/worker_profile_store.save_profile.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting calibration for worker_id={worker_id!r}...")

    if not calibration_video_path.exists():
        raise FileNotFoundError(f"[{STEP}] Calibration video not found: {calibration_video_path}")

    hand_pose_frames = _extract_calibration_hand_pose(calibration_video_path)
    print(f"[{STEP}] Extracted hand pose from {len(hand_pose_frames)} calibration frames")

    thresholds = _compute_worker_thresholds(hand_pose_frames)
    print(
        f"[{STEP}] power_grasp_dist={thresholds['power_grasp_dist']:.4f} "
        f"lateral_pinch_dist={thresholds['lateral_pinch_dist']:.4f} "
        f"(bare defaults: {cfg.POWER_GRASP_DIST:.4f} / {cfg.LATERAL_PINCH_DIST:.4f})"
    )

    profile = save_profile(worker_id, thresholds, consent_granted)

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")
    return profile


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 00: Per-Worker Calibration")
    parser.add_argument("--worker-id", type=str, required=True, help="Worker ID")
    parser.add_argument("--video", type=str, required=True, help="Path to calibration clip")
    parser.add_argument(
        "--consent-granted", action="store_true",
        help="Confirm the worker has explicitly consented to storing this calibration profile",
    )
    args = parser.parse_args()
    run(args.worker_id, Path(args.video), args.consent_granted)


if __name__ == "__main__":
    main()
