"""
DatraAI Pipeline — Step 04d: Metric 3D / Depth Estimation (v2 addendum §4)

Lifts 2D hand-pose landmarks and object-track centroids into metric 3D
(meters) using per-frame depth + per-device camera intrinsics. This is a
hard prerequisite for any future retargeting engine — sessions processed
with DEPTH_MODE == "none" are flagged "retargeting_eligible": false in
quality_certificate.json (see scripts/10_eis.py) and excluded from
"VLA_finetuning" in recommended_use unless config.ALLOW_2D_ONLY_VLA_FINETUNING
is explicitly set.

⚠️ WHAT NEEDS REAL GPU VALIDATION BEFORE RELYING ON THIS (Kaggle/Lightning):
  - _get_monocular_pipeline() / _estimate_monocular_depth_frame(): the
    monocular depth-estimation model call. Not run against real hardware in
    this environment (no GPU / model download available here). Before
    trusting it: confirm config.MONOCULAR_DEPTH_MODEL_ID is still current
    and available, confirm torch/transformers versions in requirements.txt
    resolve cleanly for it, and benchmark per-frame throughput — this is
    likely the slowest stage in the whole pipeline.
  - _read_stereo_depth_stream(): the raw/{session_id}/depth.raw binary
    format is a documented convention, not verified against any real
    stereo-rig output. Adjust to match your actual hardware's format.

Everything else in this file (camera-intrinsics math, pinhole 2D->3D
lift, keypoint depth sampling, orchestration) is pure Python/numpy,
GPU-free, and covered by tests/test_depth_estimate.py with synthetic
depth maps — no model or video required.

Input:  processed/{session_id}/hand_pose.json
        processed/{session_id}/object_tracks.json (optional — v2 addendum §3;
            produced by scripts/04c_object_track.py's real Grounding DINO +
            SAM2 detection/tracking)
        processed/{session_id}/session_meta.json
        processed/{session_id}/session.h5
        processed/{session_id}/compressed.mp4 or raw/{session_id}/raw.mp4
            (resolved by utils.video_utils.resolve_perception_source, v2 §8;
            only read if depth_mode_effective == "monocular_estimated")
        raw/{session_id}/depth.raw (optional — stereo depth stream)
        calibration/{device_id}_intrinsics.json (optional — falls back to
            an approximated pinhole model if absent)
Output: processed/{session_id}/depth_data.json
        processed/{session_id}/hand_pose_3d.json
        processed/{session_id}/object_tracks.json (enriched in place with
            "centroid_3d_m", per addendum §4)
        processed/{session_id}/depth_maps/*.npz (only if
            DEPTH_STORAGE_MODE == "dense" AND ENABLE_DENSE_DEPTH_MAPS)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import read_session_h5
from utils.video_utils import resolve_perception_source

STEP = "04d_depth_estimate"

# MediaPipe landmark indices for the keypoints depth_data.json reports by name.
_KEYPOINT_LANDMARKS = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}


# ═══════════════════════════════════════════════════════════════
# PURE GEOMETRY — no GPU, no model, fully unit-testable
# ═══════════════════════════════════════════════════════════════


def _load_camera_intrinsics(device_id: str, width: int, height: int) -> dict:
    """
    Load per-device camera intrinsics from
    config.CAMERA_INTRINSICS_PATH.format(device_id=device_id) (e.g.
    calibration/{device_id}_intrinsics.json), expected shape:
        {"fx": ..., "fy": ..., "cx": ..., "cy": ...}

    Falls back to an approximated pinhole model derived from the video
    resolution + config.CAMERA_DEFAULT_HFOV_DEG if no calibration file
    exists. The fallback is an APPROXIMATION, not real calibration data —
    3D coordinates derived from it should not be trusted for anything
    requiring true metric precision until a real calibration file is
    supplied. The returned "source" field records which path was taken so
    downstream consumers can tell.
    """
    intrinsics_path = (
        Path(__file__).resolve().parent.parent / cfg.CAMERA_INTRINSICS_PATH.format(device_id=device_id)
    )
    if intrinsics_path.exists():
        with open(intrinsics_path) as f:
            data = json.load(f)
        return {
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "source": "calibration_file",
        }

    hfov_rad = math.radians(cfg.CAMERA_DEFAULT_HFOV_DEG)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    return {
        "fx": fx,
        "fy": fx,  # assume square pixels
        "cx": width / 2.0,
        "cy": height / 2.0,
        "source": "approximated_no_calibration_file",
    }


def _sample_depth_at_normalized_point(depth_map: Optional[np.ndarray], x_norm: float, y_norm: float) -> Optional[float]:
    """
    Nearest-pixel depth sample (meters) at a normalized [0,1] image
    coordinate. Returns None if the depth map is unavailable or the sampled
    value is non-positive/non-finite (common at depth-map edges/holes).
    """
    if depth_map is None:
        return None
    h, w = depth_map.shape[:2]
    if h == 0 or w == 0:
        return None
    px = int(round(x_norm * (w - 1)))
    py = int(round(y_norm * (h - 1)))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    value = float(depth_map[py, px])
    if value <= 0 or not math.isfinite(value):
        return None
    return value


def _lift_point_to_3d(
    x_norm: float,
    y_norm: float,
    depth_m: Optional[float],
    intrinsics: dict,
    width: int,
    height: int,
) -> Optional[list]:
    """
    Standard pinhole back-projection: normalized image coords + metric
    depth + intrinsics -> [X, Y, Z] in meters, camera-centered.
    """
    if depth_m is None:
        return None
    px = x_norm * width
    py = y_norm * height
    x = (px - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (py - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return [round(float(x), 5), round(float(y), 5), round(float(depth_m), 5)]


def _ego_motion_scale_hint(head_gyro_window: np.ndarray, head_accel_window: np.ndarray) -> float:
    """
    Hook for resolving monocular depth scale ambiguity from head-IMU-derived
    ego-motion (v2 addendum §4: "use head-IMU-derived ego-motion to help
    resolve scale ambiguity across frames where possible").

    Currently a documented no-op returning 1.0: config.MONOCULAR_DEPTH_MODEL_ID
    is a METRIC depth-estimation checkpoint, so its output is already in
    real meters and there is no scale ambiguity to resolve for the default
    configuration. This hook is kept in the call path so that if a
    RELATIVE-depth model is substituted later, a real ego-motion-based
    scale-recovery algorithm (visual-inertial alignment) can be implemented
    here without restructuring the caller. Implementing that recovery
    properly is a substantial project in its own right and is explicitly
    out of scope for this pass.
    """
    return 1.0


def _read_stereo_depth_stream(
    raw_session_path: Path, n_frames: int, height: int, width: int
) -> Optional[np.ndarray]:
    """
    Reads a calibrated stereo depth stream if present, returning a
    (n_frames, height, width) float32 array in meters, or None if absent
    (caller falls back to monocular_estimated).

    Documented format (no real stereo-rig sample data was available to
    verify this against): raw/{session_id}/depth.raw — a flat binary array,
    row-major, dtype float32 by default, shape (n_frames, height, width).
    An optional raw/{session_id}/depth_meta.json can override
    height/width/dtype:
        {"height": ..., "width": ..., "dtype": "float32"}

    ⚠️ NEEDS VALIDATION against your actual stereo hardware's real output
    format — adjust this reader if it differs.
    """
    depth_raw_path = raw_session_path / "depth.raw"
    if not depth_raw_path.exists():
        return None

    expected_h, expected_w, dtype_str = height, width, "float32"
    meta_path = raw_session_path / "depth_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        expected_h = int(meta.get("height", expected_h))
        expected_w = int(meta.get("width", expected_w))
        dtype_str = meta.get("dtype", dtype_str)

    dtype = np.dtype(dtype_str)
    expected_count = n_frames * expected_h * expected_w
    raw = np.fromfile(str(depth_raw_path), dtype=dtype)

    if raw.size < expected_count:
        print(
            f"[{STEP}] WARNING: {depth_raw_path} has {raw.size} values, expected >= "
            f"{expected_count} for shape ({n_frames},{expected_h},{expected_w}) — "
            f"falling back to monocular_estimated."
        )
        return None

    return raw[:expected_count].reshape((n_frames, expected_h, expected_w)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# GPU-DEPENDENT — needs real hardware validation (see module docstring)
# ═══════════════════════════════════════════════════════════════

_MONOCULAR_PIPELINE = None


def _get_monocular_pipeline():
    """
    Lazily loads the metric monocular depth-estimation pipeline
    (config.MONOCULAR_DEPTH_MODEL_ID) via HuggingFace `transformers`.

    ⚠️ NEEDS REAL GPU VALIDATION — see module docstring.
    """
    global _MONOCULAR_PIPELINE
    if _MONOCULAR_PIPELINE is None:
        from transformers import pipeline  # heavy import — lazy on purpose

        _MONOCULAR_PIPELINE = pipeline(task="depth-estimation", model=cfg.MONOCULAR_DEPTH_MODEL_ID)
    return _MONOCULAR_PIPELINE


def _estimate_monocular_depth_frame(pipe, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Run the monocular depth model on one BGR video frame. Returns a (H, W)
    float32 depth map in meters, resized to the source frame's resolution
    so pixel coordinates line up with normalized hand-pose/object-track
    landmarks.

    ⚠️ NEEDS REAL GPU VALIDATION — see module docstring.
    """
    from PIL import Image  # lazy — only needed on this path

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    result = pipe(image)
    depth = result["predicted_depth"]
    depth_np = depth.squeeze().cpu().numpy() if hasattr(depth, "cpu") else np.asarray(depth)

    if depth_np.shape[:2] != frame_bgr.shape[:2]:
        depth_np = cv2.resize(
            depth_np, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return depth_np.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════


def run(session_id: str) -> dict:
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    raw_session_path = cfg.RAW_DIR / session_id

    pose_path = proc_dir / "hand_pose.json"
    tracks_path = proc_dir / "object_tracks.json"
    meta_path = proc_dir / "session_meta.json"
    h5_path = proc_dir / "session.h5"

    for p in (pose_path, meta_path, h5_path):
        if not p.exists():
            raise FileNotFoundError(f"[{STEP}] Required file not found: {p}")

    with open(pose_path) as f:
        pose_data = json.load(f)
    with open(meta_path) as f:
        session_meta = json.load(f)

    object_tracks = None
    if tracks_path.exists():
        with open(tracks_path) as f:
            object_tracks = json.load(f)
    else:
        print(f"[{STEP}] object_tracks.json not found — centroid_3d_m will be unavailable this run.")

    width = int(session_meta.get("video_width") or 0) or 1280
    height = int(session_meta.get("video_height") or 0) or 720
    device_id = session_meta.get("device_id", "default")

    intrinsics = _load_camera_intrinsics(device_id, width, height)
    print(f"[{STEP}] Camera intrinsics source: {intrinsics['source']} (device_id={device_id!r})")
    if intrinsics["source"] != "calibration_file":
        print(
            f"[{STEP}] ⚠ No calibration file at "
            f"{cfg.CAMERA_INTRINSICS_PATH.format(device_id=device_id)} — using an "
            f"approximated pinhole model (HFOV={cfg.CAMERA_DEFAULT_HFOV_DEG}°). Metric "
            f"3D from this session should not be trusted for anything requiring true "
            f"calibration."
        )

    h5_data = read_session_h5(h5_path)
    accel = h5_data["accel"]
    gyro = h5_data["gyro"]

    n_frames = len(pose_data)
    depth_mode_requested = cfg.DEPTH_MODE
    depth_mode_effective = depth_mode_requested
    stereo_depth = None

    if depth_mode_requested == "stereo":
        stereo_depth = _read_stereo_depth_stream(raw_session_path, n_frames, height, width)
        depth_mode_effective = "stereo" if stereo_depth is not None else "monocular_estimated"
        if stereo_depth is None:
            print(f"[{STEP}] No stereo depth stream at {raw_session_path}/depth.raw — falling back to monocular_estimated.")
    elif depth_mode_requested not in ("monocular_estimated", "none"):
        raise ValueError(f"[{STEP}] Unknown DEPTH_MODE: {depth_mode_requested!r}")

    print(f"[{STEP}] depth_mode_effective={depth_mode_effective!r}")
    depth_confidence = cfg.DEPTH_CONFIDENCE_MULTIPLIER.get(depth_mode_effective, 0.0)

    monocular_pipe = None
    video_cap = None
    if depth_mode_effective == "monocular_estimated":
        print(
            f"[{STEP}] ⚠ NEEDS REAL GPU VALIDATION: loading monocular depth model "
            f"{cfg.MONOCULAR_DEPTH_MODEL_ID!r}. Not run against real hardware in this "
            f"environment — see module docstring before trusting this on Kaggle/Lightning."
        )
        monocular_pipe = _get_monocular_pipeline()
        video_path = resolve_perception_source(session_id)
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")

    write_dense = (
        depth_mode_effective != "none"
        and cfg.DEPTH_STORAGE_MODE == "dense"
        and cfg.ENABLE_DENSE_DEPTH_MAPS
    )
    dense_dir = proc_dir / "depth_maps"
    if write_dense:
        dense_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{STEP}] Dense depth maps enabled — writing to {dense_dir}")

    depth_data_out = []
    hand_pose_3d_out = []
    frames_lifted = 0

    for i in range(n_frames):
        pose_frame = pose_data[i] if i < len(pose_data) else None
        object_frame = object_tracks[i] if object_tracks and i < len(object_tracks) else None

        depth_map = None
        if depth_mode_effective == "stereo":
            depth_map = stereo_depth[i]
        elif depth_mode_effective == "monocular_estimated":
            ret, frame_bgr = video_cap.read()
            if ret:
                depth_map = _estimate_monocular_depth_frame(monocular_pipe, frame_bgr)
                w_start, w_end = max(0, i - 2), min(n_frames, i + 3)
                _ego_motion_scale_hint(gyro[w_start:w_end], accel[w_start:w_end])

        keypoint_depths = {}
        landmarks_3d = None
        dominant_hand = None

        if pose_frame and pose_frame.get("hands_detected") and depth_map is not None:
            dominant_hand = pose_frame.get("dominant_hand")
            hand = pose_frame.get(f"{dominant_hand}_hand") if dominant_hand else None
            if hand and hand.get("landmarks"):
                landmarks = hand["landmarks"]
                for name, idx in _KEYPOINT_LANDMARKS.items():
                    if idx < len(landmarks):
                        depth_m = _sample_depth_at_normalized_point(depth_map, landmarks[idx][0], landmarks[idx][1])
                        if depth_m is not None:
                            keypoint_depths[name] = round(depth_m, 4)

                landmarks_3d = []
                for lm in landmarks:
                    depth_m = _sample_depth_at_normalized_point(depth_map, lm[0], lm[1])
                    landmarks_3d.append(_lift_point_to_3d(lm[0], lm[1], depth_m, intrinsics, width, height))
                frames_lifted += 1

        # Enrich object_tracks.json entries with centroid_3d_m in place, per addendum §4.
        if object_frame is not None and depth_map is not None:
            for obj in object_frame.get("tracked_objects", []):
                centroid = obj.get("centroid_norm")
                if centroid and len(centroid) >= 2:
                    depth_m = _sample_depth_at_normalized_point(depth_map, centroid[0], centroid[1])
                    obj["centroid_3d_m"] = _lift_point_to_3d(centroid[0], centroid[1], depth_m, intrinsics, width, height)

        depth_data_out.append({
            "frame_idx": i,
            "depth_mode": depth_mode_effective,
            "depth_confidence": depth_confidence,
            "keypoint_depths_m": keypoint_depths,
        })
        hand_pose_3d_out.append({
            "frame_idx": i,
            "hands_detected": bool(pose_frame.get("hands_detected")) if pose_frame else False,
            "dominant_hand": dominant_hand,
            "landmarks_3d_m": landmarks_3d,
            "depth_mode": depth_mode_effective,
            "depth_confidence": depth_confidence,
        })

        if write_dense and depth_map is not None:
            np.savez_compressed(dense_dir / f"{i:06d}.npz", depth=depth_map)

    if video_cap is not None:
        video_cap.release()

    depth_data_path = proc_dir / "depth_data.json"
    with open(depth_data_path, "w") as f:
        json.dump(depth_data_out, f, separators=(",", ":"))

    hand_pose_3d_path = proc_dir / "hand_pose_3d.json"
    with open(hand_pose_3d_path, "w") as f:
        json.dump(hand_pose_3d_out, f, separators=(",", ":"))

    if object_tracks is not None:
        with open(tracks_path, "w") as f:
            json.dump(object_tracks, f, separators=(",", ":"))
        print(f"[{STEP}] Enriched object_tracks.json in place with centroid_3d_m")

    print(f"[{STEP}] Lifted 3D landmarks for {frames_lifted}/{n_frames} frames")

    elapsed = time.time() - t0
    print(f"[{STEP}] Wrote depth_data.json + hand_pose_3d.json")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return {
        "session_id": session_id,
        "depth_mode_requested": depth_mode_requested,
        "depth_mode_effective": depth_mode_effective,
        "depth_confidence": depth_confidence,
        "retargeting_eligible": depth_mode_effective != "none",
    }


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 04d: Metric 3D / Depth Estimation")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
