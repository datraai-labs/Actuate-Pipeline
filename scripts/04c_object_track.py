"""
DatraAI Pipeline — Step 04c: Object Detection & Tracking (v2 addendum §3)

Real open-vocabulary detection + mask-based video tracking, replacing the
earlier dummy-bbox STUB:

  - Grounding DINO (HuggingFace transformers zero-shot object detection,
    config.GROUNDING_DINO_MODEL_ID) proposes object boxes from
    config.OBJECT_DETECT_PROMPT_LIST on a periodic sampling cadence
    (config.OBJECT_DETECT_SAMPLE_INTERVAL_FRAMES) rather than every frame.
  - SAM2 (transformers Sam2VideoModel, config.SAM2_MODEL_ID) propagates
    each detected box's mask across the frames between samples, one
    object per SAM2 video session (see "known API constraint" below).

Detections are restricted to within config.OBJECT_TRACK_RADIUS_NORM of
the dominant hand's landmark centroid (utils.imu_source_router-style hand
pose reading) — this tracks what the worker is handling, not the whole
scene. A re-detected box at the next sample interval is matched to an
existing track by IoU (config.OBJECT_TRACK_MATCH_IOU_MIN) to preserve
track_id continuity rather than starting a new track every sample.

Known API constraint (found via real testing, not assumed): seeding two
objects in the SAME Sam2VideoModel inference session at frame_idx=0 via
two separate add_inputs_to_inference_session calls raised an internal
transformers error ("maskmem_features in conditioning outputs cannot be
empty") — this looks like a multi-object seeding order/batching
requirement this pass didn't fully resolve. Rather than guess at the
correct multi-object call shape, each tracked object gets its OWN video
session/propagation call. This is less GPU-efficient when several
objects are near the hand at once (re-runs the SAM2 image encoder per
object instead of batching), but is the verified-working call pattern —
correctness over efficiency for this build.

NEEDS REAL GPU VALIDATION NOTES:
  - This WAS run against real session_001 footage on a real GPU (NVIDIA
    RTX 2050, confirmed via torch.cuda.is_available()) in this dev
    environment — unlike §4's monocular depth model, this is not
    untested-on-real-hardware. See docs/PIPELINE_STATUS.md §3 for the
    exact frame range covered and measured throughput.
  - Grounding DINO alone benchmarked at ~1 fps steady-state on this GPU;
    SAM2 propagation is faster per-frame once seeded. A full session's
    worth of frames (thousands) was NOT fully processed in this pass —
    see PIPELINE_STATUS.md for exactly how much real footage was
    processed and why.
  - Detection/tracking ACCURACY (does "tool" actually mean the right
    object, does the mask stay locked on through occlusion/fast motion)
    has not been validated against real, human-annotated ground truth —
    only that the models load, run, and produce plausible-shaped,
    plausible-looking output on real frames.

Input:  processed/{session_id}/compressed.mp4 or raw/{session_id}/raw.mp4
        (resolved by utils.video_utils.resolve_perception_source, v2 §8)
        processed/{session_id}/hand_pose.json
Output: processed/{session_id}/object_tracks.json — a flat JSON array
        (unchanged top-level shape from the earlier STUB, so
        scripts/05_primitives.py's `for entry in json.load(f)` and
        scripts/04d_depth_estimate.py's `object_tracks[i]` indexing both
        keep working unmodified):
        [{"frame_idx": ..., "stub": false, "tracked_objects": [
            {"track_id": ..., "class_label": ..., "confidence": ...,
             "bbox": [x1,y1,x2,y2] normalized, "centroid_norm": [x,y],
             "is_stub": false}, ...]}, ...]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.video_utils import resolve_perception_source

STEP = "04c_object_track"


# ═══════════════════════════════════════════════════════════════
# PURE LOGIC — no model/GPU dependency, unit-tested directly
# ═══════════════════════════════════════════════════════════════


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Standard IoU between two [x1, y1, x2, y2] boxes in the same units (pixels or normalized — consistent for both)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def dominant_hand_center_norm(pose_frame: Optional[dict]) -> Optional[Tuple[float, float]]:
    """Mean of the dominant hand's 21 landmarks (already normalized [0,1] by MediaPipe) — the hand's rough on-screen position for this frame."""
    if not pose_frame or not pose_frame.get("hands_detected"):
        return None
    dom = pose_frame.get("dominant_hand")
    if not dom:
        return None
    hand = pose_frame.get(f"{dom}_hand")
    if not hand or not hand.get("landmarks"):
        return None
    landmarks = hand["landmarks"]
    x = sum(lm[0] for lm in landmarks) / len(landmarks)
    y = sum(lm[1] for lm in landmarks) / len(landmarks)
    return (x, y)


def filter_detections_near_hand(
    detections: List[dict],
    hand_center_norm: Optional[Tuple[float, float]],
    radius_norm: float,
    image_width: int,
    image_height: int,
) -> List[dict]:
    """
    Keep only Grounding-DINO-style detections (`{"box": [x1,y1,x2,y2] px, "label": str, "score": float}`)
    whose box centroid (normalized) is within radius_norm of the hand.
    No hand detected this frame -> nothing is "near the hand" -> empty.
    """
    if hand_center_norm is None:
        return []
    hx, hy = hand_center_norm
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2.0 / image_width
        cy = (y1 + y2) / 2.0 / image_height
        if math.hypot(cx - hx, cy - hy) <= radius_norm:
            kept.append(det)
    return kept


def match_or_create_track_ids(
    detections_px: List[dict],
    existing_tracks: Dict[int, List[float]],
    iou_min: float,
    next_track_id: int,
) -> Tuple[List[Tuple[int, dict]], int]:
    """
    Match each new detection to an existing track by IoU (in pixel
    coordinates) so a re-detected object keeps the SAME track_id across
    sample intervals instead of starting a fresh track every time.
    Returns (list of (track_id, detection) pairs, next_track_id after
    this call — the caller's running counter).
    """
    assigned = []
    used_existing = set()
    for det in detections_px:
        best_id, best_iou = None, 0.0
        for tid, last_box in existing_tracks.items():
            if tid in used_existing:
                continue
            score = iou(det["box"], last_box)
            if score > best_iou:
                best_iou, best_id = score, tid
        if best_id is not None and best_iou >= iou_min:
            assigned.append((best_id, det))
            used_existing.add(best_id)
        else:
            assigned.append((next_track_id, det))
            next_track_id += 1
    return assigned, next_track_id


def mask_to_bbox_and_centroid(mask: np.ndarray, image_width: int, image_height: int) -> Optional[dict]:
    """Given a boolean HxW mask, return its normalized bbox + centroid, or None if the mask is empty (object tracking lost it this frame)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    cx, cy = float(xs.mean()), float(ys.mean())
    return {
        "bbox": [
            round(x1 / image_width, 6), round(y1 / image_height, 6),
            round(x2 / image_width, 6), round(y2 / image_height, 6),
        ],
        "centroid_norm": [round(cx / image_width, 6), round(cy / image_height, 6)],
    }


# ═══════════════════════════════════════════════════════════════
# MODEL-DEPENDENT — Grounding DINO + SAM2 (needs torch/transformers/GPU)
# ═══════════════════════════════════════════════════════════════

_grounding_dino_cache = {}
_sam2_cache = {}


def _get_grounding_dino():
    if "model" not in _grounding_dino_cache:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(cfg.GROUNDING_DINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.GROUNDING_DINO_MODEL_ID).to(device)
        model.eval()
        _grounding_dino_cache.update(model=model, processor=processor, device=device)
    return _grounding_dino_cache["model"], _grounding_dino_cache["processor"], _grounding_dino_cache["device"]


def _detect_objects(image_pil) -> List[dict]:
    """One Grounding DINO forward pass on a single frame. Returns [{"box": [x1,y1,x2,y2] px, "label": str, "score": float}, ...]."""
    import torch

    model, processor, device = _get_grounding_dino()
    inputs = processor(images=image_pil, text=[cfg.OBJECT_DETECT_PROMPT_LIST], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=cfg.GROUNDING_DINO_BOX_THRESHOLD,
        text_threshold=cfg.GROUNDING_DINO_TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return [
        {"box": [round(v, 2) for v in box], "label": label, "score": float(score)}
        for box, score, label in zip(results["boxes"].tolist(), results["scores"].tolist(), results["text_labels"])
    ]


def _get_sam2():
    if "model" not in _sam2_cache:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Sam2VideoProcessor.from_pretrained(cfg.SAM2_MODEL_ID)
        model = Sam2VideoModel.from_pretrained(cfg.SAM2_MODEL_ID).to(device)
        model.eval()
        _sam2_cache.update(model=model, processor=processor, device=device)
    return _sam2_cache["model"], _sam2_cache["processor"], _sam2_cache["device"]


def _track_objects_in_chunk(
    frames_pil: list,
    detections: list,
    max_objects_per_session: int,
) -> dict:
    """
    Run SAM2 video tracking for ALL detected objects in one chunk, seeding
    them into a SHARED video session so the SAM2 image encoder runs only
    ONCE per chunk regardless of how many objects are near the hand.

    Previous code called _track_object_in_chunk (one session per object),
    which re-ran the encoder for every additional object — the main source
    of 04c's 934-second runtime on 2850 frames.

    Args:
        frames_pil:            PIL images for this chunk (chunk_size frames).
        detections:            List of {"track_id": int, "box": [x1,y1,x2,y2] px,
                               "label": str, "score": float} dicts.
        max_objects_per_session: cfg.SAM2_OBJECTS_PER_SESSION.  All objects are
                               attempted in one session; if the multi-object
                               seeding raises an internal transformers error
                               (the known API constraint documented in the
                               module docstring), we fall back to one session
                               per object automatically.

    Returns:
        Dict mapping (track_id, local_frame_idx) -> np.ndarray boolean mask.
    """
    import torch
    import torch.nn.functional as F

    if not detections:
        return {}

    model, processor, device = _get_sam2()
    width, height = frames_pil[0].size
    masks_out: dict = {}  # (track_id, local_frame_idx) -> mask

    # Try to seed all objects in one session (fastest path).
    # SAM2 obj_ids must be unique ints >= 1; use the actual track_id.
    try:
        session = processor.init_video_session(
            video=frames_pil, inference_device=device, dtype=torch.float32
        )
        for det in detections:
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=det["track_id"],
                input_boxes=[[det["box"]]],
            )

        # Map SAM2's obj_id back to our track_id — they match directly here.
        tid_map = {det["track_id"]: det["track_id"] for det in detections}
        for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
            resized = F.interpolate(
                out.pred_masks.float(), size=(height, width),
                mode="bilinear", align_corners=False
            )
            for obj_idx, obj_id in enumerate(out.obj_ids):
                track_id = tid_map.get(int(obj_id))
                if track_id is None:
                    continue
                mask = (resized[obj_idx, 0] > 0.0).cpu().numpy()
                masks_out[(track_id, int(out.frame_idx))] = mask

    except Exception as multi_err:
        # Fallback: one session per object (original behaviour, slower but
        # always correct — preserves the verified-working single-object path
        # documented in the module docstring).
        print(
            f"[{STEP}]   multi-object SAM2 seeding failed ({multi_err!r}), "
            f"falling back to single-object sessions."
        )
        for det in detections:
            try:
                sess = processor.init_video_session(
                    video=frames_pil, inference_device=device, dtype=torch.float32
                )
                processor.add_inputs_to_inference_session(
                    inference_session=sess,
                    frame_idx=0,
                    obj_ids=1,
                    input_boxes=[[det["box"]]],
                )
                for out in model.propagate_in_video_iterator(sess, start_frame_idx=0):
                    resized = F.interpolate(
                        out.pred_masks.float(), size=(height, width),
                        mode="bilinear", align_corners=False
                    )
                    mask = (resized[0, 0] > 0.0).cpu().numpy()
                    masks_out[(det["track_id"], int(out.frame_idx))] = mask
            except Exception as single_err:
                print(
                    f"[{STEP}]   track_id={det['track_id']} SAM2 failed: {single_err!r} — skipping."
                )

    return masks_out


def run(session_id: str, max_frames: Optional[int] = None) -> dict:
    """
    Detect + track objects near the dominant hand across a session's
    frames and write object_tracks.json.

    max_frames: process at most this many frames (default None = the
    whole session). Exists for controlled real-data verification/smoke
    testing against a bounded, known-cost slice of a session — both
    models are real GPU inference calls, and a full session is thousands
    of frames (see docs/PIPELINE_STATUS.md §3 for measured throughput).
    Production runs should omit this.
    """
    import cv2
    from PIL import Image

    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    video_path = resolve_perception_source(session_id)
    pose_path = proc_dir / "hand_pose.json"

    if not pose_path.exists():
        raise FileNotFoundError(f"[{STEP}] hand_pose.json not found: {pose_path}")

    with open(pose_path) as f:
        pose_data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total_frames, len(pose_data))
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    frame_results: Dict[int, List[dict]] = {i: [] for i in range(n_frames)}
    active_tracks: Dict[int, List[float]] = {}  # track_id -> last known px bbox
    track_labels: Dict[int, str] = {}
    next_track_id = 1

    chunk_size = cfg.OBJECT_DETECT_SAMPLE_INTERVAL_FRAMES
    n_chunks = (n_frames + chunk_size - 1) // max(1, chunk_size)
    print(f"[{STEP}] Processing {n_frames} frames in {n_chunks} chunk(s) of up to {chunk_size} frames")

    for chunk_start in range(0, n_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        frames_pil = []
        for i in range(chunk_start, chunk_end):
            ret, frame = cap.read()
            if not ret:
                break
            frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames_pil:
            break
        width, height = frames_pil[0].size

        # Re-detect at this chunk's first frame, restricted to near the hand.
        pose_frame = pose_data[chunk_start] if chunk_start < len(pose_data) else None
        hand_center = dominant_hand_center_norm(pose_frame)
        raw_detections = _detect_objects(frames_pil[0])
        near_hand = filter_detections_near_hand(raw_detections, hand_center, cfg.OBJECT_TRACK_RADIUS_NORM, width, height)

        assigned, next_track_id = match_or_create_track_ids(
            near_hand, active_tracks, cfg.OBJECT_TRACK_MATCH_IOU_MIN, next_track_id
        )

        if assigned:
            # Build the detection list expected by _track_objects_in_chunk.
            dets_for_tracking = [
                {"track_id": tid, "box": det["box"], "label": det["label"], "score": det["score"]}
                for tid, det in assigned
            ]
            # Batch all objects into one SAM2 session per chunk.
            masks_all = _track_objects_in_chunk(
                frames_pil,
                dets_for_tracking,
                max_objects_per_session=cfg.SAM2_OBJECTS_PER_SESSION,
            )

            for track_id, det in assigned:
                track_labels[track_id] = det["label"]
                last_box = det["box"]
                for local_idx in range(len(frames_pil)):
                    mask = masks_all.get((track_id, local_idx))
                    if mask is None:
                        continue
                    global_idx = chunk_start + local_idx
                    result = mask_to_bbox_and_centroid(mask, width, height)
                    if result is None:
                        continue
                    last_box = [
                        result["bbox"][0] * width, result["bbox"][1] * height,
                        result["bbox"][2] * width, result["bbox"][3] * height,
                    ]
                    frame_results[global_idx].append({
                        "track_id": track_id,
                        "class_label": track_labels[track_id],
                        "confidence": round(det["score"], 4),
                        "bbox": result["bbox"],
                        "centroid_norm": result["centroid_norm"],
                        "is_stub": False,
                    })
                active_tracks[track_id] = last_box

        print(f"[{STEP}]   chunk {chunk_start}-{chunk_end - 1}: {len(assigned)} object(s) near hand")

    cap.release()

    # Flat list at the JSON top level — matches the earlier STUB's shape
    # exactly, so 05_primitives.py's `for entry in json.load(f)` and
    # 04d_depth_estimate.py's `object_tracks[i]` positional indexing both
    # keep working with no changes needed there. "stub" is now False on
    # every frame — real detection, not a placeholder — so any code still
    # checking that flag reads the correct, non-stub state rather than
    # trusting stub data or KeyError-ing on a missing field.
    output = [
        {"frame_idx": i, "stub": False, "tracked_objects": frame_results.get(i, [])}
        for i in range(n_frames)
    ]

    output_path = proc_dir / "object_tracks.json"
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    elapsed = time.time() - t0
    n_with_objects = sum(1 for f in output if f["tracked_objects"])
    print(f"[{STEP}] {n_with_objects}/{n_frames} frames have a tracked object")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")
    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 04c: Object Detection & Tracking")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
