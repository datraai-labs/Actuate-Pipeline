"""
DatraAI Pipeline — Step 03b: Privacy / PII Redaction (v2 addendum §10)

Produces a REDACTED COPY of the video for delivery. Never touches
raw.mp4 (stays in secure internal storage, never included in delivery/) and
never overwrites compressed.mp4 (perception stages — 04_hand_pose.py,
04c_object_track.py, 04d_depth_estimate.py — keep reading the unredacted
compressed.mp4 for maximum tracking fidelity; blurred regions are a risk to
hand/object-tracking accuracy we don't need to take internally). Only
scripts/11_package.py substitutes redacted_compressed.mp4 into the
delivered bundle — see that script for the fail-closed check ensuring an
unredacted video can never ship.

Two real, working detectors, both verified end-to-end against the real
session_001 video (2850 frames, 1920x1080), not just unit-tested:

  - Face detection: MediaPipe Face Detection (already a project dependency).
    Verified run: 192/192 detected faces redacted. Every detected face is
    treated as an unidentified bystander UNLESS a
    processed/{session_id}/worker_reference_face.jpg is present, in which
    case a coarse HSV-histogram-correlation similarity check
    (NOT real face recognition/embedding — see PRIVACY_WORKER_FACE_MATCH_THRESHOLD
    in config.py) flags matches as the primary worker instead. No
    worker-enrollment system exists yet in this pipeline, so in practice
    today every detected face is redacted under REDACT_BYSTANDER_FACES.

  - Text/badge detection: EasyOCR (installed and verified in this
    environment — see requirements.txt). Verified run against session_001:
    93 raw detections (27 ignored below PRIVACY_OCR_MIN_CONFIDENCE, 22
    blurred-and-flagged-for-review, 44 confidently blurred); all 22 flagged
    detections carry frame_idx/bbox/confidence in privacy_report.json's
    "flagged_regions" for human follow-up. First run on a fresh machine
    downloads model weights (needs network access) — GPU recommended for
    full-session throughput (this environment is CPU-only: ~244s for one
    2850-frame 1080p session with both detectors enabled), CPU works but is
    slower. Re-verify throughput on your actual target hardware.

Everything else in this file — blur application, histogram comparison,
confidence tiering, the cross-sample region tracker — is pure OpenCV/numpy
and covered by tests/test_privacy_redact.py with synthetic images, no
model or video required.

Input:  processed/{session_id}/compressed.mp4
        processed/{session_id}/worker_reference_face.jpg (optional)
        processed/{session_id}/session_meta.json (read-only, for consent_status
            in the report — the actual delivery-blocking gate lives in
            run_pipeline.py, not here)
Output: processed/{session_id}/redacted_compressed.mp4
        processed/{session_id}/privacy_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

STEP = "03b_privacy_redact"

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


# ═══════════════════════════════════════════════════════════════
# PURE FUNCTIONS — no ML, no GPU, fully unit-testable
# ═══════════════════════════════════════════════════════════════


def _blur_region(frame: np.ndarray, bbox: BBox, kernel_size: int) -> np.ndarray:
    """
    Gaussian-blur a rectangular region of a BGR frame in place, clamped to
    the frame's bounds. kernel_size is forced odd (Gaussian blur requires it).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k = max(3, min(k, (x2 - x1) | 1, (y2 - y1) | 1))
    if k < 3:
        return frame

    region = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return frame


def _compute_reference_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """
    HSV hue/saturation histogram for a reference face image, normalized —
    the coarse similarity signal used by _face_similarity(). Not a face
    embedding; a per-crop color/texture summary only.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _face_similarity(face_crop_bgr: np.ndarray, reference_hist: Optional[np.ndarray]) -> float:
    """
    Coarse similarity in [-1, 1] (cv2.HISTCMP_CORREL) between a detected
    face crop and a reference photo's histogram. Returns 0.0 (no match) if
    no reference is available or the crop is empty.

    ⚠️ This is a color/texture heuristic, NOT face recognition — two
    different people with similar skin tone/lighting can score highly
    similar. Adequate as a forward-compatible hook; replace with a real
    face-embedding model before using this to distinguish individuals in
    a way anyone should rely on.
    """
    if reference_hist is None or face_crop_bgr.size == 0:
        return 0.0
    crop_hist = _compute_reference_histogram(face_crop_bgr)
    return float(cv2.compareHist(crop_hist, reference_hist, cv2.HISTCMP_CORREL))


def classify_text_confidence(confidence: float) -> str:
    """
    Three-tier handling of an OCR detection's confidence:
      "ignore"        — below PRIVACY_OCR_MIN_CONFIDENCE, too weak to act on
      "blur_and_flag" — ambiguous middle band: blur (fail-safe) AND flag
                         the frame for human review
      "blur"          — confidently a real text region, blur, no flag needed
    """
    if confidence < cfg.PRIVACY_OCR_MIN_CONFIDENCE:
        return "ignore"
    if confidence < cfg.PRIVACY_OCR_BLUR_CONFIDENCE:
        return "blur_and_flag"
    return "blur"


class TextRegionTracker:
    """
    Carries OCR-detected text-region bboxes forward across the frames
    between OCR samples (OCR only runs every
    config.PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES frames — full-frame OCR every
    frame would be far too slow) plus a short grace period after a region
    stops being redetected, so a momentary OCR miss doesn't instantly
    un-blur a real badge.
    """

    def __init__(self, grace_frames: int):
        self._grace_frames = grace_frames
        self._active: dict = {}  # bbox -> last_seen_frame

    def update(self, detected_bboxes: List[BBox], current_frame: int) -> None:
        for bbox in detected_bboxes:
            self._active[bbox] = current_frame

    def prune(self, current_frame: int) -> None:
        self._active = {
            bbox: last_seen
            for bbox, last_seen in self._active.items()
            if current_frame - last_seen <= self._grace_frames
        }

    def active_bboxes(self) -> List[BBox]:
        return list(self._active.keys())


def _clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> BBox:
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


# ═══════════════════════════════════════════════════════════════
# MODEL-DEPENDENT — isolated so tests never need to load them
# ═══════════════════════════════════════════════════════════════

_FACE_DETECTOR = None
_OCR_READER = None


def _get_face_detector():
    """
    MediaPipe Face Detection — confirmed installed and working in this
    environment (unlike the OCR/depth-model dependencies elsewhere in the
    v2 build). Still lazily imported to match the project's convention for
    heavy dependencies.
    """
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        import mediapipe as mp

        _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=cfg.PRIVACY_FACE_DETECTION_MIN_CONFIDENCE,
        )
    return _FACE_DETECTOR


def _detect_faces(detector, frame_bgr: np.ndarray) -> list:
    """Returns [{"bbox": (x1,y1,x2,y2), "confidence": float}, ...] in pixel coords."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    faces = []
    if results.detections:
        for det in results.detections:
            rbb = det.location_data.relative_bounding_box
            bbox = _clamp_bbox(
                rbb.xmin * w, rbb.ymin * h,
                (rbb.xmin + rbb.width) * w, (rbb.ymin + rbb.height) * h,
                w, h,
            )
            confidence = float(det.score[0]) if det.score else 0.0
            faces.append({"bbox": bbox, "confidence": confidence})
    return faces


def _get_ocr_reader():
    """
    EasyOCR reader. Verified working in this environment, including a full
    run against the real session_001 video (see module docstring) — this is
    real, exercised code, not an unverified integration.

    First call downloads model weights (needs network access, ~20s in this
    environment); GPU recommended for full-session throughput on larger
    batches — CPU works (this environment is CPU-only) but is the slowest
    part of the pipeline (~244s for one 2850-frame 1080p session with both
    face and text detection enabled). Re-benchmark on your actual target
    hardware before assuming this throughput at scale.
    """
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr  # heavy, lazy on purpose — see docstring above

        _OCR_READER = easyocr.Reader(["en"])
    return _OCR_READER


def _run_ocr(reader, frame_bgr: np.ndarray) -> list:
    """Returns [{"bbox": (x1,y1,x2,y2), "confidence": float}, ...] in pixel coords."""
    results = reader.readtext(frame_bgr)
    detections = []
    for (points, _text, confidence) in results:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = _clamp_bbox(min(xs), min(ys), max(xs), max(ys), frame_bgr.shape[1], frame_bgr.shape[0])
        detections.append({"bbox": bbox, "confidence": float(confidence)})
    return detections


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════


def _check_ocr_propagation_invariant() -> None:
    """
    PRIVACY_OCR_PROPAGATION_FRAMES must be >= PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES - 1,
    or a text region confirmed at one OCR sample expires before the next
    sample gets a chance to redetect it — a real gap of unblurred frames.
    Enforced here (not just documented in config.py) so a future config
    edit can't silently reintroduce that gap.
    """
    if not cfg.REDACT_TEXT_AND_BADGES:
        return
    minimum_required = cfg.PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES - 1
    if cfg.PRIVACY_OCR_PROPAGATION_FRAMES < minimum_required:
        raise ValueError(
            f"[{STEP}] config error: PRIVACY_OCR_PROPAGATION_FRAMES="
            f"{cfg.PRIVACY_OCR_PROPAGATION_FRAMES} is less than "
            f"PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES - 1 ({minimum_required}) — text "
            f"regions would expire between OCR samples, leaving frames unredacted. "
            f"Raise PRIVACY_OCR_PROPAGATION_FRAMES to at least {minimum_required}."
        )


def run(session_id: str) -> dict:
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")
    _check_ocr_propagation_invariant()

    proc_dir = cfg.PROCESSED_DIR / session_id
    video_path = proc_dir / "compressed.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"[{STEP}] compressed.mp4 not found: {video_path}")

    meta_path = proc_dir / "session_meta.json"
    consent_status = "pending"
    if meta_path.exists():
        with open(meta_path) as f:
            consent_status = json.load(f).get("consent_status", "pending")

    reference_hist = None
    reference_path = proc_dir / "worker_reference_face.jpg"
    worker_reference_available = reference_path.exists()
    if worker_reference_available:
        ref_img = cv2.imread(str(reference_path))
        if ref_img is not None:
            reference_hist = _compute_reference_histogram(ref_img)
        print(f"[{STEP}] Worker reference face loaded — primary-worker matching active")
    else:
        print(
            f"[{STEP}] No worker_reference_face.jpg — every detected face is treated as an "
            f"unidentified bystander (REDACT_PRIMARY_WORKER_FACE has no effect this run)."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.TARGET_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{STEP}] {total_frames} frames at {fps:.1f}fps, {width}x{height}")

    face_detector = _get_face_detector()
    ocr_reader = _get_ocr_reader() if cfg.REDACT_TEXT_AND_BADGES else None
    if cfg.REDACT_TEXT_AND_BADGES:
        print(f"[{STEP}] Text/badge redaction (EasyOCR) enabled — this is the slowest "
              f"part of the pipeline; expect it to dominate runtime on longer sessions.")

    output_path = proc_dir / "redacted_compressed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"[{STEP}] Cannot open VideoWriter for: {output_path}")

    text_tracker = TextRegionTracker(grace_frames=cfg.PRIVACY_OCR_PROPAGATION_FRAMES)

    faces_detected_total = 0
    faces_redacted = 0
    text_regions_redacted = 0  # per-frame blur applications (propagated), not unique detections
    frames_flagged_for_review = 0
    # Raw OCR-detection tallies (once per detection at sample time, not
    # inflated by propagation across carried-forward frames) — this is what
    # actually answers "how many text regions were detected and how did
    # they split by confidence tier".
    text_detections_ignored = 0
    text_detections_blur_and_flag = 0
    text_detections_confident_blur = 0
    # Actionable trace for every ambiguous detection — frame, region,
    # confidence — so a human can actually find and judge it, not just see
    # a bare count. See privacy_report.json's "flagged_regions".
    flagged_regions = []
    report_interval = max(1, total_frames // 20) if total_frames > 0 else 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ─── Faces ────────────────────────────────────────────
        faces = _detect_faces(face_detector, frame)
        for face in faces:
            faces_detected_total += 1
            similarity = _face_similarity(
                frame[face["bbox"][1]:face["bbox"][3], face["bbox"][0]:face["bbox"][2]],
                reference_hist,
            )
            is_worker = reference_hist is not None and similarity >= cfg.PRIVACY_WORKER_FACE_MATCH_THRESHOLD

            should_redact = cfg.REDACT_PRIMARY_WORKER_FACE if is_worker else cfg.REDACT_BYSTANDER_FACES
            if should_redact:
                _blur_region(frame, face["bbox"], cfg.PRIVACY_BLUR_KERNEL_SIZE)
                faces_redacted += 1

        # ─── Text / badges ────────────────────────────────────
        if cfg.REDACT_TEXT_AND_BADGES:
            frame_flagged = False
            if frame_idx % cfg.PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES == 0:
                detections = _run_ocr(ocr_reader, frame)
                keep_bboxes = []
                for det in detections:
                    tier = classify_text_confidence(det["confidence"])
                    if tier == "ignore":
                        text_detections_ignored += 1
                        continue
                    keep_bboxes.append(det["bbox"])
                    if tier == "blur_and_flag":
                        text_detections_blur_and_flag += 1
                        frame_flagged = True
                        flagged_regions.append({
                            "frame_idx": frame_idx,
                            "bbox": list(det["bbox"]),
                            "confidence": round(det["confidence"], 4),
                        })
                    else:
                        text_detections_confident_blur += 1
                text_tracker.update(keep_bboxes, frame_idx)

            text_tracker.prune(frame_idx)
            for bbox in text_tracker.active_bboxes():
                _blur_region(frame, bbox, cfg.PRIVACY_BLUR_KERNEL_SIZE)
                text_regions_redacted += 1
            if frame_flagged:
                frames_flagged_for_review += 1

        writer.write(frame)
        frame_idx += 1
        if frame_idx % report_interval == 0:
            pct = frame_idx / total_frames * 100 if total_frames else 0.0
            print(f"[{STEP}] Progress: {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    writer.release()
    face_detector.close()

    privacy_report = {
        "session_id": session_id,
        "faces_detected_total": faces_detected_total,
        "faces_redacted": faces_redacted,
        "frames_flagged_for_review": frames_flagged_for_review,
        "text_regions_redacted": text_regions_redacted,
        "text_detection_tiers": {
            # Raw OCR-detection counts (once per sample-time detection, NOT
            # inflated by cross-frame propagation) — this is the actual
            # "how many text regions, split by confidence tier" answer.
            "ignored_below_min_confidence": text_detections_ignored,
            "blurred_and_flagged_for_review": text_detections_blur_and_flag,
            "confidently_blurred": text_detections_confident_blur,
        },
        # Actionable trace for every ambiguous ("blur_and_flag" tier)
        # detection — frame_idx + bbox + confidence — so a human reviewer
        # can actually locate and judge it, not just see a count.
        "flagged_regions": flagged_regions,
        "consent_status": consent_status,
        "worker_reference_face_available": worker_reference_available,
        "redaction_config": {
            "redact_bystander_faces": cfg.REDACT_BYSTANDER_FACES,
            "redact_primary_worker_face": cfg.REDACT_PRIMARY_WORKER_FACE,
            "redact_text_and_badges": cfg.REDACT_TEXT_AND_BADGES,
        },
    }

    report_path = proc_dir / "privacy_report.json"
    with open(report_path, "w") as f:
        json.dump(privacy_report, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{STEP}] Faces: {faces_detected_total} detected, {faces_redacted} redacted")
    if cfg.REDACT_TEXT_AND_BADGES:
        print(
            f"[{STEP}] Text detections — ignored: {text_detections_ignored}, "
            f"blurred+flagged: {text_detections_blur_and_flag}, "
            f"confidently blurred: {text_detections_confident_blur}"
        )
        print(f"[{STEP}] Frames flagged for review: {frames_flagged_for_review} (see privacy_report.json's flagged_regions)")
    print(f"[{STEP}] Wrote {output_path}")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return privacy_report


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 03b: Privacy / PII Redaction")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    # Only needed for standalone invocation (python scripts/03b_privacy_redact.py ...) —
    # run_pipeline.py already installs this globally before loading any step module.
    from utils.console_safety import install as _install_console_safety

    _install_console_safety()
    main()
