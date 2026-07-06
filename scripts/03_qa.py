"""
DatraAI Pipeline — Step 03: Quality Assurance
Blur detection, coverage check, FPS consistency, sync drift.

Input:  processed/{session_id}/compressed.mp4
        processed/{session_id}/session.h5
Output: processed/{session_id}/qa_report.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import read_session_h5

STEP = "03_qa"


class BlurQAError(Exception):
    """Raised when blur score falls below threshold."""
    pass


def _check_blur(video_path: Path) -> dict:
    """CHECK 1 — Blur detection via Laplacian variance, sampling every 30th frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")

    scores = []
    frame_idx = 0
    frames_below = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 30 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(lap_var)
            if lap_var < cfg.BLUR_THRESHOLD:
                frames_below += 1
        frame_idx += 1

    cap.release()

    if len(scores) == 0:
        return {"score": 0.0, "passed": False, "details": {"error": "No frames sampled"}}

    mean_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    passed = mean_score >= cfg.BLUR_THRESHOLD

    return {
        "score": round(mean_score, 2),
        "passed": passed,
        "details": {
            "mean_score": round(mean_score, 2),
            "min_score": round(min_score, 2),
            "frames_below_threshold_count": frames_below,
            "frames_sampled": len(scores),
        },
    }


def _check_coverage(video_path: Path) -> dict:
    """CHECK 2 — Camera coverage via optical flow, sampling every 30th frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{STEP}] Cannot open video: {video_path}")

    flow_means = []
    static_count = 0
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 30 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mean_flow = float(np.mean(np.abs(flow)))
                flow_means.append(mean_flow)
                if mean_flow < 0.1:
                    static_count += 1
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if len(flow_means) == 0:
        return {"score": 0.0, "passed": False, "details": {"error": "No flow computed"}}

    avg_flow = float(np.mean(flow_means))
    min_flow = float(np.min(flow_means))
    passed = avg_flow >= cfg.COVERAGE_THRESHOLD

    return {
        "score": round(avg_flow, 4),
        "passed": passed,
        "details": {
            "mean_flow": round(avg_flow, 4),
            "min_flow": round(min_flow, 4),
            "static_segment_count": static_count,
            "pairs_sampled": len(flow_means),
        },
    }


def _check_fps_consistency(h5_data: dict) -> dict:
    """CHECK 3 — FPS consistency from video timestamps in HDF5."""
    timestamps = h5_data["video_timestamps"]

    # Use up to first 300 frames
    ts = timestamps[: min(300, len(timestamps))]

    if len(ts) < 2:
        return {"score": 0.0, "passed": False, "details": {"error": "Not enough frames"}}

    intervals_ms = np.diff(ts) * 1000.0
    mean_interval = float(np.mean(intervals_ms))
    std_ms = float(np.std(intervals_ms))
    max_interval = float(np.max(intervals_ms))

    # Estimate dropped frames: intervals > 1.5x nominal
    nominal_interval = 1000.0 / cfg.TARGET_FPS
    dropped = int(np.sum(intervals_ms > nominal_interval * 1.5))

    passed = std_ms <= cfg.FPS_STD_THRESHOLD_MS

    return {
        "score": round(std_ms, 4),
        "passed": passed,
        "details": {
            "mean_interval_ms": round(mean_interval, 4),
            "std_ms": round(std_ms, 4),
            "max_interval_ms": round(max_interval, 4),
            "dropped_frame_estimate": dropped,
        },
    }


def _check_sync_drift(h5_data: dict) -> dict:
    """CHECK 4 — Sync drift from HDF5 metadata."""
    metadata = h5_data.get("metadata", {})
    sync_stats = metadata.get("sync_stats", {})

    max_drift_ms = sync_stats.get("max_drift_ms", 999.0)
    passed = max_drift_ms <= cfg.SYNC_DRIFT_THRESHOLD_MS

    return {
        "score": round(max_drift_ms, 4),
        "passed": passed,
        "details": sync_stats,
    }


def run(session_id: str) -> dict:
    """
    Run all 4 QA checks and write qa_report.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    video_path = proc_dir / "compressed.mp4"
    h5_path = proc_dir / "session.h5"

    if not video_path.exists():
        raise FileNotFoundError(f"[{STEP}] compressed.mp4 not found: {video_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"[{STEP}] session.h5 not found: {h5_path}")

    h5_data = read_session_h5(h5_path)

    # Run checks
    print(f"[{STEP}] Running blur detection...")
    blur_result = _check_blur(video_path)
    print(f"[{STEP}]   Blur score: {blur_result['score']} ({'PASS' if blur_result['passed'] else 'FAIL'})")

    print(f"[{STEP}] Running coverage check...")
    coverage_result = _check_coverage(video_path)
    print(f"[{STEP}]   Coverage score: {coverage_result['score']} ({'PASS' if coverage_result['passed'] else 'FAIL'})")

    print(f"[{STEP}] Running FPS consistency check...")
    fps_result = _check_fps_consistency(h5_data)
    print(f"[{STEP}]   FPS std: {fps_result['score']}ms ({'PASS' if fps_result['passed'] else 'FAIL'})")

    print(f"[{STEP}] Running sync drift check...")
    sync_result = _check_sync_drift(h5_data)
    print(f"[{STEP}]   Max drift: {sync_result['score']}ms ({'PASS' if sync_result['passed'] else 'FAIL'})")

    # Composite score (0-100)
    # Normalize each component to [0, 1] for the composite
    blur_norm = min(blur_result["score"] / 200.0, 1.0) if blur_result["score"] > 0 else 0.0
    coverage_norm = min(coverage_result["score"] / 2.0, 1.0) if coverage_result["score"] > 0 else 0.0
    fps_norm = max(0.0, 1.0 - fps_result["score"] / 20.0) if fps_result["score"] >= 0 else 0.0
    sync_norm = max(0.0, 1.0 - sync_result["score"] / 10.0) if sync_result["score"] >= 0 else 0.0

    qa_score = (
        blur_norm * cfg.QA_BLUR_WEIGHT
        + coverage_norm * cfg.QA_COVERAGE_WEIGHT
        + fps_norm * cfg.QA_FPS_WEIGHT
        + sync_norm * cfg.QA_SYNC_WEIGHT
    ) * 100.0

    overall_passed = all([
        blur_result["passed"],
        coverage_result["passed"],
        fps_result["passed"],
        sync_result["passed"],
    ])

    qa_report = {
        "session_id": session_id,
        "overall_passed": overall_passed,
        "qa_score": round(qa_score, 1),
        "checks": {
            "blur": blur_result,
            "coverage": coverage_result,
            "fps_consistency": fps_result,
            "sync_drift": sync_result,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = proc_dir / "qa_report.json"
    with open(report_path, "w") as f:
        json.dump(qa_report, f, indent=2)

    if not overall_passed:
        failed = [k for k, v in qa_report["checks"].items() if not v["passed"]]
        print(
            f"[{STEP}] ✗ QA FAILED — checks failed: {failed}\n"
            f"  Pipeline will continue but delivery will be blocked."
        )
    else:
        print(f"[{STEP}] QA Score: {qa_score:.1f}/100")

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return qa_report


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 03: Quality Assurance")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
