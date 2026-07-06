"""
DatraAI Pipeline — Step 11b: Dataset-level QC (v2 addendum §11)

Runs after scripts/11_package.py has assembled a batch's
delivery/{batch_id}/dataset_manifest.json and
delivery/{batch_id}/sessions/{session_id}/ bundles. Computes, across the
WHOLE BATCH (this is the first stage where "batch" is a meaningful unit
at all, not per-session or per-episode):

  - near-duplicate session detection (perceptual hash on each session's
    delivered video, config.DEDUP_SIMILARITY_THRESHOLD)
  - task-distribution balance (task_imbalance_ratio, from the manifest's
    existing per-episode task_distribution)
  - diversity summary (object-class diversity from real object_tracks.json
    class_label output — v2 §3 — plus worker diversity from
    session_meta.json's worker_id — v2 §2/§5)
  - stratified episode-level train/val/test split
    (config.TRAIN_VAL_TEST_SPLIT, config.SPLIT_STRATIFY_BY)

⚠️ WHAT THIS CANNOT PROVE TODAY: with only one real session in this repo
(session_001, confirmed off-taxonomy — see docs/PIPELINE_STATUS.md's
business-blocker note), every number this script produces against a real
batch is structurally meaningless at dataset scale — there is nothing for
session_001 to be a near-duplicate OF, its "task distribution" is a single
"unknown" entry, its object-class diversity is whatever §3 happened to
detect in one paperwork/stapling clip, and a stratified split of one
session's episodes proves nothing about how this behaves across many. This
script's CORRECTNESS (not its real-world validity) is what
tests/test_dataset_qc.py and utils/dataset_qc.py's unit tests establish,
using synthetic multi-session fixtures — same discipline as §2/§5/§7's
synthetic task-signature tests. dataset_qc_report.json's
"single_real_session_caveat" field says this explicitly, so it isn't lost
the moment this file is read out of context.

Input:  delivery/{batch_id}/dataset_manifest.json
        delivery/{batch_id}/sessions/{session_id}/compressed.mp4
        processed/{session_id}/object_tracks.json (optional)
        processed/{session_id}/session_meta.json
Output: delivery/{batch_id}/dataset_qc_report.json
        delivery/{batch_id}/dataset_manifest.json (extended in place with
            dedup_removed_count, task_imbalance_ratio, diversity_summary,
            splits)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.dataset_qc import (
    compute_diversity_summary,
    compute_task_imbalance_ratio,
    dedup_sessions,
    phash_frame,
    stratified_split,
)

STEP = "11b_dataset_qc"


def _hash_session_video(video_path: Path, hash_size: int = 8):
    """
    Perceptual hash of the middle frame of a delivered session's video.
    Returns None (not a crash) if the video can't be read — that session
    is excluded from dedup comparison rather than blocking the whole
    batch's QC run.
    """
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return phash_frame(gray, hash_size=hash_size)


def _collect_object_class_counts(session_id: str) -> dict:
    tracks_path = cfg.PROCESSED_DIR / session_id / "object_tracks.json"
    counts = {}
    if not tracks_path.exists():
        return counts
    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)
    for frame in tracks:
        for obj in frame.get("tracked_objects", []):
            label = obj.get("class_label")
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def _read_worker_id(session_id: str):
    meta_path = cfg.PROCESSED_DIR / session_id / "session_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f).get("worker_id")


def run(batch_id: str) -> dict:
    t0 = time.time()
    print(f"[{STEP}] Starting batch '{batch_id}'...")

    delivery_dir = cfg.DELIVERY_DIR / batch_id
    manifest_path = delivery_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"[{STEP}] dataset_manifest.json not found: {manifest_path} "
            f"— run scripts/11_package.py first"
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    session_manifests = manifest.get("sessions", [])
    session_ids = [s["session_id"] for s in session_manifests]

    # ─── Dedup ──────────────────────────────────────────────
    session_hashes = []
    for session_id in session_ids:
        video_path = delivery_dir / "sessions" / session_id / "compressed.mp4"
        h = _hash_session_video(video_path)
        if h is not None:
            session_hashes.append((session_id, h))

    dedup_result = dedup_sessions(session_hashes, cfg.DEDUP_SIMILARITY_THRESHOLD)
    print(
        f"[{STEP}] Dedup: {len(dedup_result['removed'])} near-duplicate "
        f"session(s) flagged of {len(session_ids)}"
    )

    # ─── Task balance ───────────────────────────────────────
    task_distribution = manifest.get("task_distribution", {})
    task_imbalance_ratio = compute_task_imbalance_ratio(task_distribution)

    # ─── Diversity ──────────────────────────────────────────
    object_class_counts = {}
    worker_ids = []
    for session_id in session_ids:
        for label, count in _collect_object_class_counts(session_id).items():
            object_class_counts[label] = object_class_counts.get(label, 0) + count
        worker_ids.append(_read_worker_id(session_id))

    diversity_summary = compute_diversity_summary(object_class_counts, worker_ids)

    # ─── Stratified split ───────────────────────────────────
    all_episodes = []
    for s in session_manifests:
        all_episodes.extend(s.get("episodes", []))

    splits = stratified_split(all_episodes, cfg.TRAIN_VAL_TEST_SPLIT, cfg.SPLIT_STRATIFY_BY)
    print(f"[{STEP}] Split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # ─── Extend dataset_manifest.json in place ──────────────
    manifest["dedup_removed_count"] = len(dedup_result["removed"])
    manifest["task_imbalance_ratio"] = task_imbalance_ratio
    manifest["diversity_summary"] = diversity_summary
    manifest["splits"] = splits

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ─── dataset_qc_report.json ──────────────────────────────
    single_real_session_caveat = (
        "With fewer than 2 sessions in this batch, dedup/task-balance/"
        "diversity/split numbers below are structurally undefined or "
        "trivial at dataset scale — they are NOT evidence this batch is "
        "clean, balanced, or diverse. See docs/PIPELINE_STATUS.md's "
        "business-blocker note for the project-wide real-data status."
        if len(session_ids) < 2
        else None
    )

    report = {
        "batch_id": batch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_count": len(session_ids),
        "single_real_session_caveat": single_real_session_caveat,
        "dedup": {
            "removed_count": len(dedup_result["removed"]),
            "removed_session_ids": dedup_result["removed"],
            "duplicate_of": dedup_result["duplicate_of"],
            "similarities": dedup_result["similarities"],
            "similarity_threshold": cfg.DEDUP_SIMILARITY_THRESHOLD,
        },
        "task_distribution": task_distribution,
        "task_imbalance_ratio": task_imbalance_ratio,
        "diversity_summary": diversity_summary,
        "splits": {name: len(ids) for name, ids in splits.items()},
        "split_episode_ids": splits,
    }

    report_path = delivery_dir / "dataset_qc_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[{STEP}] Report: {report_path}")
    if single_real_session_caveat:
        print(f"[{STEP}] ⚠ {single_real_session_caveat}")

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return report


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 11b: Dataset-level QC")
    parser.add_argument("--batch", type=str, required=True, help="Batch ID")
    args = parser.parse_args()
    run(args.batch)


if __name__ == "__main__":
    main()
