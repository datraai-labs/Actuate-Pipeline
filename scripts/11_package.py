"""
DatraAI Pipeline — Step 11: Package
Assemble final delivery bundle, dataset manifest, data card, optional S3 upload.

Input:  processed/{session_id}/ (all output files)
Output: delivery/{batch_id}/sessions/{session_id}/ (assembled bundle)
        delivery/{batch_id}/dataset_manifest.json
        delivery/{batch_id}/data_card.md
        Optional: S3 upload + presigned URLs
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.episode_utils import load_episodes, filter_frames, filter_segments

STEP = "11_package"


def _assemble_action_labels(proc_dir: Path, session_id: str) -> dict:
    """
    Assemble the master action_labels.json, nested by episode (v2 addendum
    §6) — task/phases/primitives/quality per episode, with session-wide
    facts (validation, session_mean_EIS) kept at the top level since
    08_validate.py wasn't restructured per-episode by this section.
    """
    task_label = {"episodes": []}
    task_path = proc_dir / "task_label.json"
    if task_path.exists():
        with open(task_path, encoding="utf-8") as f:
            task_label = json.load(f)
    task_by_episode = {e["episode_id"]: e for e in task_label.get("episodes", [])}

    phases = {"segments": []}
    phases_path = proc_dir / "phases.json"
    if phases_path.exists():
        with open(phases_path, encoding="utf-8") as f:
            phases = json.load(f)

    all_prims = []
    prim_path = proc_dir / "primitives.json"
    if prim_path.exists():
        with open(prim_path, encoding="utf-8") as f:
            all_prims = json.load(f)

    validation = {}
    val_path = proc_dir / "validation_report.json"
    if val_path.exists():
        with open(val_path, encoding="utf-8") as f:
            validation = json.load(f)

    quality = {"episodes": [], "session_mean_EIS": 0}
    cert_path = proc_dir / "quality_certificate.json"
    if cert_path.exists():
        with open(cert_path, encoding="utf-8") as f:
            quality = json.load(f)
    quality_by_episode = {e["episode_id"]: e for e in quality.get("episodes", [])}

    episodes = load_episodes(proc_dir) if (proc_dir / "episodes.json").exists() else []

    episode_entries = []
    for ep in episodes:
        episode_id = ep["episode_id"]
        ep_segments = filter_segments(phases.get("segments", []), ep["start_frame"], ep["end_frame"])
        # Only include frames with non-empty active_primitives, same as v1.
        ep_prims = [
            p for p in filter_frames(all_prims, ep["start_frame"], ep["end_frame"])
            if len(p.get("active_primitives", [])) > 0
        ]
        ep_quality = quality_by_episode.get(episode_id, {})
        ep_task = task_by_episode.get(episode_id, {})

        # v2 addendum §9 — confidence made a first-class, nested value:
        # task-level (from 07_task_classify.py's signature-match score),
        # segment-level (06_phase_segment.py's per-segment mean_confidence,
        # with the phase/frame range it applies to), and frame-level
        # (already present per-frame in L3_primitives' own
        # "primitive_confidences" field, hence frame_level_available: true
        # rather than duplicating those floats a third time here).
        confidence_tree = {
            "task_level_confidence": ep_task.get("confidence", 0.0),
            "segment_level_confidences": [
                {
                    "phase": seg.get("phase"),
                    "start_frame": seg.get("start_frame"),
                    "end_frame": seg.get("end_frame"),
                    "mean_confidence": seg.get("mean_confidence", 0.0),
                }
                for seg in ep_segments
            ],
            "frame_level_available": True,
        }

        episode_entries.append({
            "episode_id": episode_id,
            "start_frame": ep["start_frame"],
            "end_frame": ep["end_frame"],
            "duration_sec": ep.get("duration_sec", 0.0),
            "L1_task": ep_task.get("L1_task", "unknown"),
            "L1_task_confidence": ep_task.get("confidence", 0.0),
            "L2_phases": ep_segments,
            "L3_primitives": ep_prims,
            "quality": {
                "EIS": ep_quality.get("EIS", 0),
                "flags": ep_quality.get("flags", []),
            },
            "confidence_tree": confidence_tree,
        })

    return {
        "session_id": session_id,
        "pipeline_version": cfg.PIPELINE_VERSION,
        "processing_date": datetime.now(timezone.utc).isoformat(),
        "episodes": episode_entries,
        "validation": validation,
        "session_mean_EIS": quality.get("session_mean_EIS", 0),
    }


def _generate_data_card(batch_id: str, manifest: dict) -> str:
    """Generate data_card.md markdown content."""
    sessions = manifest.get("sessions", [])
    total_duration = manifest.get("total_duration_hours", 0.0)
    episode_count = manifest.get("episode_count", 0)
    mean_eis = manifest.get("mean_EIS", 0.0)
    task_dist = manifest.get("task_distribution", {})
    qa_passed = sum(1 for s in sessions if s.get("qa_passed", False))
    qa_total = len(sessions)

    # Task distribution table — counted per EPISODE (v2 addendum §6), since
    # one session can contain multiple task instances.
    task_table = "| Task | Episode Count |\n|------|-------|\n"
    for task, count in sorted(task_dist.items(), key=lambda x: -x[1]):
        task_table += f"| {task} | {count} |\n"

    card = f"""# DatraAI Dataset — {batch_id}

## Overview

- **Batch ID**: `{batch_id}`
- **Sessions**: {len(sessions)}
- **Episodes**: {episode_count} (a session may contain multiple distinct task instances — see action_labels.json's "episodes" array)
- **Total Duration**: {total_duration:.1f} hours
- **Environment**: Factory floor, egocentric head-mounted cameras
- **Collection Device**: Panoculon wearable (1080p, 30fps, IMU 200Hz)

## Modalities

| Modality | Format | Specs |
|----------|--------|-------|
| RGB Video | MP4 (re-encoded during privacy redaction — not the original H.265 ingest codec; see Privacy section) | 1080p, 30fps |
| IMU | HDF5 (synced) | 200Hz raw → 30Hz synced, 3-axis accel + 3-axis gyro |
| Hand Pose | JSON | MediaPipe 21-keypoint, 2 hands, per-frame |
| Action Labels | JSON | L1 task + L2 phases + L3 primitives, nested per episode — a session may contain multiple task instances |
| Language Grounding | JSON | Natural language instruction per episode |

## Quality

- **Mean EIS**: {mean_eis:.1f} / 100
- **QA Pass Rate**: {qa_passed}/{qa_total} ({qa_passed/max(1,qa_total)*100:.0f}%)
- **Sync Drift**: Verified < 2ms for all passing sessions

## Privacy

Every session's video was passed through automated face and text/badge
redaction before delivery (see each session's `privacy_report.json` for
per-session detection/redaction counts). Bystander faces are blurred by
default; text/badge detection uses OCR and may not catch every instance —
`frames_flagged_for_review` in the report indicates ambiguous detections a
human should double-check. This is automated redaction, not a guarantee of
zero PII — treat accordingly per your data handling agreement.

## Task Distribution

{task_table}

## File Format

Each session folder contains:

```
sessions/session_XXX/
├── compressed.mp4           # Privacy-redacted video (see Privacy section)
├── session.h5               # HDF5: synced timestamps + IMU arrays
├── hand_pose.json           # Per-frame hand keypoints
├── action_labels.json       # L1 + L2 + L3 labels
├── language_grounding.json  # NL instruction
├── quality_certificate.json # EIS + flags
└── privacy_report.json      # Redaction counts + consent status
```

### Loading in Python

```python
import h5py
import json
import cv2

# Load HDF5
with h5py.File("session.h5", "r") as f:
    timestamps = f["video"]["timestamps"][:]      # [N] float64, epoch seconds
    accel = f["imu"]["accel"][:]                  # [N, 3] float32, m/s²
    gyro = f["imu"]["gyro"][:]                    # [N, 3] float32, rad/s
    meta = json.loads(f.attrs["metadata"])

# Load labels — nested per episode (v2 addendum §6); a session can
# contain multiple distinct task instances, each with its own L1/L2/L3.
with open("action_labels.json") as f:
    labels = json.load(f)
    for episode in labels["episodes"]:
        episode_id = episode["episode_id"]
        task = episode["L1_task"]
        phases = episode["L2_phases"]
        primitives = episode["L3_primitives"]

# Load video frame
cap = cv2.VideoCapture("compressed.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()
```

## Pipeline

- **Version**: {cfg.PIPELINE_VERSION}
- **Processing Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
- **Steps**: Ingest → Sync → QA → Privacy Redact → Hand Pose → Primitives → Phases → Episode Segment → Task (per episode) → Validate → Language (per episode) → EIS (per episode) → Package

## License

Licensed to [CLIENT]. Non-exclusive. Contact DatraAI for terms.
"""
    return card


def run(
    session_ids: list,
    batch_id: str = None,
    upload: bool = False,
) -> dict:
    """
    Package sessions into a delivery bundle.
    """
    t0 = time.time()

    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d')}"

    print(f"[{STEP}] Packaging batch '{batch_id}' with {len(session_ids)} session(s)...")

    delivery_dir = cfg.DELIVERY_DIR / batch_id
    sessions_dir = delivery_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    session_manifests = []
    task_distribution = {}
    total_duration = 0.0
    eis_scores = []

    for session_id in session_ids:
        proc_dir = cfg.PROCESSED_DIR / session_id
        if not proc_dir.exists():
            print(f"[{STEP}] ⚠ Skipping {session_id} — processed dir not found")
            continue

        # v2 addendum §10 — never let an unredacted video reach delivery/.
        # redacted_compressed.mp4 (scripts/03b_privacy_redact.py) is the
        # ONLY video file this step will copy. If it's missing (03b hasn't
        # run, or failed), the session is excluded from the batch entirely
        # rather than silently falling back to the unredacted compressed.mp4.
        redacted_video_path = proc_dir / "redacted_compressed.mp4"
        if not redacted_video_path.exists():
            print(
                f"[{STEP}] ✗ Skipping {session_id} — redacted_compressed.mp4 not found "
                f"(run scripts/03b_privacy_redact.py first). Refusing to deliver "
                f"unredacted video."
            )
            continue

        session_out = sessions_dir / session_id
        session_out.mkdir(parents=True, exist_ok=True)

        # Files to copy. "compressed.mp4" is intentionally NOT in this list —
        # redacted_compressed.mp4 (copied below, renamed to compressed.mp4 in
        # the delivered bundle so downstream loaders/data_card.md don't need
        # a special case) is the only video that ships.
        copy_files = [
            "session.h5",
            "hand_pose.json",
            "language_grounding.json",
            "quality_certificate.json",
            "privacy_report.json",
        ]

        for fname in copy_files:
            src = proc_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(session_out / fname))

        shutil.copy2(str(redacted_video_path), str(session_out / "compressed.mp4"))

        # Assemble action_labels.json
        action_labels = _assemble_action_labels(proc_dir, session_id)
        with open(session_out / "action_labels.json", "w", encoding="utf-8") as f:
            json.dump(action_labels, f, indent=2)

        # Collect manifest info — episode-level (v2 addendum §6): task
        # distribution and episode_count are counted per EPISODE, not per
        # session, since one session can contain multiple task instances.
        meta_path = proc_dir / "session_meta.json"
        cert_path = proc_dir / "quality_certificate.json"
        qa_path = proc_dir / "qa_report.json"
        task_path = proc_dir / "task_label.json"

        duration = 0.0
        session_mean_eis = 0
        qa_passed = False
        episode_summaries = []

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
                duration = meta.get("duration_seconds", 0.0)

        if cert_path.exists():
            with open(cert_path, encoding="utf-8") as f:
                cert = json.load(f)
                session_mean_eis = cert.get("session_mean_EIS", 0)
                eis_by_episode = {e["episode_id"]: e.get("EIS", 0) for e in cert.get("episodes", [])}
        else:
            eis_by_episode = {}

        task_by_episode = {}
        if task_path.exists():
            with open(task_path, encoding="utf-8") as f:
                tl = json.load(f)
                task_by_episode = {e["episode_id"]: e.get("L1_task", "unknown") for e in tl.get("episodes", [])}

        if qa_path.exists():
            with open(qa_path, encoding="utf-8") as f:
                qa = json.load(f)
                qa_passed = qa.get("overall_passed", False)

        for episode_id, task in task_by_episode.items():
            episode_eis = eis_by_episode.get(episode_id, 0)
            eis_scores.append(episode_eis)
            task_distribution[task] = task_distribution.get(task, 0) + 1
            episode_summaries.append({"episode_id": episode_id, "L1_task": task, "EIS": episode_eis})

        total_duration += duration

        session_manifests.append({
            "session_id": session_id,
            "episode_count": len(episode_summaries),
            "session_mean_EIS": session_mean_eis,
            "episodes": episode_summaries,
            "duration_sec": round(duration, 1),
            "qa_passed": qa_passed,
        })

        print(
            f"[{STEP}]   ✓ {session_id}: {len(episode_summaries)} episode(s), "
            f"session_mean_EIS={session_mean_eis}, duration={duration:.1f}s"
        )

    # ─── Dataset manifest ────────────────────────────────────
    mean_eis = sum(eis_scores) / max(1, len(eis_scores))
    episode_count = sum(s["episode_count"] for s in session_manifests)

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_count": len(session_manifests),
        "episode_count": episode_count,
        "total_duration_hours": round(total_duration / 3600.0, 2),
        "task_distribution": task_distribution,
        "mean_EIS": round(mean_eis, 1),
        "modalities": [
            "RGB_video",
            "IMU_200Hz",
            "hand_pose_21kp",
            "action_labels_L1L2L3",
            "language_grounding",
        ],
        "format": "HDF5 + MP4 + JSON",
        "sessions": session_manifests,
    }

    manifest_path = delivery_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ─── Data card ───────────────────────────────────────────
    data_card = _generate_data_card(batch_id, manifest)
    card_path = delivery_dir / "data_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(data_card)

    print(f"[{STEP}] Manifest: {manifest_path}")
    print(f"[{STEP}] Data card: {card_path}")

    # ─── S3 upload ───────────────────────────────────────────
    if upload:
        print(f"[{STEP}] Uploading to S3...")
        try:
            from utils.s3_utils import check_credentials, upload_folder, generate_presigned_url

            if not check_credentials():
                print(f"[{STEP}] ✗ S3 upload skipped — credentials not configured")
            else:
                s3_prefix = f"deliveries/{batch_id}"
                upload_folder(delivery_dir, s3_prefix=s3_prefix)

                manifest_url = generate_presigned_url(
                    key=f"{s3_prefix}/dataset_manifest.json"
                )
                card_url = generate_presigned_url(
                    key=f"{s3_prefix}/data_card.md"
                )
                print(f"[{STEP}] Manifest URL (7-day): {manifest_url}")
                print(f"[{STEP}] Data Card URL (7-day): {card_url}")
        except Exception as e:
            print(f"[{STEP}] ✗ S3 upload failed: {e}")
    else:
        print(f"[{STEP}] S3 upload skipped (use --upload to enable)")

    # ─── Summary ─────────────────────────────────────────────
    print(f"\n[{STEP}] ═══ DELIVERY SUMMARY ═══")
    print(f"[{STEP}] Batch: {batch_id}")
    print(f"[{STEP}] Sessions: {len(session_manifests)}")
    print(f"[{STEP}] Episodes: {episode_count}")
    print(f"[{STEP}] Duration: {total_duration/3600:.2f} hours")
    print(f"[{STEP}] Mean EIS (per episode): {mean_eis:.1f}")
    print(f"[{STEP}] Tasks (per episode): {task_distribution}")

    # List all files created
    print(f"\n[{STEP}] Files created:")
    for f in sorted(delivery_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"[{STEP}]   {f.relative_to(delivery_dir)} ({size_str})")

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 11: Package delivery")
    parser.add_argument(
        "--session",
        type=str,
        nargs="+",
        required=True,
        help="Session ID(s) to package",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID (default: batch_YYYYMMDD)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to S3 after packaging",
    )
    args = parser.parse_args()

    session_ids = [Path(s).name for s in args.session]
    run(session_ids, batch_id=args.batch_id, upload=args.upload)


if __name__ == "__main__":
    main()
