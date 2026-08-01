"""
DatraAI Pipeline — Step 10: Episode Integrity Score (EIS)
Composite quality metric computation, per episode (v2 addendum §6).

Runs once per episode in episodes.json rather than once per session.
Sync drift, blur, causal-ordering, retargeting eligibility, and IMU-mount
plausibility are session-wide facts (their upstream checks — 03_qa.py,
08_validate.py, 04d_depth_estimate.py, 05_primitives.py's mount check —
were not restructured per-episode by this section) and so are shared
across every episode's score; hand presence and label confidence are
genuinely episode-specific and computed from each episode's own frame
range. A `session_mean_EIS` rollup is kept at the top level for
convenience alongside the per-episode array.

label_confidence (v2 addendum §9) is now a real weighted average across
every confidence layer available for the episode — the task-classification
score from task_label.json AND each overlapping phases.json segment's own
mean_confidence — not just the bare task-confidence number used before §9.
See _compute_label_confidence for the weighting.

Input:  processed/{session_id}/qa_report.json
        processed/{session_id}/validation_report.json
        processed/{session_id}/task_label.json  (per-episode array)
        processed/{session_id}/episodes.json
        processed/{session_id}/hand_pose.json
        processed/{session_id}/phases.json  (optional — §9 mean_confidence)
        processed/{session_id}/session.h5
Output: processed/{session_id}/quality_certificate.json
        {"session_id": ..., "episodes": [{"episode_id": ..., "EIS": ..., ...}, ...],
         "session_mean_EIS": ..., "retargeting_eligible": ..., ...}
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import read_session_h5
from utils.episode_utils import load_episodes, filter_frames, filter_segments

STEP = "10_eis"


def _compute_label_confidence(task_confidence: float, ep_segments: list) -> float:
    """
    Real weighted average across every confidence layer available for this
    episode (v2 addendum §9) — not just the bare task-classification
    score. The task-level confidence counts as one entry weighted by the
    episode's total segment frame count, alongside each phase segment's
    own mean_confidence (06_phase_segment.py) weighted by that segment's
    own frame count — giving the task-level score and the aggregate of
    segment-level scores equal overall weight, with segments internally
    weighted by how much of the episode they cover. Falls back to the bare
    task confidence if no segment data is available (older phases.json
    without mean_confidence, or an episode with zero overlapping segments)
    so this never breaks on missing §9 data.
    """
    total_seg_frames = sum(
        max(0, s.get("end_frame", -1) - s.get("start_frame", 0) + 1) for s in ep_segments
    )
    if total_seg_frames <= 0:
        return task_confidence

    weighted_sum = task_confidence * total_seg_frames
    weight_total = total_seg_frames
    for s in ep_segments:
        frames = max(0, s.get("end_frame", -1) - s.get("start_frame", 0) + 1)
        weighted_sum += s.get("mean_confidence", 0.0) * frames
        weight_total += frames

    return round(weighted_sum / weight_total, 4)


def compute_eis(
    max_drift_ms: float,
    blur_score: float,
    hand_presence_rate: float,
    label_confidence: float,
    causal_passed: bool,
    causal_inversion_count: int = 0,
    causal_total_events: int = 1,
) -> dict:
    """
    Compute Episode Integrity Score (0-100) and its components.
    Pure function for testability.
    """
    # Component 1: Sync drift (weight 0.30)
    sync_score = max(0.0, 1.0 - max_drift_ms / 5.0)

    # Component 2: Blur (weight 0.20)
    blur_norm = min(blur_score / 200.0, 1.0)

    # Component 3: Hand presence (weight 0.20)
    hand_score = hand_presence_rate

    # Component 4: Label confidence (weight 0.15)
    label_score = label_confidence

    # Component 5: Causal check (weight 0.15)
    if causal_passed:
        causal_score = 1.0
    else:
        causal_score = max(0.0, 1.0 - (causal_inversion_count / max(1, causal_total_events)))

    # Weighted sum
    eis_raw = (
        sync_score * cfg.SYNC_WEIGHT
        + blur_norm * cfg.BLUR_WEIGHT
        + hand_score * cfg.HAND_PRESENCE_WEIGHT
        + label_score * cfg.LABEL_CONFIDENCE_WEIGHT
        + causal_score * cfg.CAUSAL_CHECK_WEIGHT
    )

    eis = int(round(eis_raw * 100))
    eis = max(0, min(100, eis))

    return {
        "eis": eis,
        "eis_raw": round(eis_raw, 6),
        "components": {
            "sync_drift": {"score": round(sync_score, 4), "weight": cfg.SYNC_WEIGHT, "max_drift_ms": max_drift_ms},
            "blur": {"score": round(blur_norm, 4), "weight": cfg.BLUR_WEIGHT, "blur_value": blur_score},
            "hand_presence": {"score": round(hand_score, 4), "weight": cfg.HAND_PRESENCE_WEIGHT, "rate": hand_presence_rate},
            "label_confidence": {"score": round(label_score, 4), "weight": cfg.LABEL_CONFIDENCE_WEIGHT},
            "causal_check": {"score": round(causal_score, 4), "weight": cfg.CAUSAL_CHECK_WEIGHT},
        },
    }


def run(session_id: str) -> dict:
    """
    Compute EIS per episode and write quality_certificate.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id

    # ─── Load session-wide inputs ─────────────────────────────
    with open(proc_dir / "qa_report.json") as f:
        qa_report = json.load(f)

    with open(proc_dir / "validation_report.json") as f:
        validation = json.load(f)

    with open(proc_dir / "task_label.json") as f:
        task_label = json.load(f)
    task_by_episode = {e["episode_id"]: e for e in task_label.get("episodes", [])}

    with open(proc_dir / "hand_pose.json") as f:
        hand_pose = json.load(f)

    # v2 addendum §9 — optional: phases.json's per-segment mean_confidence
    # feeds label_confidence below. Missing entirely (older run) or missing
    # mean_confidence per-segment (older phases.json) both degrade
    # gracefully via _compute_label_confidence's fallback.
    phases_path = proc_dir / "phases.json"
    phase_segments = []
    if phases_path.exists():
        with open(phases_path) as f:
            phase_segments = json.load(f).get("segments", [])

    h5_data = read_session_h5(proc_dir / "session.h5")
    sync_stats = h5_data["metadata"].get("sync_stats", {})

    episodes = load_episodes(proc_dir)

    # Missing upstream data is scored fail-closed (as if the check had
    # failed) on every component, and surfaced via `flags` rather than
    # silently defaulting some components optimistically and others
    # pessimistically. These are session-wide (03_qa.py / 08_validate.py
    # weren't restructured per-episode), so computed once and shared.
    missing_data_flags = []

    if not sync_stats:
        missing_data_flags.append("missing_sync_stats")
    # Gate on the CLEAN drift (dropout-spanning frames excluded — audit §3a);
    # legacy h5 without it falls back to the conflated metric. 5.0 -> sync_score 0
    # (fail-closed).
    max_drift_ms = sync_stats.get("max_drift_clean_ms", sync_stats.get("max_drift_ms", 5.0))

    # An IMU-t0-fallback anchor makes video/IMU alignment true BY CONSTRUCTION —
    # a clean drift score is then not evidence of alignment, and the certificate
    # must say so rather than let the sync component read as validation (§3b).
    temporal_alignment_validated = bool(
        sync_stats.get("temporal_alignment_validated", False)
    )
    temporal_alignment_note = sync_stats.get(
        "temporal_alignment_note",
        "sync_stats predates anchor validation (audit 2026-08-01) — treated as UNVALIDATED",
    )
    if not temporal_alignment_validated:
        missing_data_flags.append("temporal_alignment_unvalidated")

    blur_score = qa_report.get("checks", {}).get("blur", {}).get("score", 0.0)

    causal_check = validation.get("checks", {}).get("causal_ordering", {})
    if not causal_check:
        missing_data_flags.append("missing_causal_check")
        causal_passed = False
        causal_inversions = 1
        causal_total = 1
    else:
        causal_passed = causal_check.get("passed", True)
        causal_inversions = causal_check.get("inversion_count", 0)
        causal_total = causal_check.get("total_events", 1)

    # ─── Retargeting eligibility (v2 addendum §4) — session-wide ─
    depth_data_path = proc_dir / "depth_data.json"
    depth_mode_effective = "none"
    if depth_data_path.exists():
        with open(depth_data_path) as f:
            depth_data = json.load(f)
        if depth_data:
            depth_mode_effective = depth_data[0].get("depth_mode", "none")
    elif cfg.DEPTH_MODE == "none":
        depth_mode_effective = "none"
    else:
        depth_mode_effective = "none"
    retargeting_eligible = depth_mode_effective != "none"

    # ─── IMU mount plausibility (v2 addendum §1) — session-wide ──
    mount_check_path = proc_dir / "imu_mount_check.json"
    imu_mount_implausible = False
    imu_mount_warning = None
    if mount_check_path.exists():
        with open(mount_check_path) as f:
            imu_mount_check = json.load(f)
        plausibility = imu_mount_check.get("mount_plausibility", {})
        if plausibility.get("checked") and not plausibility.get("plausible", True):
            imu_mount_implausible = True
            imu_mount_warning = plausibility["reason"]

    print(f"[{STEP}] Scoring {len(episodes)} episode(s)...")

    episode_results = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        task_entry = task_by_episode.get(episode_id, {})

        hand_pose_in_ep = filter_frames(hand_pose, episode["start_frame"], episode["end_frame"])
        total_frames = len(hand_pose_in_ep)
        hands_detected_count = sum(1 for f in hand_pose_in_ep if f.get("hands_detected", False))
        hand_presence_rate = hands_detected_count / max(1, total_frames)

        ep_segments = filter_segments(phase_segments, episode["start_frame"], episode["end_frame"])
        label_confidence = _compute_label_confidence(task_entry.get("confidence", 0.0), ep_segments)

        eis_result = compute_eis(
            max_drift_ms=max_drift_ms,
            blur_score=blur_score,
            hand_presence_rate=hand_presence_rate,
            label_confidence=label_confidence,
            causal_passed=causal_passed,
            causal_inversion_count=causal_inversions,
            causal_total_events=causal_total,
        )
        eis = eis_result["eis"]

        # ─── Recommended use ─────────────────────────────────
        if eis >= 85:
            recommended_use = ["imitation_learning", "VLA_finetuning", "foundation_model_pretraining"]
        elif eis >= 70:
            recommended_use = ["foundation_model_pretraining", "action_recognition"]
        else:
            recommended_use = []  # quarantine

        if not retargeting_eligible and not cfg.ALLOW_2D_ONLY_VLA_FINETUNING:
            recommended_use = [u for u in recommended_use if u != "VLA_finetuning"]

        # ─── Flags ───────────────────────────────────────────
        flags = list(missing_data_flags)
        if eis < 70:
            flags.append("quarantine_low_eis")
        if not qa_report.get("overall_passed", True):
            flags.append("qa_failed")
        if not validation.get("overall_valid", True):
            flags.append("validation_failed")
        if task_entry.get("needs_human_review", False):
            flags.append("needs_human_review")
        if not retargeting_eligible and not cfg.ALLOW_2D_ONLY_VLA_FINETUNING:
            flags.append("no_metric_3d_retargeting_ineligible")
        if imu_mount_implausible:
            flags.append("imu_mount_implausible")

        entry = {
            "episode_id": episode_id,
            "EIS": eis,
            "components": eis_result["components"],
            "flags": flags,
            "recommended_use": recommended_use,
        }
        if imu_mount_warning:
            entry["imu_mount_warning"] = imu_mount_warning
        episode_results.append(entry)

        print(f"[{STEP}]   {episode_id}: EIS={eis}/100" + (f" flags={flags}" if flags else ""))

    session_mean_eis = round(sum(e["EIS"] for e in episode_results) / max(1, len(episode_results)), 1)

    # ─── Output ──────────────────────────────────────────────
    output = {
        "session_id": session_id,
        "episodes": episode_results,
        "session_mean_EIS": session_mean_eis,
        "retargeting_eligible": retargeting_eligible,
        "depth_mode": depth_mode_effective,
        "temporal_alignment": {
            "validated": temporal_alignment_validated,
            "note": temporal_alignment_note,
        },
        "pipeline_version": cfg.PIPELINE_VERSION,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_path = proc_dir / "quality_certificate.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{STEP}] Session mean EIS: {session_mean_eis}/100")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 10: EIS Computation")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
