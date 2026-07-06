"""
DatraAI Pipeline — Step 07: Task Classification
Primitive combination lookup → L1 task label, per episode (v2 addendum §6),
with an object-identity bonus from real detection/tracking (v2 addendum §3).

Runs once per episode in episodes.json rather than once per session —
primitives.json is filtered to each episode's frame range before scoring,
so multiple task instances within one recording get independent labels
instead of being averaged into a single session-wide signature match.

The object-identity bonus (config.EXPECTED_OBJECT_CLASSES,
config.OBJECT_MATCH_BONUS) exists specifically because the primitive
vocabulary alone can't discriminate material_transfer/box_seal/
pick_and_place (see config.TASK_SIGNATURES' inline comment) — those three
tasks are distinguished only by WHICH object the worker is handling, which
scripts/04c_object_track.py's real class_label output can now supply.
"active-manipulation frames" here is approximated as "frames where at
least one primitive is firing" (primitives.json's active_primitives list
is non-empty) — this script doesn't have phases.json's actual phase
segmentation available per-frame, and re-deriving that boundary here would
duplicate 06_phase_segment.py's logic; treating any primitive activity as
"doing something with the hands" is a reasonable proxy for that purpose.

Input:  processed/{session_id}/primitives.json
        processed/{session_id}/episodes.json
        processed/{session_id}/object_tracks.json (optional — v2 addendum §3)
Output: processed/{session_id}/task_label.json
        {"session_id": ..., "episodes": [{"episode_id": ..., "L1_task": ..., ...}, ...]}
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.episode_utils import load_episodes, filter_frames

STEP = "07_task_classify"


def _dominant_object_class(primitives_in_episode: list, object_tracks_in_episode: list) -> Optional[str]:
    """
    Most frequently tracked object class_label during this episode's
    active-primitive frames (see module docstring for what "active" means
    here). None if object_tracks.json wasn't available for this session,
    or no tracked object overlapped an active-primitive frame — the
    object-identity bonus below is skipped entirely in that case, not
    defaulted to some placeholder class.
    """
    object_tracks_by_frame = {f["frame_idx"]: f for f in object_tracks_in_episode}
    class_counts = {}
    for frame in primitives_in_episode:
        if not frame.get("active_primitives"):
            continue
        obj_frame = object_tracks_by_frame.get(frame.get("frame_idx"))
        if not obj_frame:
            continue
        for obj in obj_frame.get("tracked_objects", []):
            label = obj.get("class_label")
            if label:
                class_counts[label] = class_counts.get(label, 0) + 1
    if not class_counts:
        return None
    return max(class_counts, key=class_counts.get)


def _classify_episode(primitives_in_episode: list, dominant_object_class: Optional[str] = None) -> dict:
    """
    Score every task signature against one episode's primitive frame
    counts, then apply the object-identity bonus (v2 addendum §3) if a
    dominant_object_class was found. Pure function (given pre-filtered
    inputs) — same signature-matching logic as v1, just scoped to an
    episode's frames instead of a whole session's.
    """
    prim_counts = {}
    for frame in primitives_in_episode:
        for prim in frame.get("active_primitives", []):
            prim_counts[prim] = prim_counts.get(prim, 0) + 1

    all_scores = {}
    for task_name, signature in cfg.TASK_SIGNATURES.items():
        if len(signature) == 0:
            all_scores[task_name] = 0.0
            continue

        component_scores = []
        for required_prim, required_count in signature.items():
            actual_count = prim_counts.get(required_prim, 0)
            score = min(actual_count / required_count, 1.0)
            component_scores.append(score)

        match_score = sum(component_scores) / len(component_scores)
        all_scores[task_name] = round(match_score, 4)

    # v2 addendum §3 — object-identity bonus, applied BEFORE the tie/
    # threshold logic below so a genuine object match can actually break a
    # primitive-vocabulary tie or push a task over the confidence
    # threshold, not just be a cosmetic addition after the decision is
    # already made.
    object_bonus_applied = {}
    if dominant_object_class:
        for task_name, expected_classes in cfg.EXPECTED_OBJECT_CLASSES.items():
            if task_name in all_scores and dominant_object_class in expected_classes:
                boosted = round(min(all_scores[task_name] + cfg.OBJECT_MATCH_BONUS, 1.0), 4)
                object_bonus_applied[task_name] = round(boosted - all_scores[task_name], 4)
                all_scores[task_name] = boosted

    if not all_scores:
        task_label, confidence, needs_review = "unknown", 0.0, True
    else:
        max_score = max(all_scores.values())
        if max_score < 0.5:
            # Low single-task score — same fallback as before.
            task_label, confidence, needs_review = "unknown", max_score, True
        else:
            # A confident-looking top score can still be a tie: >=2 tasks
            # landing within TASK_TIE_MARGIN of the max is a genuine
            # ambiguity, not a real match — picking one arbitrarily (e.g.
            # by dict insertion order in TASK_SIGNATURES) silently produces
            # a confident-looking WRONG label. A tie is at least as strong
            # a signal for human review as a low single-task score.
            tied_tasks = [t for t, s in all_scores.items() if (max_score - s) <= cfg.TASK_TIE_MARGIN]
            if len(tied_tasks) > 1:
                task_label, confidence, needs_review = "unknown", max_score, True
            else:
                task_label, confidence, needs_review = tied_tasks[0], max_score, False

    return {
        "L1_task": task_label,
        "confidence": round(confidence, 4),
        "method": "primitive_signature_v1",
        "all_scores": all_scores,
        "needs_human_review": needs_review,
        "primitive_counts": prim_counts,
        "dominant_object_class": dominant_object_class,
        "object_bonus_applied": object_bonus_applied,
    }


def run(session_id: str) -> dict:
    """
    Classify each episode's L1 task and write task_label.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    prim_path = proc_dir / "primitives.json"

    if not prim_path.exists():
        raise FileNotFoundError(f"[{STEP}] primitives.json not found: {prim_path}")

    with open(prim_path) as f:
        primitives = json.load(f)

    # Optional (v2 addendum §3) — a session with no object_tracks.json
    # (e.g. produced before this stage existed) just gets no object bonus,
    # not a crash.
    object_tracks_path = proc_dir / "object_tracks.json"
    object_tracks = []
    if object_tracks_path.exists():
        with open(object_tracks_path) as f:
            object_tracks = json.load(f)

    episodes = load_episodes(proc_dir)
    print(f"[{STEP}] Classifying {len(episodes)} episode(s)...")

    episode_results = []
    for episode in episodes:
        prims_in_ep = filter_frames(primitives, episode["start_frame"], episode["end_frame"])
        tracks_in_ep = filter_frames(object_tracks, episode["start_frame"], episode["end_frame"])
        dominant_object_class = _dominant_object_class(prims_in_ep, tracks_in_ep)
        classification = _classify_episode(prims_in_ep, dominant_object_class)
        entry = {"episode_id": episode["episode_id"], **classification}
        episode_results.append(entry)

        marker = " ⚠ needs review" if classification["needs_human_review"] else ""
        obj_note = f" [object: {dominant_object_class}]" if dominant_object_class else ""
        print(
            f"[{STEP}]   {episode['episode_id']}: {classification['L1_task']} "
            f"(confidence={classification['confidence']:.2f}){obj_note}{marker}"
        )

    output = {
        "session_id": session_id,
        "episodes": episode_results,
    }

    output_path = proc_dir / "task_label.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 07: Task Classification")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
