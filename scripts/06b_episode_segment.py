"""
DatraAI Pipeline — Step 06b: Multi-Episode Segmentation (v2 addendum §6)

A session's recording often contains multiple distinct task instances
separated by idle gaps (finish one bolt, walk to next station, start
another) rather than one continuous task. This splits phases.json's
segments into episodes on idle gaps exceeding EPISODE_GAP_THRESHOLD_SEC, so
scripts/07_task_classify.py, scripts/09_language_ground.py, and
scripts/10_eis.py can classify/describe/score each task instance
separately instead of averaging them into one session-wide label.

A session with no qualifying gap (the common case) produces exactly ONE
episode spanning the whole recording — this falls out of the same
boundary-scanning loop as the multi-episode case, not a special-cased
branch, specifically to avoid an off-by-one/zero-episode bug in the most
common real session shape.

Input:  processed/{session_id}/phases.json
Output: processed/{session_id}/episodes.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

STEP = "06b_episode_segment"


def _split_into_episodes(segments: List[dict], gap_threshold_sec: float) -> List[List[dict]]:
    """
    Group phases.json segments into episodes, cutting whenever an "idle"
    segment's duration exceeds gap_threshold_sec. The gap segment itself
    belongs to neither the preceding nor the following episode — it's the
    transition, not part of either task.

    No qualifying gap anywhere -> one group containing every segment (falls
    out of the loop naturally: episode_start_idx never advances, so the
    final flush emits segments[0:] as a single episode).
    """
    if not segments:
        return []

    groups = []
    episode_start_idx = 0

    for i, seg in enumerate(segments):
        duration = seg["end_sec"] - seg["start_sec"]
        is_gap = seg["phase"] == "idle" and duration > gap_threshold_sec
        if is_gap:
            if i > episode_start_idx:
                groups.append(segments[episode_start_idx:i])
            episode_start_idx = i + 1

    if episode_start_idx < len(segments):
        groups.append(segments[episode_start_idx:])

    return groups


def _build_episode_entry(session_id: str, episode_idx: int, segment_group: List[dict]) -> dict:
    start_frame = segment_group[0]["start_frame"]
    end_frame = segment_group[-1]["end_frame"]
    start_sec = segment_group[0]["start_sec"]
    end_sec = segment_group[-1]["end_sec"]
    return {
        "episode_id": f"{session_id}_ep{episode_idx:02d}",
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": round(start_sec, 4),
        "end_sec": round(end_sec, 4),
        "duration_sec": round(end_sec - start_sec, 4),
    }


def run(session_id: str) -> dict:
    """
    Split a session's phases.json into episodes and write episodes.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    phases_path = proc_dir / "phases.json"
    if not phases_path.exists():
        raise FileNotFoundError(f"[{STEP}] phases.json not found: {phases_path}")

    with open(phases_path) as f:
        phases = json.load(f)
    segments = phases.get("segments", [])

    print(f"[{STEP}] Splitting on idle gaps > {cfg.EPISODE_GAP_THRESHOLD_SEC}s...")
    raw_groups = _split_into_episodes(segments, cfg.EPISODE_GAP_THRESHOLD_SEC)

    episodes = []
    dropped_short = 0
    for group in raw_groups:
        entry = _build_episode_entry(session_id, len(episodes), group)
        if entry["duration_sec"] < cfg.MIN_EPISODE_DURATION_SEC:
            dropped_short += 1
            print(
                f"[{STEP}]   Dropping fragment ({entry['duration_sec']:.2f}s < "
                f"{cfg.MIN_EPISODE_DURATION_SEC}s): frames {entry['start_frame']}-{entry['end_frame']}"
            )
            continue
        episodes.append(entry)

    if not episodes and segments:
        # Every candidate group was below MIN_EPISODE_DURATION_SEC (e.g. a
        # very short recording) — fall back to the whole session as one
        # episode rather than silently producing zero episodes and
        # dropping the entire recording from everything downstream.
        print(
            f"[{STEP}] ⚠ All candidate episodes were below MIN_EPISODE_DURATION_SEC — "
            f"falling back to the whole session as a single episode."
        )
        episodes = [_build_episode_entry(session_id, 0, segments)]

    print(f"[{STEP}] {len(episodes)} episode(s) extracted, {dropped_short} short fragment(s) dropped")
    for ep in episodes:
        print(f"[{STEP}]   {ep['episode_id']}: frames {ep['start_frame']}-{ep['end_frame']} ({ep['duration_sec']:.1f}s)")

    output = {
        "session_id": session_id,
        "episode_gap_threshold_sec": cfg.EPISODE_GAP_THRESHOLD_SEC,
        "min_episode_duration_sec": cfg.MIN_EPISODE_DURATION_SEC,
        "episodes": episodes,
    }

    output_path = proc_dir / "episodes.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 06b: Multi-Episode Segmentation")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
