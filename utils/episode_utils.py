"""
DatraAI Pipeline — Episode Utilities (v2 addendum §6)
Shared helpers for the scripts that process a session per-episode
(07_task_classify.py, 09_language_ground.py, 10_eis.py) rather than
per-session, once episodes.json exists (scripts/06b_episode_segment.py).
"""

import json
from pathlib import Path
from typing import List


def load_episodes(proc_dir: Path) -> List[dict]:
    """
    Load episodes.json's episode list. Raises FileNotFoundError if it
    doesn't exist — episodes.json is a required input for every
    per-episode stage, not an optional one; run 06b_episode_segment.py
    first.
    """
    episodes_path = proc_dir / "episodes.json"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"episodes.json not found: {episodes_path} — run "
            f"scripts/06b_episode_segment.py before any per-episode stage."
        )
    with open(episodes_path) as f:
        data = json.load(f)
    return data.get("episodes", [])


def filter_frames(items: List[dict], start_frame: int, end_frame: int, frame_key: str = "frame_idx") -> List[dict]:
    """Select entries whose frame_key falls within [start_frame, end_frame] inclusive."""
    return [item for item in items if start_frame <= item.get(frame_key, -1) <= end_frame]


def filter_segments(segments: List[dict], start_frame: int, end_frame: int) -> List[dict]:
    """
    Select phases.json segments that overlap an episode's frame range
    (a segment overlaps if it starts before the episode ends AND ends
    after the episode starts) — segments aren't keyed by a single
    frame_idx, so filter_frames() doesn't apply directly.
    """
    return [
        seg for seg in segments
        if seg.get("start_frame", 0) <= end_frame and seg.get("end_frame", -1) >= start_frame
    ]
