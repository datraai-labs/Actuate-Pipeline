"""
DatraAI Pipeline — Step 06: Phase Segmentation
Heuristic phase sequencer: primitives → L2 phase labels per frame.

Each segment also gets a "mean_confidence" (v2 addendum §9), averaged
across its constituent frames' confidence in the specific primitive(s)
that justify its phase label (per PHASE_RELEVANT_PRIMITIVES below) —
propagating primitives.json's per-frame primitive_confidences up to the
phase level.

Input:  processed/{session_id}/primitives.json
Output: processed/{session_id}/phases.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

STEP = "06_phase_segment"

# Which primitive(s) justify each phase label, per _assign_phase's priority
# logic below (v2 addendum §9) — used to pick which primitive_confidences
# entries a segment's mean_confidence is averaged from.
PHASE_RELEVANT_PRIMITIVES = {
    "active_manipulation": ["wrist_pronate", "wrist_supinate", "wrist_flex", "power_grasp", "lateral_pinch"],
    "release": ["contact_release"],
    "grasp": ["power_grasp", "lateral_pinch"],
    "reach": ["reach_onset"],
    "reposition": ["transport"],
    "idle": ["idle"],
}


def _frame_relevant_confidence(flags: dict, confidences: dict, phase: str) -> float:
    """
    Confidence that this frame's own primitive evidence supports its
    assigned phase label — averages PHASE_RELEVANT_PRIMITIVES[phase]'s
    confidences, restricted to whichever of them actually fired (True) on
    this frame. If none fired here (this frame's final label was inherited
    from a neighboring segment during gap-fill/short-segment smoothing
    rather than from its own raw primitives), falls back to averaging ALL
    of the phase's relevant primitives regardless of boolean state, since
    that's still the most relevant measurement context available.
    """
    relevant = PHASE_RELEVANT_PRIMITIVES.get(phase, [])
    if not relevant:
        return 0.0
    active_relevant = [p for p in relevant if flags.get(p, False)]
    pool = active_relevant if active_relevant else relevant
    values = [confidences.get(p, 0.0) for p in pool]
    return float(sum(values) / len(values)) if values else 0.0


def _assign_phase(flags: dict) -> str:
    """
    Assign L2 phase label from primitive flags using priority rules.

    Priority order:
    1. active_manipulation: wrist rotation/flex + grasp
    2. release: contact_release
    3. grasp: power_grasp or lateral_pinch without contact
    4. reach: reach_onset without grasp or contact
    5. reposition: transport without grasp
    6. idle: idle primitive
    7. fallback: idle
    """
    pronate = flags.get("wrist_pronate", False)
    supinate = flags.get("wrist_supinate", False)
    flex = flags.get("wrist_flex", False)
    reach = flags.get("reach_onset", False)
    power = flags.get("power_grasp", False)
    pinch = flags.get("lateral_pinch", False)
    contact = flags.get("contact_onset", False)
    release = flags.get("contact_release", False)
    transport = flags.get("transport", False)
    idle_flag = flags.get("idle", False)

    has_grasp = power or pinch
    has_rotation = pronate or supinate or flex

    # Priority 1: active_manipulation
    if has_rotation and has_grasp:
        return "active_manipulation"

    # Priority 2: release
    if release:
        return "release"

    # Priority 3: grasp
    if has_grasp and not contact:
        return "grasp"

    # Priority 4: reach
    if reach and not has_grasp and not contact:
        return "reach"

    # Priority 5: reposition
    if transport and not has_grasp:
        return "reposition"

    # Priority 6: idle
    if idle_flag:
        return "idle"

    # Fallback
    return "idle"


def _fill_short_gaps(labels: list, max_gap: int) -> list:
    """
    Fill short gaps: if a phase is interrupted for < max_gap frames
    by a different phase, fill the gap with the surrounding phase.
    """
    n = len(labels)
    if n == 0:
        return labels

    result = list(labels)

    i = 0
    while i < n:
        # Find a run of the current label
        current_label = result[i]
        j = i + 1
        while j < n and result[j] == current_label:
            j += 1

        # Now result[i:j] is a run of current_label.
        # Check if there's a short gap followed by the same label.
        if j < n:
            gap_start = j
            gap_label = result[gap_start]
            gap_end = gap_start + 1
            while gap_end < n and result[gap_end] == gap_label:
                gap_end += 1

            gap_len = gap_end - gap_start

            # Check if after the gap, the same original label resumes
            if gap_len < max_gap and gap_end < n and result[gap_end] == current_label:
                # Fill the gap
                for k in range(gap_start, gap_end):
                    result[k] = current_label
                # Don't advance i — re-check the extended run
                continue

        i = j

    return result


def _merge_adjacent(labels: list) -> list:
    """Merge adjacent segments of the same phase (no-op on the label array, just ensures cleanliness)."""
    # This is implicitly handled by the segment extraction, but we do a pass for safety.
    return labels


def _remove_short_segments(labels: list, min_duration: int) -> list:
    """
    Remove segments shorter than min_duration frames by replacing them
    with the surrounding (preceding) phase label.
    """
    n = len(labels)
    if n == 0:
        return labels

    result = list(labels)

    i = 0
    while i < n:
        j = i + 1
        while j < n and result[j] == result[i]:
            j += 1
        run_len = j - i

        if run_len < min_duration:
            # Replace with the label of the preceding segment (or next if at start)
            replacement = result[i - 1] if i > 0 else (result[j] if j < n else "idle")
            for k in range(i, j):
                result[k] = replacement

        i = j

    return result


def _extract_segments(labels: list, timestamps: list) -> list:
    """
    Convert frame-level labels into segment list.
    """
    if len(labels) == 0:
        return []

    segments = []
    seg_start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            start_sec = timestamps[seg_start] if seg_start < len(timestamps) else 0.0
            end_sec = timestamps[i - 1] if (i - 1) < len(timestamps) else 0.0
            segments.append({
                "phase": current_label,
                "start_frame": seg_start,
                "end_frame": i - 1,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "duration_frames": i - seg_start,
            })
            seg_start = i
            current_label = labels[i]

    # Final segment
    start_sec = timestamps[seg_start] if seg_start < len(timestamps) else 0.0
    end_sec = timestamps[-1] if len(timestamps) > 0 else 0.0
    segments.append({
        "phase": current_label,
        "start_frame": seg_start,
        "end_frame": len(labels) - 1,
        "start_sec": round(start_sec, 4),
        "end_sec": round(end_sec, 4),
        "duration_frames": len(labels) - seg_start,
    })

    return segments


def run(session_id: str) -> dict:
    """
    Assign L2 phase labels per frame and write phases.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    prim_path = proc_dir / "primitives.json"

    if not prim_path.exists():
        raise FileNotFoundError(f"[{STEP}] primitives.json not found: {prim_path}")

    with open(prim_path) as f:
        primitives = json.load(f)

    n_frames = len(primitives)
    print(f"[{STEP}] Processing {n_frames} frames...")

    # Extract timestamps
    timestamps = [p["timestamp_sec"] for p in primitives]

    # ─── Initial phase assignment ────────────────────────────
    raw_labels = []
    for p in primitives:
        flags = p["raw_flags"]
        phase = _assign_phase(flags)
        raw_labels.append(phase)

    # ─── Smoothing ───────────────────────────────────────────
    print(f"[{STEP}] Smoothing phase labels...")

    # Step 1: Fill short gaps
    labels = _fill_short_gaps(raw_labels, cfg.PHASE_GAP_FILL_MAX_FRAMES)

    # Step 2: Merge adjacent (implicit)
    labels = _merge_adjacent(labels)

    # Step 3: Remove short segments
    labels = _remove_short_segments(labels, cfg.PHASE_MIN_DURATION_FRAMES)

    # ─── Segment extraction ──────────────────────────────────
    segments = _extract_segments(labels, timestamps)

    # ─── Per-segment mean_confidence (v2 addendum §9) ────────
    # Uses each frame's own raw_flags/primitive_confidences (not the
    # smoothed active_primitives) against its FINAL (smoothed) phase
    # label — see _frame_relevant_confidence's fallback for frames whose
    # label was inherited during gap-fill/short-segment smoothing.
    frame_confidences = [
        _frame_relevant_confidence(
            primitives[i]["raw_flags"],
            primitives[i].get("primitive_confidences", {}),
            labels[i],
        )
        for i in range(n_frames)
    ]
    for seg in segments:
        seg_values = frame_confidences[seg["start_frame"]: seg["end_frame"] + 1]
        seg["mean_confidence"] = round(sum(seg_values) / len(seg_values), 4) if seg_values else 0.0

    # ─── Phase summary ───────────────────────────────────────
    phase_counter = Counter(labels)
    phase_summary = {}
    for phase in cfg.PHASES:
        count_frames = phase_counter.get(phase, 0)
        count_segments = sum(1 for s in segments if s["phase"] == phase)
        phase_summary[phase] = {
            "count": count_segments,
            "total_frames": count_frames,
        }
        if count_frames > 0:
            print(f"[{STEP}]   {phase}: {count_segments} segments, {count_frames} frames ({count_frames/n_frames*100:.1f}%)")

    # ─── Output ──────────────────────────────────────────────
    output = {
        "session_id": session_id,
        "frame_labels": labels,
        "segments": segments,
        "phase_summary": phase_summary,
    }

    output_path = proc_dir / "phases.json"
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    elapsed = time.time() - t0
    print(f"[{STEP}] {len(segments)} segments extracted")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 06: Phase Segmentation")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
