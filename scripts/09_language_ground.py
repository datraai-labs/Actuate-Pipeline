"""
DatraAI Pipeline — Step 09: Language Grounding
Template-based natural language instruction generator, per episode
(v2 addendum §6), with an optional VLM-hybrid path (v2 addendum §7,
revised) that grounds generation in actual sampled video frames instead of
structured facts alone.

Runs once per episode in episodes.json rather than once per session —
hand_pose.json and primitives.json are filtered to each episode's frame
range before computing dominant hand / grasp type, so a session containing
multiple task instances gets an independent instruction per instance
instead of one instruction averaged across all of them.

config.LANGUAGE_GEN_MODE selects the generation path:
  - "template" (default): structured-facts-only templating, no API calls,
    no cost. Unchanged from the pre-§7 behavior.
  - "vlm": calls Claude with sampled video frames + structured facts as
    supporting context (see utils/vlm_language.py). Raises on API failure
    — no fallback. Use "hybrid" for production robustness.
  - "hybrid": tries "vlm" per episode, falls back to "template" for that
    episode on any API error (network, rate limit, auth, etc.) — logged,
    not silent.

The VLM path also cross-checks its own independent task_guess against
07_task_classify.py's label for the same episode (surfaced as
task_classification_disagreement) and, for a spot-check sample of
episodes (config.VLM_HALLUCINATION_SPOTCHECK_RATE), runs a second call
verifying the generated instruction against the same frames
(hallucination_check) — see utils/vlm_language.py for both.

Input:  processed/{session_id}/task_label.json  (per-episode array)
        processed/{session_id}/episodes.json
        processed/{session_id}/hand_pose.json
        processed/{session_id}/primitives.json
        processed/{session_id}/session_meta.json
        processed/{session_id}/validation_report.json (optional)
        processed/{session_id}/phases.json (vlm/hybrid mode — frame sampling)
        processed/{session_id}/compressed.mp4 (vlm/hybrid mode — frame source)
Output: processed/{session_id}/language_grounding.json
        {"session_id": ..., "episodes": [{"episode_id": ..., "instruction": ..., ...}, ...],
         "vlm_cost_summary": {...}}  (vlm_cost_summary present only if any VLM calls were made)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.episode_utils import load_episodes, filter_frames
from utils import vlm_language as vlm

STEP = "09_language_ground"


class _SafeFormatDict(dict):
    """dict for str.format_map that leaves unknown placeholders as literal text instead of raising KeyError."""

    def __missing__(self, key):
        return f"{{{key}}}"


def _derive_grounding_facts(episode: dict, task_entry: dict, hand_pose_in_ep: list, primitives_in_ep: list, validation) -> dict:
    """
    Pure derivation of the facts both the template path and the VLM path
    need: dominant hand / grasp type from this episode's own frames (not a
    session-wide average), task label + confidence, and a success
    assessment. Given pre-filtered, pre-matched inputs.
    """
    hand_counts = Counter()
    for frame in hand_pose_in_ep:
        dom = frame.get("dominant_hand")
        if dom:
            hand_counts[dom] += 1
    dominant_hand = hand_counts.most_common(1)[0][0] if hand_counts else "right"

    lateral_count = 0
    power_count = 0
    for p in primitives_in_ep:
        active = p.get("active_primitives", [])
        if "lateral_pinch" in active:
            lateral_count += 1
        if "power_grasp" in active:
            power_count += 1
    grasp_type = "pinch grasp" if lateral_count > power_count else "power grasp"

    duration = episode.get("duration_sec", 0.0)
    task_confidence = task_entry.get("confidence", 0.0)
    validation_valid = validation.get("overall_valid", True) if validation else True
    success = validation_valid and task_confidence > 0.7
    task = task_entry.get("L1_task", "unknown")

    return {
        "task": task,
        "task_confidence": round(task_confidence, 4),
        "grasp_type": grasp_type,
        "dominant_hand": dominant_hand,
        # No object-detection stage exists upstream in this pipeline, so
        # these are placeholders, not grounded data — flagged via
        # object_grounded/location_grounded below.
        "object_class": "the object",
        "target_location": "the target location",
        "success": success,
        "duration_seconds": round(duration, 1),
    }


def _ground_episode(episode: dict, task_entry: dict, hand_pose_in_ep: list, primitives_in_ep: list, validation) -> dict:
    """
    Template-only instruction generation. Pure given pre-filtered,
    pre-matched inputs (no I/O, no API calls).
    """
    facts = _derive_grounding_facts(episode, task_entry, hand_pose_in_ep, primitives_in_ep, validation)
    outcome = "completed successfully" if facts["success"] else "attempted"

    template = cfg.INSTRUCTION_TEMPLATES.get(facts["task"], cfg.INSTRUCTION_TEMPLATES["default"])
    fields = {
        "grasp_type": facts["grasp_type"],
        "dominant_hand": facts["dominant_hand"],
        "object_class": facts["object_class"],
        "target_location": facts["target_location"],
        "outcome": outcome,
        "success": facts["success"],
        "duration": facts["duration_seconds"],
        "task": facts["task"],
    }
    # format_map with a __missing__ fallback means a template referencing an
    # unexpected key degrades gracefully instead of raising mid-fallback.
    instruction = template.format_map(_SafeFormatDict(fields))

    return {
        "instruction": instruction,
        "template_fields": {
            "task": facts["task"],
            "grasp_type": facts["grasp_type"],
            "dominant_hand": facts["dominant_hand"],
            "object_class": facts["object_class"],
            "target_location": facts["target_location"],
            "success": facts["success"],
            "duration_seconds": facts["duration_seconds"],
        },
        "object_grounded": False,
        "location_grounded": False,
        "template_version": "v1",
        "generation_method": "template",
    }


def _ground_episode_vlm(
    client,
    episode: dict,
    task_entry: dict,
    hand_pose_in_ep: list,
    primitives_in_ep: list,
    validation,
    video_path: Path,
    phase_segments: list,
    recent_instructions: list,
) -> dict:
    """
    VLM-grounded instruction generation (v2 addendum §7, revised). Raises
    the underlying SDK exception on any API failure — the caller (run())
    decides whether to fall back to _ground_episode, per LANGUAGE_GEN_MODE.
    """
    facts = _derive_grounding_facts(episode, task_entry, hand_pose_in_ep, primitives_in_ep, validation)

    frame_indices = vlm.sample_representative_frames(episode, phase_segments)
    frame_images = []
    sampled_indices = []
    for idx in frame_indices:
        b64 = vlm.encode_frame_base64(video_path, idx)
        if b64 is not None:
            frame_images.append(b64)
            sampled_indices.append(idx)
    if not frame_images:
        raise RuntimeError(f"no readable frames among sampled indices {frame_indices} in {video_path}")

    t0 = time.time()
    result = vlm.generate_instruction_vlm(client, facts, frame_images, recent_instructions)
    latency_sec = time.time() - t0
    cost_usd = vlm.estimate_cost_usd(result["usage"])

    parsed = result["parsed"]
    instruction = parsed["instruction"]

    disagreement = vlm.check_task_disagreement(parsed["task_guess"], facts["task"])

    hallucination_check = {"checked": False}
    episode_id = episode.get("episode_id", "")
    if vlm.should_spotcheck(episode_id, cfg.VLM_HALLUCINATION_SPOTCHECK_RATE):
        t1 = time.time()
        hall_result = vlm.check_instruction_hallucination(client, instruction, frame_images)
        hall_latency = time.time() - t1
        hall_cost = vlm.estimate_cost_usd(hall_result["usage"])
        hallucination_check = {
            "checked": True,
            "consistent": hall_result["parsed"]["consistent"],
            "unsupported_claims": hall_result["parsed"]["unsupported_claims"],
            "cost_usd": hall_cost,
            "latency_sec": round(hall_latency, 3),
        }
        cost_usd += hall_cost
        latency_sec += hall_latency

    return {
        "instruction": instruction,
        "template_fields": {
            "task": facts["task"],
            "grasp_type": facts["grasp_type"],
            "dominant_hand": facts["dominant_hand"],
            "object_class": parsed["objects_mentioned"][0] if parsed["objects_mentioned"] else facts["object_class"],
            "target_location": facts["target_location"],
            "success": facts["success"],
            "duration_seconds": facts["duration_seconds"],
        },
        "object_grounded": bool(parsed["objects_mentioned"]),
        "location_grounded": False,
        "template_version": "v1",
        "generation_method": "vlm",
        "task_classification_disagreement": disagreement,
        "hallucination_check": hallucination_check,
        "vlm_audit": {
            "frames_sampled": sampled_indices,
            "prompt_system": result["prompt_system"],
            "raw_response": result["raw_response_text"],
            "model": result["model"],
            "cost_usd": round(cost_usd, 6),
            "latency_sec": round(latency_sec, 3),
        },
    }


def run(session_id: str) -> dict:
    """
    Generate a natural language instruction for each episode and write
    language_grounding.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id

    with open(proc_dir / "task_label.json") as f:
        task_label = json.load(f)
    task_by_episode = {e["episode_id"]: e for e in task_label.get("episodes", [])}

    with open(proc_dir / "hand_pose.json") as f:
        hand_pose = json.load(f)

    with open(proc_dir / "primitives.json") as f:
        primitives = json.load(f)

    validation_path = proc_dir / "validation_report.json"
    validation = None
    if validation_path.exists():
        with open(validation_path) as f:
            validation = json.load(f)

    episodes = load_episodes(proc_dir)

    mode = cfg.LANGUAGE_GEN_MODE
    client = None
    phase_segments = []
    video_path = proc_dir / "compressed.mp4"
    if mode in ("vlm", "hybrid"):
        import anthropic
        client = anthropic.Anthropic()
        phases_path = proc_dir / "phases.json"
        if phases_path.exists():
            with open(phases_path) as f:
                phase_segments = json.load(f).get("segments", [])

    print(f"[{STEP}] Grounding {len(episodes)} episode(s)... (mode={mode})")

    episode_results = []
    recent_instructions = []
    vlm_calls = 0
    vlm_total_cost = 0.0
    vlm_total_latency = 0.0

    for episode in episodes:
        episode_id = episode["episode_id"]
        task_entry = task_by_episode.get(episode_id, {})
        hand_pose_in_ep = filter_frames(hand_pose, episode["start_frame"], episode["end_frame"])
        primitives_in_ep = filter_frames(primitives, episode["start_frame"], episode["end_frame"])

        grounding = None
        if mode in ("vlm", "hybrid"):
            try:
                grounding = _ground_episode_vlm(
                    client, episode, task_entry, hand_pose_in_ep, primitives_in_ep, validation,
                    video_path, phase_segments, recent_instructions,
                )
                vlm_calls += 1
                vlm_total_cost += grounding["vlm_audit"]["cost_usd"]
                vlm_total_latency += grounding["vlm_audit"]["latency_sec"]
            except Exception as e:
                if mode == "vlm":
                    raise
                print(f"[{STEP}] ⚠ VLM generation failed for {episode_id} ({e!r}) — falling back to template")
                grounding = _ground_episode(episode, task_entry, hand_pose_in_ep, primitives_in_ep, validation)
                grounding["generation_method"] = "template_fallback"
                grounding["vlm_fallback_reason"] = repr(e)
        else:
            grounding = _ground_episode(episode, task_entry, hand_pose_in_ep, primitives_in_ep, validation)

        entry = {"episode_id": episode_id, **grounding}
        episode_results.append(entry)
        recent_instructions.append(grounding["instruction"])
        if len(recent_instructions) > 5:
            recent_instructions.pop(0)

        marker = ""
        if grounding.get("task_classification_disagreement", {}).get("disagree"):
            marker += " ⚠ task disagreement"
        print(f"[{STEP}]   {episode_id}: \"{grounding['instruction']}\"{marker}")

    output = {
        "session_id": session_id,
        "episodes": episode_results,
    }

    if vlm_calls > 0:
        total_video_sec = sum(e.get("duration_sec", 0.0) for e in episodes)
        cost_per_hour = (vlm_total_cost / total_video_sec * 3600.0) if total_video_sec > 0 else 0.0
        cost_summary = {
            "vlm_calls": vlm_calls,
            "total_cost_usd": round(vlm_total_cost, 6),
            "avg_cost_per_episode_usd": round(vlm_total_cost / vlm_calls, 6),
            "avg_latency_per_episode_sec": round(vlm_total_latency / vlm_calls, 3),
            "cost_per_hour_of_video_usd": round(cost_per_hour, 4),
        }
        output["vlm_cost_summary"] = cost_summary
        print(
            f"[{STEP}] VLM cost: {vlm_calls} call(s), ${cost_summary['total_cost_usd']:.4f} total, "
            f"${cost_summary['avg_cost_per_episode_usd']:.4f}/episode, "
            f"${cost_summary['cost_per_hour_of_video_usd']:.2f}/hour-of-video "
            f"(field benchmark reference: ~$2.64/hour — not a target, a sanity anchor)"
        )

    output_path = proc_dir / "language_grounding.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 09: Language Grounding")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
