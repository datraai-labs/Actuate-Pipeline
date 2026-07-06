"""
DatraAI Pipeline — VLM-Hybrid Language Grounding (v2 addendum §7, revised)

Instead of generating instructions purely from structured facts (the
original §7 spec), this grounds generation in actual sampled video frames
from the episode via a vision-capable Claude call — structured facts are
supporting context, not the sole input. scripts/09_language_ground.py
selects LANGUAGE_GEN_MODE ("template" / "vlm" / "hybrid") and calls into
this module; the pure, no-network functions here (frame sampling, cost
estimation, spot-check gating, disagreement comparison) are unit-tested
directly, while the two functions that call the Claude API
(generate_instruction_vlm, check_instruction_hallucination) take an
injected client so tests can use a fake and real-data verification uses
the real anthropic.Anthropic() client.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import config as cfg

GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "Natural-language instruction describing what the worker did in this episode, grounded in the sampled frames.",
        },
        "task_guess": {
            "type": "string",
            "description": "Your own independent read of the L1 task category from the frames — not simply copied from the provided task label.",
        },
        "task_guess_confidence": {
            "type": "number",
            "description": "0-1 confidence in task_guess.",
        },
        "objects_mentioned": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Objects/tools the instruction text refers to, for downstream hallucination spot-checking.",
        },
    },
    "required": ["instruction", "task_guess", "task_guess_confidence", "objects_mentioned"],
    "additionalProperties": False,
}

HALLUCINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "consistent": {
            "type": "boolean",
            "description": "True if the instruction describes only things actually visible in the frames.",
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any object/action/detail the instruction mentions that is NOT supported by the images. Empty if consistent.",
        },
    },
    "required": ["consistent", "unsupported_claims"],
    "additionalProperties": False,
}

# Phases considered "key manipulation moments" for frame sampling — interior
# frames are drawn from segments in this set so the sample actually spans
# reach -> grasp -> manipulate -> release, not arbitrary timestamps.
_KEY_PHASES = ("grasp", "active_manipulation", "release")


def sample_representative_frames(episode: dict, phase_segments: List[dict]) -> List[int]:
    """
    Pick 3-5 representative frame indices for one episode: episode start,
    one or two interior frames from key-manipulation-phase segments
    (reach->grasp->manipulate->release transitions), and episode end.
    Falls back to evenly spaced interior frames if the episode has too few
    key-phase segments to reach VLM_MIN_SAMPLE_FRAMES on its own — a short
    or simple episode still gets multiple frames, not just start+end.
    """
    start_frame = episode["start_frame"]
    end_frame = episode["end_frame"]
    if end_frame <= start_frame:
        return [start_frame]

    ep_segments = sorted(
        (
            s for s in phase_segments
            if s.get("start_frame", 0) <= end_frame and s.get("end_frame", -1) >= start_frame
        ),
        key=lambda s: s.get("start_frame", 0),
    )

    interior = []
    for seg in ep_segments:
        if seg.get("phase") in _KEY_PHASES:
            seg_start = max(seg.get("start_frame", start_frame), start_frame)
            seg_end = min(seg.get("end_frame", end_frame), end_frame)
            mid = (seg_start + seg_end) // 2
            if start_frame < mid < end_frame:
                interior.append(mid)

    max_interior = cfg.VLM_MAX_SAMPLE_FRAMES - 2
    if len(interior) > max_interior:
        # Spread the selection across the full list rather than always
        # keeping the earliest max_interior key-phase frames.
        step = len(interior) / max_interior
        interior = [interior[int(i * step)] for i in range(max_interior)]

    needed = cfg.VLM_MIN_SAMPLE_FRAMES - 2
    if len(interior) < needed:
        n_extra = needed - len(interior)
        duration = end_frame - start_frame
        evenly_spaced = [
            start_frame + round(duration * (i + 1) / (n_extra + 1))
            for i in range(n_extra)
        ]
        interior = sorted(set(interior) | set(evenly_spaced))

    frames = sorted(set([start_frame] + interior + [end_frame]))
    return frames[: cfg.VLM_MAX_SAMPLE_FRAMES]


def encode_frame_base64(video_path: Path, frame_idx: int) -> Optional[str]:
    """
    Extract one frame from compressed.mp4 and JPEG-encode it as base64.
    Returns None if the frame can't be read (e.g. frame_idx past EOF) —
    callers should skip that frame rather than fail the whole episode.
    """
    import cv2  # lazy import — heavy dependency, only needed on the real (non-template) path

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    import base64
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def estimate_cost_usd(usage: dict) -> float:
    """
    Rough $ cost for one API call from its token usage, using
    config.VLM_PRICING_USD_PER_MTOK. Ignores prompt-cache token types
    (cache_creation_input_tokens / cache_read_input_tokens) — episodes
    sample distinct frames each call, so caching isn't in play here.
    """
    pricing = cfg.VLM_PRICING_USD_PER_MTOK
    input_cost = usage.get("input_tokens", 0) / 1_000_000 * pricing["input"]
    output_cost = usage.get("output_tokens", 0) / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 6)


def should_spotcheck(episode_id: str, rate: float) -> bool:
    """
    Deterministic (hash-based, not random) decision on whether this
    episode gets the extra hallucination-verification call — a spot-check
    sample rather than double-calling on every episode. Deterministic so
    the same episode always gets the same decision across re-runs, and so
    this is unit-testable without mocking randomness.
    """
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def check_task_disagreement(vlm_task_guess: str, classifier_task: str) -> dict:
    """
    Compare the VLM's independent task_guess against 07_task_classify.py's
    label for the same episode. Normalizes case/spacing before comparing
    (the VLM may say "Bolt Tightening", the classifier says
    "bolt_tightening") — this is a heuristic normalization, not semantic
    matching, so near-miss phrasing can still register as a disagreement.
    """
    guess_norm = (vlm_task_guess or "").strip().lower().replace(" ", "_").replace("-", "_")
    classifier_norm = (classifier_task or "").strip().lower().replace(" ", "_").replace("-", "_")
    disagree = bool(guess_norm) and guess_norm != classifier_norm and classifier_norm != "unknown"
    return {
        "disagree": disagree,
        "vlm_task_guess": vlm_task_guess,
        "classifier_task": classifier_task,
    }


def _build_generation_system_prompt(recent_instructions: List[str]) -> str:
    system = (
        "You are labeling egocentric factory-floor video for a robot-training dataset. "
        "You will see several frames sampled from one task episode (episode start, "
        "key manipulation moments, and episode end), plus structured facts extracted "
        "by an upstream pipeline. Write ONE natural-language instruction describing "
        "what the worker did, grounded in what is actually visible in the frames — do "
        "not invent objects, locations, or actions that are not visible. "
        "Separately, look at the frames yourself and state your own independent read "
        "of the task category — do not just repeat the provided task label if the "
        "frames suggest otherwise. Your guess is used as an independent check on the "
        "upstream classifier, so disagreeing when you have real visual evidence is "
        "exactly the useful case, not an error."
    )
    if recent_instructions:
        system += (
            "\n\nAvoid repeating the phrasing or sentence structure of these recent "
            "instructions from the same session:\n- " + "\n- ".join(recent_instructions)
        )
    return system


def generate_instruction_vlm(
    client,
    structured_facts: dict,
    frame_images_b64: List[str],
    recent_instructions: List[str],
) -> dict:
    """
    Call Claude (vision) to generate a frame-grounded instruction plus an
    independent task guess and a list of objects the instruction mentions
    (for hallucination spot-checking). Raises the SDK's typed exceptions
    on API failure — callers decide the fallback (see
    scripts/09_language_ground.py's LANGUAGE_GEN_MODE == "hybrid" path).
    """
    system = _build_generation_system_prompt(recent_instructions)

    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        for b64 in frame_images_b64
    ]
    content.append({
        "type": "text",
        "text": "Structured facts (supporting context, not the only input):\n" + json.dumps(structured_facts, indent=2),
    })

    response = client.messages.create(
        model=cfg.VLM_MODEL,
        max_tokens=cfg.VLM_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": GENERATION_SCHEMA}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)

    return {
        "parsed": parsed,
        "prompt_system": system,
        "raw_response_text": text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "model": response.model,
    }


def check_instruction_hallucination(client, instruction_text: str, frame_images_b64: List[str]) -> dict:
    """
    Second, independent VLM call: show the same frames again alongside the
    already-generated instruction text and ask whether it describes only
    what's actually visible. This is the "spot-check against the actual
    images" hallucination check (v2 addendum §7 revision) — distinct from
    checking against structured_facts, since the VLM path can hallucinate
    something absent from the frames even if it's consistent with the
    structured facts it was given.
    """
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        for b64 in frame_images_b64
    ]
    content.append({
        "type": "text",
        "text": (
            "Here is a generated instruction describing what happens in these frames:\n\n"
            f"\"{instruction_text}\"\n\n"
            "Does this instruction describe ONLY things actually visible in these frames? "
            "List any object, action, or detail it mentions that is NOT supported by the images."
        ),
    })

    response = client.messages.create(
        model=cfg.VLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": HALLUCINATION_SCHEMA}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)

    return {
        "parsed": parsed,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
