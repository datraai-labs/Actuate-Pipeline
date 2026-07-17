"""L6 VLM primitives -- frame sampling, encoding, and the two Claude vision calls.

Ported from v1's `utils/vlm_language.py`, which was already built the right way round: the
client is INJECTED, so every pure function here unit-tests without network and the judge's
red->green gate runs against a fake client before it ever costs money. Kept that seam.

Upgrades over v1:
- the hallucination check grows into a four-dimension LLM-as-judge (hand / object / action /
  global consistency), grounded in perception facts -- see `judge_caption`.
- no sampling parameters: `temperature` is REMOVED on the current model family (400 if sent).
  Determinism is not on offer; the judge relies on structured output + a threshold instead.
- score bounds live in field DESCRIPTIONS, not schema `minimum`/`maximum` -- structured
  outputs do not support numeric constraints; scores are clamped client-side.

API key: read from the environment / .env.local by `get_api_key`. NEVER hardcoded, logged,
or committed. No key -> callers skip annotation with a warning; the pipeline must not fail.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

#: Current Opus model (see claude-api skill: default to the most capable current model).
VLM_MODEL = "claude-opus-4-8"
VLM_MAX_TOKENS = 1024

#: $ per MTok for VLM_MODEL (input, output) -- for the pre-call cost estimate the working
#: discipline requires ("tell me the estimated cost before running").
PRICING_USD_PER_MTOK = {"input": 5.00, "output": 25.00}

#: ~tokens per 1080p->JPEG frame at the API's internal downscaling; used for estimates only.
_EST_TOKENS_PER_IMAGE = 1600

MIN_SAMPLE_FRAMES = 3
MAX_SAMPLE_FRAMES = 5

#: Phases considered key manipulation moments (from v1) -- interior sample frames are drawn
#: from these segments so the sample spans reach->grasp->manipulate->release.
KEY_PHASES = ("grasp", "active_manipulation", "release")

#: Judge dimensions and the flag threshold: any dimension below this is flagged for human
#: review, never silently accepted (EgoLive annotation-QA; v1 §10.5).
JUDGE_DIMENSIONS = ("hand_consistency", "object_consistency", "action_consistency",
                    "global_consistency")
JUDGE_THRESHOLD = 0.7

PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "paraphrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Semantically equivalent rephrasings of the task instruction. "
                           "Each must differ in wording AND sentence structure, not just "
                           "word swaps. All must stay faithful to the original meaning.",
        },
    },
    "required": ["paraphrases"],
    "additionalProperties": False,
}

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "hand": {"type": "string", "description": "What the hand(s) are doing -- side, "
                 "pose, grasp state. Only what is visible."},
        "object": {"type": "string", "description": "Objects/tools visible and being "
                   "manipulated. Only what is visible."},
        "action": {"type": "string", "description": "The action being performed in this "
                   "segment. Only what is visible."},
        "scene": {"type": "string", "description": "The overall scene/workspace context."},
        "instruction": {"type": "string", "description": "One imperative instruction a "
                        "robot could follow for this segment, grounded in the frames."},
    },
    "required": ["hand", "object", "action", "scene", "instruction"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "hand_consistency": {"type": "number", "description":
                             "0-1: does the hand description match the frames?"},
        "object_consistency": {"type": "number", "description":
                               "0-1: are all mentioned objects actually visible? Any "
                               "object not in the frames must drive this LOW."},
        "action_consistency": {"type": "number", "description":
                               "0-1: does the described action match what the frames show?"},
        "global_consistency": {"type": "number", "description":
                               "0-1: overall, does the caption describe ONLY these frames?"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}, "description":
                               "Every object, action, or detail the caption mentions that "
                               "is NOT supported by the images. Empty if fully grounded."},
        "verdict": {"type": "string", "enum": ["consistent", "inconsistent"]},
    },
    "required": [*JUDGE_DIMENSIONS, "unsupported_claims", "verdict"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------ key handling
def get_api_key(explicit: str | None = None) -> str | None:
    """ANTHROPIC_API_KEY from the arg, the environment, or .env.local at the repo root.

    Returns None when absent -- callers WARN AND SKIP, they do not fail the pipeline.
    The key's value must never be printed, logged, or written anywhere by this module.
    """
    if explicit:
        return explicit
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parents[3] / ".env.local"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY"):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return None


def make_client(api_key: str | None = None):
    """Real Anthropic client, or None when no key is available (degrade, don't fail)."""
    key = get_api_key(api_key)
    if key is None:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key)


# ------------------------------------------------------------------ pure helpers (v1 port)
def sample_representative_frames(start_frame: int, end_frame: int,
                                 phase_segments: list[dict]) -> list[int]:
    """3-5 representative frames: start, key-phase interiors, end (v1 logic, de-configed)."""
    if end_frame <= start_frame:
        return [start_frame]

    interior = []
    for seg in sorted(phase_segments, key=lambda s: s.get("start_frame", 0)):
        if seg.get("phase") in KEY_PHASES:
            s = max(seg.get("start_frame", start_frame), start_frame)
            e = min(seg.get("end_frame", end_frame), end_frame)
            mid = (s + e) // 2
            if start_frame < mid < end_frame:
                interior.append(mid)

    max_interior = MAX_SAMPLE_FRAMES - 2
    if len(interior) > max_interior:
        step = len(interior) / max_interior
        interior = [interior[int(i * step)] for i in range(max_interior)]

    needed = MIN_SAMPLE_FRAMES - 2
    if len(interior) < needed:
        n_extra = needed - len(interior)
        duration = end_frame - start_frame
        interior = sorted(set(interior) | {
            start_frame + round(duration * (i + 1) / (n_extra + 1)) for i in range(n_extra)})

    return sorted(set([start_frame] + interior + [end_frame]))[:MAX_SAMPLE_FRAMES]


def encode_frame_base64(video_path: Path, frame_idx: int) -> str | None:
    """One frame -> base64 JPEG; None when unreadable (skip the frame, not the episode)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def cost_usd(usage: dict) -> float:
    """Actual $ for one call from its reported token usage."""
    return round(usage.get("input_tokens", 0) / 1e6 * PRICING_USD_PER_MTOK["input"]
                 + usage.get("output_tokens", 0) / 1e6 * PRICING_USD_PER_MTOK["output"], 6)


def estimate_annotation_cost(n_segments: int, frames_per_call: int = 4,
                             n_judge_calls: int | None = None) -> float:
    """Pre-call estimate for one episode's annotation, in USD (upper-bound-ish).

    paraphrase call + one caption call per segment + one judge call per caption.
    """
    n_judge = n_segments if n_judge_calls is None else n_judge_calls
    per_vision_call = frames_per_call * _EST_TOKENS_PER_IMAGE + 800   # images + prompt
    calls_in = 500 + (n_segments + n_judge) * per_vision_call         # paraphrase is text-only
    calls_out = (1 + n_segments + n_judge) * 400
    return round(calls_in / 1e6 * PRICING_USD_PER_MTOK["input"]
                 + calls_out / 1e6 * PRICING_USD_PER_MTOK["output"], 4)


def _image_blocks(frames_b64: list[str]) -> list[dict]:
    return [{"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
            for b64 in frames_b64]


def _json_response(response) -> tuple[dict, dict]:
    """(parsed json, usage) from a structured-output message."""
    text = next(b.text for b in response.content if b.type == "text")
    usage = {"input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    return json.loads(text), usage


# ------------------------------------------------------------------ the three API calls
def call_paraphrase(client, task: str, n: int) -> tuple[list[str], dict]:
    """N semantically diverse paraphrases of the operator-verified task string."""
    response = client.messages.create(
        model=VLM_MODEL,
        max_tokens=VLM_MAX_TOKENS,
        system=("You write training-data paraphrases of robot task instructions (TRI LBM "
                "protocol: one is sampled per training step, so DIVERSITY is the point). "
                f"Produce exactly {n} rephrasings that differ in wording and sentence "
                "structure -- not word swaps -- while preserving the exact meaning. "
                "Grammatically correct, imperative mood."),
        messages=[{"role": "user", "content": f"Task instruction: {task!r}"}],
        output_config={"format": {"type": "json_schema", "schema": PARAPHRASE_SCHEMA}},
    )
    parsed, usage = _json_response(response)
    return list(parsed["paraphrases"])[:n], usage


def call_caption(client, frames_b64: list[str], facts: dict) -> tuple[dict, dict]:
    """Structured caption (hand/object/action/scene + instruction) for one segment."""
    content = _image_blocks(frames_b64)
    content.append({"type": "text", "text":
                    "Structured facts from the perception pipeline (supporting context, "
                    "not the only input):\n" + json.dumps(facts, indent=2, sort_keys=True)})
    response = client.messages.create(
        model=VLM_MODEL,
        max_tokens=VLM_MAX_TOKENS,
        system=("You label egocentric manipulation video for a robot-training dataset. "
                "These frames are ONE temporal segment of an episode. Describe ONLY what "
                "is visible -- never invent objects, locations, or actions."),
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": CAPTION_SCHEMA}},
    )
    return _json_response(response)


def call_judge(client, caption: dict, frames_b64: list[str], facts: dict) -> tuple[dict, dict]:
    """Independent second call: score the caption against the frames, 4 dimensions.

    Never the generating call scoring itself. Grounding the judge in the perception facts
    (detected objects, interaction state) is what makes it more than a vibe check.
    """
    content = _image_blocks(frames_b64)
    content.append({"type": "text", "text": (
        "A generated caption for these frames:\n\n"
        + json.dumps(caption, indent=2, sort_keys=True)
        + "\n\nIndependent perception-pipeline facts about the same frames:\n"
        + json.dumps(facts, indent=2, sort_keys=True)
        + "\n\nScore the caption's consistency with the FRAMES on each dimension (0-1). "
          "List every object, action, or detail the caption mentions that is not "
          "supported by the images."
    )})
    response = client.messages.create(
        model=VLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    parsed, usage = _json_response(response)
    for dim in JUDGE_DIMENSIONS:          # schema can't carry numeric bounds; clamp here
        parsed[dim] = min(1.0, max(0.0, float(parsed[dim])))
    return parsed, usage
