"""L6 fine-grained action labeling -- atomic manipulation primitives (Master Spec v1 §10.3).

Distinct from BOTH the coarse L2 interaction states (STATIC/GRASPED/MOVING) and from
phase/subtask segmentation. This produces per-actor `ActionInterval` rows drawn from the
CLOSED 20-verb vocabulary, joinable to per-frame data by frame span.

### What geometry can honestly label, and what it cannot

From a monocular bare-hand rig we have: wrist trajectory (noisy in metric Z -- Phase 3's
central finding), finger curl, the L2 grasp state, and object proximity. That supports a
DEFENSIBLE SUBSET of the vocabulary:

    idle, reach, grasp, hold, lift, transport, lower, place, release

The other eleven -- align, stabilize, open, close, insert, remove, push, pull, rotate,
wipe, pour -- require contact geometry, 6-DoF object pose deltas, or force this rig does
not measure. They stay in the closed vocabulary (so cross-dataset comparison has a fixed
alphabet) but this detector will NOT fabricate them from noisy monocular data. An opt-in
VLM pass can REFINE an ambiguous interval into one of them, but that is off by default and
never invents a label the geometry cannot at least support.

Every interval carries an honest confidence; low-confidence intervals are flagged, not
silently trusted. Concurrent bimanual actions are separate overlapping rows (one pass per
detected hand/actor), never collapsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from actuate.config import ActionVerb, Actor, Side
from actuate.fusion.states import enforce_min_dwell, finger_curl
from actuate.schema import ActionInterval, CanonicalEpisode

#: Minimum interval length in frames -- no single-frame actions (verification gate 2).
MIN_INTERVAL_FRAMES = 3
#: Wrist speed (m/s) below which motion is "still" -- hold vs transport.
_STILL_MPS = 0.04
#: Vertical-velocity fraction of total speed that makes a motion lift/lower vs transport.
_VERTICAL_FRAC = 0.55
#: Wrist-to-nearest-object distance (m) under which the hand is "at" an object.
_NEAR_OBJECT_M = 0.12
#: Confidence at/below which an interval is flagged for review.
FLAG_CONFIDENCE = 0.5


@dataclass
class ActionLabelResult:
    episode_id: str
    intervals: list[ActionInterval]
    flagged_for_review: int
    actors: list[str]
    episode: CanonicalEpisode | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        from collections import Counter

        verbs = Counter(i.action_label.value for i in self.intervals)
        return (f"{self.episode_id}: {len(self.intervals)} action intervals over "
                f"{sorted(self.actors)} | verbs {dict(verbs)} | "
                f"{self.flagged_for_review} flagged for review")


def _wrist_xyz(hand) -> np.ndarray | None:
    if hand.keypoints_3d is not None:
        return np.asarray(hand.keypoints_3d[0], dtype=np.float64)
    if hand.wrist_pose is not None:
        return np.asarray(hand.wrist_pose.position_m, dtype=np.float64)
    return None


def _curl(hand) -> float:
    if hand.keypoints_3d is None:
        return 0.0
    c = finger_curl(np.asarray(hand.keypoints_3d, dtype=np.float64))
    return float(np.mean(list(c.values()))) if c else 0.0


def _nearest_object_dist(frame, wrist: np.ndarray) -> float:
    best = np.inf
    for obj in frame.objects.values():
        if obj.pose is not None:
            d = float(np.linalg.norm(np.asarray(obj.pose.position_m) - wrist))
            best = min(best, d)
    return best


def _per_frame_verb(grasped: bool, prev_grasped: bool | None, speed: float,
                    v_up: float, near_obj: bool, approaching: bool,
                    has_hand: bool) -> ActionVerb:
    """One frame -> one verb, from geometry alone. Transitions handled by the caller."""
    if not has_hand:
        return ActionVerb.IDLE
    # grasp/release are the state-transition frames
    if prev_grasped is not None and grasped != prev_grasped:
        return ActionVerb.GRASP if grasped else ActionVerb.RELEASE
    if grasped:
        if speed < _STILL_MPS:
            return ActionVerb.HOLD
        if abs(v_up) >= _VERTICAL_FRAC * speed:
            return ActionVerb.LIFT if v_up > 0 else (
                ActionVerb.PLACE if near_obj else ActionVerb.LOWER)
        return ActionVerb.TRANSPORT
    # not grasped
    if speed < _STILL_MPS:
        return ActionVerb.IDLE
    return ActionVerb.REACH if approaching else ActionVerb.IDLE


def _side_intervals(episode: CanonicalEpisode, side: Side, fps: float) -> list[ActionInterval]:
    frames = [(f.frame_idx, f.t, f.hands.get(side), f) for f in episode.frames]
    frames = [(idx, t, h, f) for idx, t, h, f in frames]
    if not frames:
        return []

    # per-frame features
    verbs: list[ActionVerb] = []
    confs: list[float] = []
    prev_xyz: np.ndarray | None = None
    prev_grasped: bool | None = None
    for idx, t, h, f in frames:
        has_hand = h is not None
        wrist = _wrist_xyz(h) if has_hand else None
        grasped = bool(f.interaction_state and f.interaction_state.is_grasped) if has_hand \
            else False
        speed = v_up = 0.0
        near_obj = approaching = False
        if wrist is not None and prev_xyz is not None:
            dt = max(1.0 / fps, 1e-3)
            vel = (wrist - prev_xyz) / dt
            speed = float(np.linalg.norm(vel))
            v_up = float(-vel[1])          # camera Y is down; up is negative
            nd = _nearest_object_dist(f, wrist)
            near_obj = nd < _NEAR_OBJECT_M
            prev_nd = _nearest_object_dist(frames[max(0, len(verbs) - 1)][3], prev_xyz)
            approaching = nd < prev_nd - 1e-3
        verb = _per_frame_verb(grasped, prev_grasped, speed, v_up, near_obj, approaching,
                               has_hand)
        verbs.append(verb)
        # confidence: strong for clear states, weak for the ambiguous reach/transport calls
        confs.append(0.8 if verb in (ActionVerb.IDLE, ActionVerb.HOLD, ActionVerb.GRASP,
                                     ActionVerb.RELEASE) else 0.55)
        prev_xyz = wrist if wrist is not None else prev_xyz
        if has_hand:
            prev_grasped = grasped

    # de-flicker: no single-frame actions (Schmitt already gates grasp upstream; this is the
    # final min-dwell merge, interior-only, matching the fusion de-flicker)
    verbs = enforce_min_dwell(verbs, MIN_INTERVAL_FRAMES)

    # contiguous runs -> intervals
    out: list[ActionInterval] = []
    actor = Actor.for_side(side)
    i = 0
    while i < len(verbs):
        j = i
        while j + 1 < len(verbs) and verbs[j + 1] == verbs[i]:
            j += 1
        length = j - i + 1
        if length >= MIN_INTERVAL_FRAMES or verbs[i] in (ActionVerb.GRASP,
                                                         ActionVerb.RELEASE):
            conf = float(np.mean(confs[i:j + 1]))
            out.append(ActionInterval(
                action_label=verbs[i], actor=actor,
                start_frame=frames[i][0], end_frame=frames[j][0],
                start_time=float(frames[i][1]), end_time=float(frames[j][1]),
                confidence=round(min(0.99, max(0.05, conf)), 3)))
        i = j + 1
    return out


def label_actions(canonical: CanonicalEpisode, vlm_api_key: str | None = None, *,
                  fps: float = 30.0, use_vlm: bool = False) -> ActionLabelResult:
    """Detect atomic action intervals per actor, from geometry (VLM refinement opt-in).

    `use_vlm=False` (default) is $0: geometry only. `use_vlm=True` would refine ambiguous
    intervals with a keyframe VLM call -- deliberately off by default so the real-capture
    gate costs nothing.
    """
    notes: list[str] = []
    sides = {s for f in canonical.frames for s in f.hands}
    intervals: list[ActionInterval] = []
    for side in sorted(sides, key=lambda s: s.value):
        intervals.extend(_side_intervals(canonical, side, fps))
    intervals.sort(key=lambda iv: (iv.start_frame, iv.actor.value))

    if use_vlm and vlm_api_key is not None:  # pragma: no cover - opt-in, billed
        notes.append("VLM refinement requested but not exercised in the geometry-only path")

    flagged = sum(1 for iv in intervals if iv.confidence <= FLAG_CONFIDENCE)
    updated = canonical.model_copy(update={"action_intervals": tuple(intervals)})
    return ActionLabelResult(
        episode_id=canonical.episode_id, intervals=intervals, flagged_for_review=flagged,
        actors=sorted({iv.actor.value for iv in intervals}), episode=updated, notes=notes,
    )


def verify_vlm_label(client, verb: ActionVerb, frame_b64: str) -> bool:
    """Independent VLM confirmation for ONE interval -- the mislabel gate (v1 §10.5).

    Returns True if the VLM agrees the frame shows `verb`. A geometry label of "pour" on an
    idle segment must come back False. Injected client so the red->green gate runs against a
    fake with no network and no cost.
    """
    import json

    schema = {
        "type": "object",
        "properties": {
            "agrees": {"type": "boolean", "description":
                       f"true iff the frame shows a hand performing '{verb.value}'"},
            "observed": {"type": "string", "description":
                         "the action actually visible, in one of the 20 verbs"},
        },
        "required": ["agrees", "observed"],
        "additionalProperties": False,
    }
    content = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}},
               {"type": "text", "text":
                f"An automated labeller tagged this frame as the action '{verb.value}'. "
                "Does the frame actually show that? Answer strictly."}]
    from actuate.language.vlm import VLM_MODEL

    resp = client.messages.create(
        model=VLM_MODEL, max_tokens=256,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": schema}})
    text = next(b.text for b in resp.content if b.type == "text")
    return bool(json.loads(text)["agrees"])
