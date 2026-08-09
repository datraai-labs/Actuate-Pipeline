"""L6 orchestration -- paraphrases, subtask instructions, subgoal anchors, judge gate.

`annotate` fills the canonical schema fields that have existed since v1 of the schema but
were never populated: `task_paraphrases[]`, `subtasks[]` (start/end/instruction), and
`subgoal_frames[]`. Temporal structure comes from the EXISTING v1 phase segmentation
(phases.json) -- the phase boundaries ARE the subgoal anchors (π0.7's "subgoal images");
this module does not re-segment.

Every VLM-generated caption passes the LLM-as-judge (`vlm.call_judge`). A caption scoring
below threshold on any dimension is FLAGGED FOR REVIEW and its instruction still ships only
with `confidence` set from the judge -- flagged, never silently accepted, never silently
dropped (a dropped segment would hide the disagreement a reviewer needs to see).

No API key -> warn and return a skipped report. The pipeline must not fail on a missing
key; language annotation is enrichment, not a gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from actuate.language import vlm
from actuate.schema import CanonicalEpisode
from actuate.schema.episode import SubgoalFrame, Subtask


@dataclass
class ConsistencyScore:
    hand: float
    object: float
    action: float
    global_: float
    unsupported_claims: list[str]
    verdict: str

    @property
    def min_score(self) -> float:
        return min(self.hand, self.object, self.action, self.global_)

    @property
    def flagged(self) -> bool:
        return self.min_score < vlm.JUDGE_THRESHOLD or self.verdict != "consistent"


@dataclass
class AnnotationReport:
    episode_id: str
    skipped: bool                          # True when no key / no inputs -- NOT a failure
    skip_reason: str | None
    paraphrases: list[str] = field(default_factory=list)
    subtasks: list[Subtask] = field(default_factory=list)
    subgoal_frames: list[SubgoalFrame] = field(default_factory=list)
    judge_scores: list[ConsistencyScore] = field(default_factory=list)
    flagged_for_review: int = 0
    cost_usd: float = 0.0
    episode: CanonicalEpisode | None = None    # updated copy, when not skipped
    task_disagreement: bool = False
    vlm_task_guesses: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped:
            return f"episode {self.episode_id}: annotation SKIPPED ({self.skip_reason})"
        return (f"episode {self.episode_id}: {len(self.paraphrases)} paraphrases | "
                f"{len(self.subtasks)} subtasks | {len(self.subgoal_frames)} subgoal frames "
                f"| {self.flagged_for_review} caption(s) flagged for review | "
                f"${self.cost_usd:.4f}")


#: Rule-based rephrasing templates for the no-key fallback. Not as good as an LLM -- but a
#: single string is worse still (TRI LBM samples one paraphrase per training step, so zero
#: variety hurts robustness). These restructure the sentence rather than swap synonyms.
_PARAPHRASE_TEMPLATES = (
    "{t}.",
    "Please {tl}.",
    "Your task: {tl}.",
    "Go ahead and {tl}.",
    "The goal is to {tl}.",
)
#: Light synonym map for the restructuring fallback (deliberately small and safe).
_SYNONYMS = {"sort": "organize", "staple": "fasten", "pick up": "grab",
             "place": "put", "move": "transport", "workbench": "work surface"}


def _rule_based_paraphrases(task_str: str, n: int) -> list[str]:
    base = task_str.strip().rstrip(".")
    lowered = base[0].lower() + base[1:] if base else base
    swapped = lowered
    for a, b in _SYNONYMS.items():
        swapped = swapped.replace(a, b)
    variants: list[str] = []
    for tmpl in _PARAPHRASE_TEMPLATES:
        variants.append(tmpl.format(t=base, tl=lowered))
        if swapped != lowered:
            variants.append(tmpl.format(t=swapped.capitalize(), tl=swapped))
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:n]


def paraphrase(task_str: str, n: int = 5, *, client=None, api_key: str | None = None
               ) -> list[str]:
    """N diverse paraphrases of a task string.

    LLM path when a client/key is available; otherwise a RULE-BASED fallback (sentence
    restructuring + a small synonym map) so a no-key run still ships more than one string.
    """
    client = client or vlm.make_client(api_key)
    if client is None:
        return _rule_based_paraphrases(task_str, n)
    result, _ = vlm.call_paraphrase(client, task_str, n)
    return result


def segment_subtasks(canonical, api_key: str | None = None, *, session_dir=None,
                     video=None, client=None) -> list:
    """Public subtask segmentation (Part B interface).

    Uses the existing v1 phase boundaries as subtask spans and generates a per-segment
    instruction via VLM. Delegates to `annotate` (which owns the shared per-segment
    caption+judge loop) and returns just the `Subtask` list; a no-key or no-video run
    returns []. Kept as its own entry point per the Master Spec §L6 interface list.
    """
    from pathlib import Path

    report = annotate(canonical, api_key, session_dir=session_dir,
                      video=Path(video) if video else None, client=client)
    return list(report.subtasks)


def judge(caption: dict, frames_b64: list[str], facts: dict, *, client=None,
          api_key: str | None = None) -> ConsistencyScore:
    """LLM-as-judge for one caption against its frames (independent second call)."""
    client = client or vlm.make_client(api_key)
    if client is None:
        raise RuntimeError("no ANTHROPIC_API_KEY available; cannot judge")
    parsed, _ = vlm.call_judge(client, caption, frames_b64, facts)
    return ConsistencyScore(
        hand=parsed["hand_consistency"], object=parsed["object_consistency"],
        action=parsed["action_consistency"], global_=parsed["global_consistency"],
        unsupported_claims=list(parsed["unsupported_claims"]), verdict=parsed["verdict"],
    )


#: Adjacent same-phase segments closer than this many frames apart are ONE subtask.
_MERGE_GAP_FRAMES = 90   # 3 s at 30 fps


def _load_segments(session_dir: Path) -> list[dict]:
    """v1 phase segments, non-idle, with flicker healed.

    v1's rule-engine segmentation flickers -- the real capture yields 15 non-idle segments
    over 95 s, many of them the same phase separated by sub-3-second gaps. Those are one
    demonstration unit, not several; merging adjacent same-phase segments gives better
    subtask structure AND fewer (billed) VLM calls. Distinct phases are never merged.
    """
    p = Path(session_dir) / "phases.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data if isinstance(data, list) else data.get("segments", data.get("phases", []))
    merged: list[dict] = []
    for s in sorted((s for s in raw if s.get("phase") != "idle"),
                    key=lambda s: s.get("start_frame", 0)):
        if (merged and merged[-1]["phase"] == s.get("phase")
                and s.get("start_frame", 0) - merged[-1]["end_frame"] <= _MERGE_GAP_FRAMES):
            merged[-1]["end_frame"] = s["end_frame"]
        else:
            merged.append({"phase": s.get("phase"), "start_frame": s.get("start_frame", 0),
                           "end_frame": s.get("end_frame", 0)})
    return merged


def _facts_for(episode: CanonicalEpisode, start: int, end: int) -> dict:
    """Perception-derived facts for one segment -- what grounds the judge."""
    hands, objects, states = set(), set(), set()
    for f in episode.frames:
        if not (start <= f.frame_idx <= end):
            continue
        hands.update(s.value for s in f.hands)
        objects.update(f.objects.keys())
        if f.interaction_state is not None:
            states.add(f.interaction_state.value)
    return {
        "task": episode.task,
        "hands_detected": sorted(hands),
        "objects_tracked": sorted(objects),
        "interaction_states": sorted(states),
        "frame_range": [start, end],
    }


def annotate(
    episode: CanonicalEpisode,
    api_key: str | None = None,
    *,
    session_dir: Path | None = None,
    video: Path | None = None,
    n_paraphrases: int = 5,
    client=None,
) -> AnnotationReport:
    """Fill task_paraphrases / subtasks / subgoal_frames on a canonical episode.

    `client` is injectable for tests; otherwise built from the key. Missing key or missing
    inputs -> a SKIPPED report with the reason, never an exception.
    """
    def _skip(reason: str) -> AnnotationReport:
        return AnnotationReport(episode_id=episode.episode_id, skipped=True,
                                skip_reason=reason)

    if not episode.task:
        return _skip("episode has no operator-verified task; paraphrasing a missing task "
                     "would invent a label")

    client = client or vlm.make_client(api_key)
    if client is None:
        # degrade, don't skip: rule-based paraphrases still ship (better than one string);
        # captions/judge (which need the VLM) are the parts that can't run without a key.
        paras = _rule_based_paraphrases(episode.task, n_paraphrases)
        updated = episode.model_copy(update={"task_paraphrases": tuple(paras)})
        return AnnotationReport(
            episode_id=episode.episode_id, skipped=False,
            skip_reason="no ANTHROPIC_API_KEY: rule-based paraphrases only; VLM subtasks + "
                        "judge skipped",
            paraphrases=paras, episode=updated)
    if video is None or not Path(video).exists():
        return _skip(f"no video at {video}; captions and the judge need frames")

    total_cost = 0.0

    # ---- paraphrases -------------------------------------------------------------
    paras, usage = vlm.call_paraphrase(client, episode.task, n_paraphrases)
    total_cost += vlm.cost_usd(usage)

    # ---- subtasks + subgoals from the EXISTING phase boundaries -------------------
    segments = _load_segments(session_dir) if session_dir else []
    frame_ids = {f.frame_idx for f in episode.frames}
    lo, hi = (min(frame_ids), max(frame_ids)) if frame_ids else (0, 0)
    segments = [s for s in segments
                if s.get("phase") != "idle"
                and s.get("start_frame", 0) <= hi and s.get("end_frame", 0) >= lo]

    subtasks: list[Subtask] = []
    subgoals: list[SubgoalFrame] = []
    scores: list[ConsistencyScore] = []
    task_guesses: list[str] = []

    for seg in segments:
        start = max(int(seg["start_frame"]), lo)
        end = min(int(seg["end_frame"]), hi)
        sample = vlm.sample_representative_frames(start, end, [seg])
        frames_b64 = [b for i in sample if (b := vlm.encode_frame_base64(video, i))]
        if not frames_b64:
            continue

        facts = _facts_for(episode, start, end)
        facts["phase"] = seg.get("phase")
        caption, usage = vlm.call_caption(client, frames_b64, facts)
        total_cost += vlm.cost_usd(usage)
        if caption.get("task_guess"):
            task_guesses.append(str(caption["task_guess"]))

        parsed, usage = vlm.call_judge(client, caption, frames_b64, facts)
        total_cost += vlm.cost_usd(usage)
        score = ConsistencyScore(
            hand=parsed["hand_consistency"], object=parsed["object_consistency"],
            action=parsed["action_consistency"], global_=parsed["global_consistency"],
            unsupported_claims=list(parsed["unsupported_claims"]),
            verdict=parsed["verdict"],
        )
        scores.append(score)

        subtasks.append(Subtask(
            instruction=caption["instruction"], start_frame=start, end_frame=end,
            confidence=score.min_score,      # the judge's floor IS the confidence
        ))
        # the segment boundary is the subgoal anchor (π0.7 subgoal image)
        subgoals.append(SubgoalFrame(frame_idx=end, label=seg.get("phase")))

    disagreements = [
        guess for guess in task_guesses if vlm.task_disagreement(guess, episode.task)
    ]
    task_disagrees = bool(disagreements)
    flagged = sum(1 for s in scores if s.flagged) + int(task_disagrees)
    derivation_notes = dict(episode.derivation_notes)
    if task_disagrees:
        derivation_notes["task_classification_disagreement"] = (
            f"VLM independent task read disagreed with {episode.task!r}: "
            f"{sorted(set(disagreements))}. Routed to human review; not silently resolved."
        )
    updated = episode.model_copy(update={
        "task_paraphrases": tuple(paras),
        "subtasks": tuple(subtasks),
        "subgoal_frames": tuple(subgoals),
        "derivation_notes": derivation_notes,
    })
    return AnnotationReport(
        episode_id=episode.episode_id, skipped=False, skip_reason=None,
        paraphrases=paras, subtasks=subtasks, subgoal_frames=subgoals,
        judge_scores=scores, flagged_for_review=flagged,
        cost_usd=round(total_cost, 4), episode=updated,
        task_disagreement=task_disagrees, vlm_task_guesses=task_guesses,
    )
