"""L6 -- Language rich-context: paraphrases, subtask instructions, LLM-as-judge.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L6.

Ported from v1's working `utils/vlm_language.py` (injected client -- testable without
network) and upgraded: multi-paraphrase generation (TRI LBM), subtask instructions anchored
on the EXISTING v1 phase boundaries (π0.7 subgoal images), structured hand/object/action/
scene captions, and a four-dimension LLM-as-judge that FLAGS sub-threshold captions for
review rather than silently accepting them (EgoLive annotation-QA; v1 §10.5).

API key: ANTHROPIC_API_KEY env or .env.local -- never hardcoded, never logged. Missing key
-> annotation SKIPS with a warning; the pipeline does not fail.
"""

from __future__ import annotations

from actuate.language.actions import (
    ActionLabelResult,
    label_actions,
    verify_vlm_label,
)
from actuate.language.annotate import (
    AnnotationReport,
    ConsistencyScore,
    annotate,
    judge,
    paraphrase,
    segment_subtasks,
)
from actuate.language.vlm import (
    JUDGE_THRESHOLD,
    VLM_MODEL,
    estimate_annotation_cost,
    get_api_key,
    make_client,
)

__all__ = [
    "JUDGE_THRESHOLD",
    "VLM_MODEL",
    "ActionLabelResult",
    "AnnotationReport",
    "ConsistencyScore",
    "annotate",
    "estimate_annotation_cost",
    "get_api_key",
    "judge",
    "label_actions",
    "make_client",
    "paraphrase",
    "segment_subtasks",
    "verify_vlm_label",
]
