from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from actuate.schema.semantic import (
    ClaimReview,
    ClaimStatus,
    EvidenceRef,
    ReviewDecision,
    SemanticClaim,
    VLMRecord,
    vlm_record_json_schema,
)

CAPTURE_ID = "a" * 64


def _evidence(**updates):
    values = {
        "capture_id": CAPTURE_ID,
        "source_content_hash": CAPTURE_ID,
        "source_start_ms": 0,
        "source_end_ms": 1000,
        "frame_indices": (0,),
        "preview_uri": "work/preview.webp",
    }
    values.update(updates)
    return EvidenceRef(**values)


def _visible_claim():
    return {
        "kind": "visible_fact",
        "claim_id": "claim-1",
        "field": "activity",
        "value": "a hand moves toward an object",
        "evidence": [{"frame_id": "frame-1", "timestamp_ms": 500}],
        "clarity": "clear",
    }


def test_prompt_record_is_a_discriminated_contract():
    record = VLMRecord(
        scope="window",
        source_record_id="window-1",
        claims=(_visible_claim(),),
        review_flags=frozenset(),
    )
    assert record.claims[0].kind == "visible_fact"


def test_generated_schema_keeps_claim_field_authority_separate():
    schema = vlm_record_json_schema()
    assert schema["$id"] == "corpus-b-vlm-record-v1"
    definitions = schema["$defs"]
    assert set(definitions["VisibleField"]["enum"]) == {
        "activity",
        "hand_visibility",
        "visible_hand_count",
        "object",
        "tool",
        "environment",
        "visual_quality",
        "possible_pii",
        "failure_or_recovery",
    }
    assert set(definitions["InferenceField"]["enum"]) >= {
        "task",
        "subtask",
        "completion_state",
        "corpus_relation",
    }


def test_task_completion_and_corpus_relations_are_not_visible_facts():
    for field in ("task", "subtask", "completion_state", "corpus_relation"):
        claim = _visible_claim() | {"field": field}
        with pytest.raises(ValidationError):
            VLMRecord(
                scope="window",
                source_record_id="window-1",
                claims=(claim,),
                review_flags=frozenset(),
            )


def test_source_hash_must_equal_capture_identity():
    with pytest.raises(ValidationError, match="must equal capture_id"):
        _evidence(source_content_hash="b" * 64)


def test_evidence_rejects_empty_frames_inverted_span_and_invalid_box():
    with pytest.raises(ValidationError):
        _evidence(frame_indices=())
    with pytest.raises(ValidationError, match="must not precede"):
        _evidence(source_start_ms=2, source_end_ms=1)
    with pytest.raises(ValidationError, match="ordered normalized"):
        _evidence(bbox_xyxy=(0.8, 0.1, 0.2, 0.9))


def test_model_claim_cannot_create_human_verified_state():
    with pytest.raises(ValidationError):
        SemanticClaim(
            call_id="call-1",
            claim=_visible_claim(),
            evidence_refs=(_evidence(),),
            status="human_verified",
        )


def test_unknown_is_preserved_instead_of_defaulted():
    claim = SemanticClaim(
        call_id="call-1",
        claim={
            "kind": "unknown",
            "claim_id": "claim-1",
            "field": "tool",
            "reason": "not_visible",
        },
        evidence_refs=(),
        status=ClaimStatus.UNKNOWN,
    )
    assert claim.status is ClaimStatus.UNKNOWN


def test_human_correction_requires_reviewer_time_and_corrected_claim():
    with pytest.raises(ValidationError, match="corrected_claim"):
        ClaimReview(
            review_id="review-1",
            claim_id="claim-1",
            decision=ReviewDecision.HUMAN_CORRECTED,
            reviewer_id="reviewer-1",
            reviewed_at=datetime.now(timezone.utc),
        )
    naive = datetime.now(timezone.utc).replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone"):
        ClaimReview(
            review_id="review-1",
            claim_id="claim-1",
            decision=ReviewDecision.HUMAN_VERIFIED,
            reviewer_id="reviewer-1",
            reviewed_at=naive,
        )
