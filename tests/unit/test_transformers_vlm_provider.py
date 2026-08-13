from __future__ import annotations

import json

import pytest

from actuate.language.providers.transformers_vlm import (
    SemanticOutputError,
    build_observer_prompt,
    parse_observer_output,
)
from actuate.language.semantic import ObserverRequest
from actuate.schema.semantic import ClaimStatus, EvidenceRef


def _request() -> ObserverRequest:
    return ObserverRequest(
        evidence=(
            EvidenceRef(
                capture_id="a" * 64,
                source_content_hash="a" * 64,
                source_start_ms=1234,
                source_end_ms=2000,
                frame_indices=(37,),
                preview_uri="preview/frame-37.jpg",
            ),
        )
    )


def _record(source_record_id: str) -> dict:
    claims = [
        {
            "kind": "visible_fact",
            "claim_id": "activity-1",
            "field": "activity",
            "value": "Two gloved hands hold a purple cloth.",
            "evidence": [{"frame_id": "frame-0-37", "timestamp_ms": 1234}],
            "clarity": "clear",
        }
    ]
    claims.extend(
        {
            "kind": "unknown",
            "claim_id": f"unknown-{field}",
            "field": field,
            "reason": "insufficient_temporal_context",
        }
        for field in (
            "hand_visibility",
            "visible_hand_count",
            "object",
            "tool",
            "environment",
            "visual_quality",
            "possible_pii",
            "failure_or_recovery",
        )
    )
    return {
        "scope": "window",
        "source_record_id": source_record_id,
        "claims": claims,
        "review_flags": ["insufficient_visual_evidence"],
    }


def test_conforming_output_becomes_only_proposed_or_unknown_claims():
    request = _request()
    _, _, source_record_id = build_observer_prompt(request)

    output = parse_observer_output(
        json.dumps(_record(source_record_id)),
        request,
        source_record_id,
        "call-1",
    )

    assert output.claims[0].status is ClaimStatus.MODEL_PROPOSED
    assert output.claims[0].evidence_refs == request.evidence
    assert all(claim.status is ClaimStatus.UNKNOWN for claim in output.claims[1:])


@pytest.mark.parametrize(
    "raw_response",
    (
        "visible_fact: the person is folding a purple cloth",
        json.dumps(
            {
                "scope": "window",
                "source_record_id": "window-1",
                "visible_fact": ["a hand holds cloth"],
                "inference": [],
                "unknown": [],
            }
        ),
    ),
)
def test_unconstrained_smoke_shapes_are_rejected(raw_response):
    request = _request()
    _, _, source_record_id = build_observer_prompt(request)
    with pytest.raises(SemanticOutputError):
        parse_observer_output(raw_response, request, source_record_id, "call-1")


def test_unknown_frame_citation_is_rejected():
    request = _request()
    _, _, source_record_id = build_observer_prompt(request)
    record = _record(source_record_id)
    record["claims"][0]["evidence"][0]["frame_id"] = "invented-frame"

    with pytest.raises(SemanticOutputError, match="unknown evidence"):
        parse_observer_output(json.dumps(record), request, source_record_id, "call-1")


def test_forbidden_authority_claim_is_rejected():
    request = _request()
    _, _, source_record_id = build_observer_prompt(request)
    record = _record(source_record_id)
    record["claims"][0]["value"] = "Consent granted for this recording."

    with pytest.raises(SemanticOutputError, match="forbidden authority"):
        parse_observer_output(json.dumps(record), request, source_record_id, "call-1")


def test_initial_output_must_cover_every_window_field():
    request = _request()
    _, _, source_record_id = build_observer_prompt(request)
    record = _record(source_record_id)
    record["claims"] = record["claims"][:1]

    with pytest.raises(SemanticOutputError, match="omitted required fields"):
        parse_observer_output(json.dumps(record), request, source_record_id, "call-1")


def test_inference_must_cite_a_visible_claim():
    request = _request()
    _, _, source_record_id = build_observer_prompt(request)
    record = _record(source_record_id)
    record["claims"][0] = {
        "kind": "inference",
        "claim_id": "activity-1",
        "field": "activity",
        "value": "folding cloth",
        "based_on_claim_ids": ["missing-visible-claim"],
        "reason": "the cloth changes shape",
        "alternatives": [],
    }

    with pytest.raises(SemanticOutputError, match="non-visible"):
        parse_observer_output(json.dumps(record), request, source_record_id, "call-1")
