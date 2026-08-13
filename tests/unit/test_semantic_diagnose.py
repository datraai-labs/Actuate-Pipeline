from __future__ import annotations

from collections import defaultdict, deque

from actuate.language.semantic import (
    BOUNDARY_REVIEW_THRESHOLD_MS,
    CorroborationState,
    EvidenceExpansion,
    ObserverOutput,
    ObserverRequest,
    StopReason,
    diagnose_window,
    reconcile_claims,
)
from actuate.schema.semantic import ClaimStatus, EvidenceRef, SemanticClaim, VLMRecord

ACTIVITY_ONLY = frozenset({"activity"})


def _evidence(start: float = 0, end: float = 1, frame: int = 0) -> EvidenceRef:
    return EvidenceRef(
        capture_id="c" * 64,
        source_content_hash="c" * 64,
        source_start_ms=round(start * 1_000),
        source_end_ms=round(end * 1_000),
        frame_indices=(frame,),
        preview_uri="artifact://preview",
    )


def _claim(
    claim_id: str,
    field: str,
    value: str,
    *,
    claim_type: str = "visible_fact",
    evidence: EvidenceRef | None = None,
    clarity: str = "clear",
) -> SemanticClaim:
    if claim_type == "visible_fact":
        claim = {
            "kind": claim_type,
            "claim_id": claim_id,
            "field": field,
            "value": value,
            "evidence": ({"frame_id": f"frame-{claim_id}", "timestamp_ms": 0},),
            "clarity": clarity,
        }
    elif claim_type == "inference":
        claim = {
            "kind": claim_type,
            "claim_id": claim_id,
            "field": field,
            "value": value,
            "based_on_claim_ids": ("visible-basis",),
            "reason": "supported by the visible action sequence",
            "alternatives": (),
        }
    else:
        claim = {
            "kind": "unknown",
            "claim_id": claim_id,
            "field": field,
            "reason": "insufficient_temporal_context",
        }
    return SemanticClaim(
        call_id=f"call-{claim_id}",
        claim=claim,
        evidence_refs=() if evidence is None else (evidence,),
        status=ClaimStatus.UNKNOWN if claim_type == "unknown" else ClaimStatus.MODEL_PROPOSED,
    )


def _output(claims, *review_flags) -> ObserverOutput:
    return ObserverOutput(
        record=VLMRecord(
            scope="window",
            source_record_id="window-1",
            claims=tuple(claim.claim for claim in claims),
            review_flags=frozenset(review_flags),
        ),
        claims=tuple(claims),
    )


class _Provider:
    def __init__(self, responses):
        self.responses = defaultdict(deque)
        for observer, values in responses.items():
            self.responses[observer].extend(values)
        self.calls: list[tuple[str, ObserverRequest]] = []

    def observe(self, observer_id, request):
        self.calls.append((observer_id, request))
        response = self.responses[observer_id].popleft()
        return response if isinstance(response, ObserverOutput) else _output(response)


class _EvidenceSupplier:
    def __init__(self, *expansions):
        self.expansions = deque(expansions)
        self.calls = []

    def expand(self, request):
        self.calls.append(request)
        return self.expansions.popleft()


def test_initial_observers_are_blind_and_independent():
    evidence = (_evidence(),)
    claims = (_claim("a", "activity", "walking", evidence=evidence[0]),)
    provider = _Provider({"observer_a": [claims], "observer_b": [claims]})
    supplier = _EvidenceSupplier(EvidenceExpansion(stop_reason=StopReason.HUMAN_RESOLVED))

    diagnose_window(evidence, provider, supplier, required_fields=ACTIVITY_ONLY)

    first, second = provider.calls
    assert first[0] == "observer_a" and second[0] == "observer_b"
    assert first[1] == second[1] == ObserverRequest(evidence)
    assert first[1].disputed_claims == ()


def test_boundary_difference_over_1000ms_requires_review():
    left = (_claim("a", "activity", "walking", evidence=_evidence(0, 2)),)
    at_threshold = (_claim("b", "activity", "WALKING", evidence=_evidence(1, 3)),)
    over_threshold = (_claim("c", "activity", " walking ", evidence=_evidence(1.001, 3.001)),)

    accepted = reconcile_claims(_output(left), _output(at_threshold))[0]
    flagged = reconcile_claims(_output(left), _output(over_threshold))[0]

    assert accepted.boundary_disagreement_ms == BOUNDARY_REVIEW_THRESHOLD_MS
    assert not accepted.needs_review
    assert flagged.boundary_disagreement_ms == BOUNDARY_REVIEW_THRESHOLD_MS + 1
    assert flagged.needs_review
    assert all(
        claim.corroboration == CorroborationState.REVIEW_REQUIRED for claim in flagged.claims
    )


def test_matching_inferences_remain_model_proposed():
    left = (_claim("a", "task", "weld seam", claim_type="inference"),)
    right = (_claim("b", "task", "Weld  seam", claim_type="inference"),)

    field = reconcile_claims(_output(left), _output(right))[0]

    assert field.needs_review
    assert all(claim.claim.status == ClaimStatus.MODEL_PROPOSED for claim in field.claims)
    assert all(claim.corroboration == CorroborationState.AGREED_MODELS for claim in field.claims)


def test_partial_visible_fact_requires_review():
    evidence = _evidence()
    left = (_claim("a", "activity", "walking", evidence=evidence, clarity="partial"),)
    right = (_claim("b", "activity", "walking", evidence=evidence),)

    assert reconcile_claims(_output(left), _output(right))[0].needs_review


def test_record_review_flag_is_preserved_and_triggers_review():
    evidence = _evidence()
    left = (_claim("a", "activity", "walking", evidence=evidence),)
    right = (_claim("b", "activity", "walking", evidence=evidence),)

    assert reconcile_claims(_output(left, "possible_pii"), _output(right))[0].needs_review


def test_no_model_call_without_new_evidence_and_two_no_progress_rounds_stop():
    evidence = (_evidence(),)
    left = (_claim("a", "activity", "walking", evidence=evidence[0]),)
    right = (_claim("b", "activity", "standing", evidence=evidence[0]),)
    provider = _Provider({"observer_a": [left], "observer_b": [right]})
    supplier = _EvidenceSupplier(
        EvidenceExpansion(evidence=evidence),
        EvidenceExpansion(evidence=evidence),
    )

    result = diagnose_window(evidence, provider, supplier, required_fields=ACTIVITY_ONLY)

    assert result.stop_reason == StopReason.NO_NEW_EVIDENCE
    assert len(result.rounds) == 2
    assert len(provider.calls) == 2


def test_new_evidence_can_resolve_a_dispute():
    initial = (_evidence(),)
    expanded = _evidence(1, 2, 30)
    left = (_claim("a", "activity", "walking", evidence=initial[0]),)
    right = (_claim("b", "activity", "standing", evidence=initial[0]),)
    resolved = (_claim("f", "activity", "walking", evidence=expanded),)
    provider = _Provider({"observer_a": [left], "observer_b": [right], "followup": [resolved]})
    supplier = _EvidenceSupplier(EvidenceExpansion(evidence=(expanded,)))

    result = diagnose_window(initial, provider, supplier, required_fields=ACTIVITY_ONLY)

    assert result.stop_reason == StopReason.RESOLVED
    assert result.unresolved_fields == frozenset()
    assert result.rounds[0].resolved_fields == ACTIVITY_ONLY
    followup_request = provider.calls[-1][1]
    assert followup_request.target_field == "activity"
    assert {claim.claim.claim_id for claim in followup_request.disputed_claims} == {"a", "b"}


def test_unknown_followup_abstains_instead_of_guessing():
    initial = (_evidence(),)
    expanded = _evidence(1, 2, 30)
    left = (_claim("a", "activity", "walking", evidence=initial[0]),)
    right = (_claim("b", "activity", "standing", evidence=initial[0]),)
    unknown = (_claim("u", "activity", "unknown", claim_type="unknown"),)
    provider = _Provider({"observer_a": [left], "observer_b": [right], "followup": [unknown]})
    supplier = _EvidenceSupplier(EvidenceExpansion(evidence=(expanded,)))

    result = diagnose_window(initial, provider, supplier, required_fields=ACTIVITY_ONLY)

    assert result.stop_reason == StopReason.ABSTAINED
    assert result.claims[-1].status == ClaimStatus.UNKNOWN


def test_initial_unknown_expands_evidence_before_abstaining():
    initial = (_evidence(),)
    expanded = _evidence(1, 2, 30)
    unknown_a = (_claim("a", "activity", "unknown", claim_type="unknown"),)
    unknown_b = (_claim("b", "activity", "unknown", claim_type="unknown"),)
    followup = (_claim("f", "activity", "unknown", claim_type="unknown"),)
    provider = _Provider(
        {"observer_a": [unknown_a], "observer_b": [unknown_b], "followup": [followup]}
    )
    supplier = _EvidenceSupplier(EvidenceExpansion(evidence=(expanded,)))

    result = diagnose_window(initial, provider, supplier, required_fields=ACTIVITY_ONLY)

    assert result.stop_reason == StopReason.ABSTAINED
    assert len(supplier.calls) == 1
    assert len(provider.calls) == 3


def test_explicit_human_resolution_stops_before_followup_model_call():
    evidence = (_evidence(),)
    left = (_claim("a", "activity", "walking", evidence=evidence[0]),)
    right = (_claim("b", "activity", "standing", evidence=evidence[0]),)
    provider = _Provider({"observer_a": [left], "observer_b": [right]})
    supplier = _EvidenceSupplier(EvidenceExpansion(stop_reason=StopReason.HUMAN_RESOLVED))

    result = diagnose_window(evidence, provider, supplier, required_fields=ACTIVITY_ONLY)

    assert result.stop_reason == StopReason.HUMAN_RESOLVED
    assert len(provider.calls) == 2
