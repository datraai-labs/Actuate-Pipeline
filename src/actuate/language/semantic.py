"""Provider-neutral, evidence-driven semantic diagnosis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from actuate.schema.semantic import EvidenceRef, SemanticClaim, VLMRecord

BOUNDARY_REVIEW_THRESHOLD_MS = 1_000
WINDOW_REQUIRED_FIELDS = frozenset(
    (
        "activity",
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
INFERENCE_REVIEW_FIELDS = frozenset(("task", "completion_state", "failure_or_recovery"))


class CorroborationState(str, Enum):
    SINGLE_OBSERVER = "single_observer"
    AGREED_MODELS = "agreed_models"
    REVIEW_REQUIRED = "review_required"
    FOLLOWUP_RESOLVED = "followup_resolved"


class StopReason(str, Enum):
    RESOLVED = "resolved"
    ABSTAINED = "abstained"
    NO_NEW_EVIDENCE = "no_new_evidence"
    HUMAN_RESOLVED = "human_resolved"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class ObserverRequest:
    evidence: tuple[EvidenceRef, ...]
    target_field: str | None = None
    disputed_claims: tuple[SemanticClaim, ...] = ()


@dataclass(frozen=True)
class ObserverOutput:
    record: VLMRecord
    claims: tuple[SemanticClaim, ...]


class SemanticProvider(Protocol):
    def observe(self, observer_id: str, request: ObserverRequest) -> ObserverOutput: ...


@dataclass(frozen=True)
class EvidenceRequest:
    target_fields: frozenset[str]
    seen_evidence: tuple[EvidenceRef, ...]
    round_index: int


@dataclass(frozen=True)
class EvidenceExpansion:
    evidence: tuple[EvidenceRef, ...] = ()
    stop_reason: StopReason | None = None


class EvidenceSupplier(Protocol):
    def expand(self, request: EvidenceRequest) -> EvidenceExpansion: ...


@dataclass(frozen=True)
class ReconciledClaim:
    claim: SemanticClaim
    corroboration: CorroborationState
    contradiction_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconciledField:
    field: str
    claims: tuple[ReconciledClaim, ...]
    needs_review: bool
    boundary_disagreement_ms: int = 0


@dataclass(frozen=True)
class DiagnosisRound:
    round_index: int
    requested_fields: frozenset[str]
    new_evidence: tuple[EvidenceRef, ...]
    claims: tuple[SemanticClaim, ...]
    resolved_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DiagnosisResult:
    claims: tuple[SemanticClaim, ...]
    rounds: tuple[DiagnosisRound, ...]
    unresolved_fields: frozenset[str]
    stop_reason: StopReason


def _normalized(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _evidence_key(evidence: EvidenceRef) -> str:
    return evidence.model_dump_json(exclude_none=True)


def _field(claim: SemanticClaim) -> str:
    return claim.claim.field.value


def _boundary_disagreement_ms(left: SemanticClaim, right: SemanticClaim) -> int:
    if not left.evidence_refs or not right.evidence_refs:
        return 0
    left_start = min(item.source_start_ms for item in left.evidence_refs)
    right_start = min(item.source_start_ms for item in right.evidence_refs)
    left_end = max(item.source_end_ms for item in left.evidence_refs)
    right_end = max(item.source_end_ms for item in right.evidence_refs)
    return max(abs(left_start - right_start), abs(left_end - right_end))


def reconcile_claims(
    observer_a: ObserverOutput, observer_b: ObserverOutput
) -> tuple[ReconciledField, ...]:
    """Reconcile blind outputs without a model vote or confidence averaging."""
    fields = sorted({_field(claim) for claim in (*observer_a.claims, *observer_b.claims)})
    flagged = bool(observer_a.record.review_flags or observer_b.record.review_flags)
    reconciled = []
    for field in fields:
        left = [claim for claim in observer_a.claims if _field(claim) == field]
        right = [claim for claim in observer_b.claims if _field(claim) == field]
        claims = left + right
        paired = len(left) == 1 and len(right) == 1
        same = paired and (
            left[0].claim.kind == right[0].claim.kind
            and _normalized(getattr(left[0].claim, "value", None))
            == _normalized(getattr(right[0].claim, "value", None))
        )
        boundary_ms = _boundary_disagreement_ms(left[0], right[0]) if same else 0
        boundary_review = boundary_ms > BOUNDARY_REVIEW_THRESHOLD_MS

        if same and not boundary_review:
            annotated = tuple(
                ReconciledClaim(claim, CorroborationState.AGREED_MODELS) for claim in claims
            )
            needs_review = (
                flagged
                or left[0].claim.kind == "inference"
                and field in INFERENCE_REVIEW_FIELDS
                or left[0].claim.kind == "visible_fact"
                and left[0].claim.clarity == "partial"
                or right[0].claim.kind == "visible_fact"
                and right[0].claim.clarity == "partial"
            )
        else:
            annotated = tuple(
                ReconciledClaim(
                    claim,
                    CorroborationState.REVIEW_REQUIRED,
                    tuple(
                        other.claim.claim_id
                        for other in claims
                        if other.claim.claim_id != claim.claim.claim_id
                    ),
                )
                for claim in claims
            )
            needs_review = True

        reconciled.append(
            ReconciledField(
                field=field,
                claims=annotated,
                needs_review=needs_review,
                boundary_disagreement_ms=boundary_ms,
            )
        )
    return tuple(reconciled)


def _targets(
    fields: tuple[ReconciledField, ...], required_fields: frozenset[str]
) -> frozenset[str]:
    present = {field.field for field in fields}
    targets = set(required_fields - present)
    for field in fields:
        if field.needs_review or all(claim.claim.claim.kind == "unknown" for claim in field.claims):
            targets.add(field.field)
    return frozenset(targets)


def diagnose_window(
    initial_evidence: tuple[EvidenceRef, ...],
    provider: SemanticProvider,
    evidence_supplier: EvidenceSupplier,
    *,
    required_fields: frozenset[str] = WINDOW_REQUIRED_FIELDS,
) -> DiagnosisResult:
    """Run two blind observers, then request targeted evidence until a stop condition."""
    blind_request = ObserverRequest(evidence=initial_evidence)
    observer_a = provider.observe("observer_a", blind_request)
    observer_b = provider.observe("observer_b", blind_request)
    fields = reconcile_claims(observer_a, observer_b)
    claims = tuple(item.claim for field in fields for item in field.claims)
    targets = _targets(fields, required_fields)
    if not targets:
        return DiagnosisResult(claims, (), targets, StopReason.RESOLVED)
    seen = {_evidence_key(item) for item in initial_evidence}
    evidence = list(initial_evidence)
    rounds = []
    no_progress = 0
    round_index = 1
    while True:
        expansion = evidence_supplier.expand(EvidenceRequest(targets, tuple(evidence), round_index))
        if expansion.stop_reason is not None:
            return DiagnosisResult(claims, tuple(rounds), targets, expansion.stop_reason)

        novel = tuple(item for item in expansion.evidence if _evidence_key(item) not in seen)
        if not novel:
            no_progress += 1
            rounds.append(DiagnosisRound(round_index, targets, (), ()))
            if no_progress == 2:
                return DiagnosisResult(claims, tuple(rounds), targets, StopReason.NO_NEW_EVIDENCE)
            round_index += 1
            continue

        no_progress = 0
        evidence.extend(novel)
        seen.update(_evidence_key(item) for item in novel)
        followup_claims = []
        next_targets = set()
        for field in sorted(targets):
            disputed = tuple(claim for claim in claims if _field(claim) == field)
            output = provider.observe(
                "followup",
                ObserverRequest(tuple(evidence), field, disputed),
            )
            matching = tuple(claim for claim in output.claims if _field(claim) == field)
            followup_claims.extend(matching)
            if (
                output.record.review_flags
                or len(matching) != 1
                or matching[0].claim.kind == "inference"
            ):
                next_targets.add(field)

        annotated_followups = tuple(followup_claims)
        claims = (*claims, *annotated_followups)
        resolved_fields = targets - frozenset(next_targets)
        rounds.append(
            DiagnosisRound(round_index, targets, novel, annotated_followups, resolved_fields)
        )
        targets = frozenset(next_targets)
        if not targets:
            reason = (
                StopReason.ABSTAINED
                if annotated_followups
                and all(claim.claim.kind == "unknown" for claim in annotated_followups)
                else StopReason.RESOLVED
            )
            return DiagnosisResult(claims, tuple(rounds), targets, reason)
        round_index += 1
