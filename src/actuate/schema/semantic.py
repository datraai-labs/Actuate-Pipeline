"""Independent semantic evidence artifacts for raw-delivery review."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

SEMANTIC_SCHEMA_VERSION = "corpus-b-vlm-record-v1"


class RecordScope(str, Enum):
    WINDOW = "window"
    EPISODE = "episode"
    CORPUS = "corpus"


class VisibleField(str, Enum):
    ACTIVITY = "activity"
    HAND_VISIBILITY = "hand_visibility"
    VISIBLE_HAND_COUNT = "visible_hand_count"
    OBJECT = "object"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    VISUAL_QUALITY = "visual_quality"
    POSSIBLE_PII = "possible_pii"
    FAILURE_OR_RECOVERY = "failure_or_recovery"


class InferenceField(str, Enum):
    ACTIVITY = "activity"
    OBJECT = "object"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    TASK = "task"
    SUBTASK = "subtask"
    COMPLETION_STATE = "completion_state"
    FAILURE_OR_RECOVERY = "failure_or_recovery"
    CORPUS_RELATION = "corpus_relation"


class UnknownField(str, Enum):
    ACTIVITY = "activity"
    HAND_VISIBILITY = "hand_visibility"
    VISIBLE_HAND_COUNT = "visible_hand_count"
    OBJECT = "object"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    VISUAL_QUALITY = "visual_quality"
    POSSIBLE_PII = "possible_pii"
    TASK = "task"
    SUBTASK = "subtask"
    COMPLETION_STATE = "completion_state"
    FAILURE_OR_RECOVERY = "failure_or_recovery"
    CORPUS_RELATION = "corpus_relation"


class UnknownReason(str, Enum):
    NOT_VISIBLE = "not_visible"
    OCCLUDED = "occluded"
    BLURRED_OR_TOO_SMALL = "blurred_or_too_small"
    INSUFFICIENT_TEMPORAL_CONTEXT = "insufficient_temporal_context"
    MULTIPLE_PLAUSIBLE_VALUES = "multiple_plausible_values"
    MISSING_INTERVAL = "missing_interval"
    REQUIRES_SENSOR_OR_METADATA_EVIDENCE = "requires_sensor_or_metadata_evidence"
    UNSUPPORTED_CAUSAL_OR_INTENT_CLAIM = "unsupported_causal_or_intent_claim"


class ReviewFlag(str, Enum):
    INSUFFICIENT_VISUAL_EVIDENCE = "insufficient_visual_evidence"
    POSSIBLE_PII = "possible_pii"
    POSSIBLE_FAILURE_OR_RECOVERY = "possible_failure_or_recovery"
    BOUNDARY_UNCERTAIN = "boundary_uncertain"
    DECLARED_METADATA_CONFLICT = "declared_metadata_conflict"
    RARE_OR_UNMAPPED_LABEL = "rare_or_unmapped_label"
    UNSUPPORTED_UPSTREAM_CLAIM = "unsupported_upstream_claim"


class EvidencePoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    region: tuple[float, float, float, float] | None = None

    @field_validator("region")
    @classmethod
    def normalized_region(
        cls, region: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if region is None:
            return None
        x1, y1, x2, y2 = region
        if not all(0 <= value <= 1 for value in region) or x1 >= x2 or y1 >= y2:
            raise ValueError("region must be an ordered normalized xyxy box")
        return region


class VisibleFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["visible_fact"]
    claim_id: str = Field(min_length=1)
    field: VisibleField
    value: str = Field(min_length=1)
    evidence: tuple[EvidencePoint, ...] = Field(min_length=1)
    clarity: Literal["clear", "partial"]


class Inference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["inference"]
    claim_id: str = Field(min_length=1)
    field: InferenceField
    value: str = Field(min_length=1)
    based_on_claim_ids: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    alternatives: tuple[str, ...]


class UnknownClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["unknown"]
    claim_id: str = Field(min_length=1)
    field: UnknownField
    reason: UnknownReason


Claim = Annotated[VisibleFact | Inference | UnknownClaim, Field(discriminator="kind")]


class VLMRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: RecordScope
    source_record_id: str = Field(min_length=1)
    claims: tuple[Claim, ...]
    review_flags: frozenset[ReviewFlag]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(ge=0)
    frame_indices: tuple[int, ...] = Field(min_length=1)
    preview_uri: str = Field(min_length=1)
    bbox_xyxy: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def valid_evidence(self) -> EvidenceRef:
        if self.capture_id != self.source_content_hash:
            raise ValueError("source_content_hash must equal capture_id")
        if self.source_end_ms < self.source_start_ms:
            raise ValueError("source_end_ms must not precede source_start_ms")
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("frame indices must be non-negative")
        if self.bbox_xyxy is not None:
            x1, y1, x2, y2 = self.bbox_xyxy
            if not all(0 <= value <= 1 for value in self.bbox_xyxy) or x1 >= x2 or y1 >= y2:
                raise ValueError("bbox_xyxy must be an ordered normalized box")
        return self


class CallStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class ModelCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str = Field(min_length=1)
    parent_call_id: str | None
    trigger: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(ge=0)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_version: str = Field(min_length=1)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_response_uri: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    video_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    status: CallStatus
    created_at: datetime
    model_output: VLMRecord

    @model_validator(mode="after")
    def ordered_span(self) -> ModelCall:
        if self.source_end_ms < self.source_start_ms:
            raise ValueError("source_end_ms must not precede source_start_ms")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return self


class ClaimStatus(str, Enum):
    MODEL_PROPOSED = "model_proposed"
    UNKNOWN = "unknown"


class SemanticClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str = Field(min_length=1)
    claim: Claim
    evidence_refs: tuple[EvidenceRef, ...]
    status: ClaimStatus

    @model_validator(mode="after")
    def model_status(self) -> SemanticClaim:
        if isinstance(self.claim, UnknownClaim):
            if self.status is not ClaimStatus.UNKNOWN:
                raise ValueError("unknown model claims must remain unknown")
        elif self.status is not ClaimStatus.MODEL_PROPOSED:
            raise ValueError("model claims cannot create a human-reviewed status")
        if isinstance(self.claim, VisibleFact) and not self.evidence_refs:
            raise ValueError("visible facts require source evidence")
        return self


class ReviewDecision(str, Enum):
    HUMAN_VERIFIED = "human_verified"
    HUMAN_CORRECTED = "human_corrected"
    MODEL_PROPOSED = "model_proposed"
    UNKNOWN = "unknown"
    HUMAN_REJECTED = "human_rejected"
    SPECIALIST_ESCALATION = "specialist_escalation"


class ClaimReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    decision: ReviewDecision
    reviewer_id: str = Field(min_length=1)
    reviewed_at: datetime
    corrected_claim: Claim | None = None

    @model_validator(mode="after")
    def valid_review(self) -> ClaimReview:
        if self.reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        if (self.decision is ReviewDecision.HUMAN_CORRECTED) != (self.corrected_claim is not None):
            raise ValueError("corrected_claim is required only for human_corrected")
        return self


class SemanticRunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    capture_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_uri: str = Field(min_length=1)
    provider_models: tuple[str, ...] = Field(min_length=1)
    prompt_revision: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    trigger_policy_version: str = Field(min_length=1)
    artifact_uris: tuple[str, ...]
    input_window_count: int = Field(ge=0)
    output_claim_count: int = Field(ge=0)
    actual_cost_usd: float = Field(ge=0)
    stop_reason: str = Field(min_length=1)
    unresolved_contradiction_count: int = Field(ge=0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return created_at


class SemanticClaimAdapter(RootModel[Claim]):
    """Validate a standalone model claim with the approved discriminator."""


def vlm_record_json_schema() -> dict:
    schema = VLMRecord.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = SEMANTIC_SCHEMA_VERSION
    return schema
