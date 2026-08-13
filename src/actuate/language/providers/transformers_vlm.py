"""Schema-constrained local Transformers VLM provider."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

from pydantic import ValidationError

from actuate.language.semantic import ObserverOutput, ObserverRequest
from actuate.language.semantic_prompts import PROMPTS, render_user_prompt
from actuate.schema.semantic import (
    ClaimStatus,
    EvidenceRef,
    Inference,
    RecordScope,
    SemanticClaim,
    UnknownClaim,
    VisibleFact,
    VLMRecord,
    vlm_record_json_schema,
    vlm_record_schema_hash,
)

if TYPE_CHECKING:
    from PIL.Image import Image


DECODING_BACKEND = "lm-format-enforcer"
MAX_NEW_TOKENS = 1024
WINDOW_FIELDS = frozenset(
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
FORBIDDEN_AUTHORITY_PHRASES = (
    "consent granted",
    "consent denied",
    "rights cleared",
    "ownership verified",
    "pii safe",
    "safe for delivery",
    "camera imu synchronized",
    "camera-imu synchronized",
    "calibration valid",
    "sensor units verified",
    "sensor axes verified",
    "force measured",
    "contact measured",
    "accept this dataset",
    "reject this dataset",
    "delete this dataset",
    "quarantine this dataset",
    "commercial price",
)


class SemanticOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str


@dataclass(frozen=True)
class GeneratedResponse:
    raw_response: str
    source_record_id: str
    call_id: str


PINNED_MODELS = {
    "observer_a": ModelSpec(
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    ),
    "observer_b": ModelSpec(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "482adb537c021c86670beed01cd58990d01e72e4",
    ),
}


def _frame_map(request: ObserverRequest) -> tuple[dict[str, int | str], ...]:
    assert request.evidence
    assert all(len(item.frame_indices) == 1 for item in request.evidence)
    return tuple(
        {
            "frame_id": f"frame-{position}-{item.frame_indices[0]}",
            "timestamp_ms": item.source_start_ms,
        }
        for position, item in enumerate(request.evidence)
    )


def _request_id(request: ObserverRequest) -> str:
    payload = {
        "evidence": [item.model_dump(mode="json") for item in request.evidence],
        "target_field": request.target_field,
        "disputed_claims": [item.model_dump(mode="json") for item in request.disputed_claims],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_observer_prompt(request: ObserverRequest) -> tuple[str, str, str]:
    frame_map_json = json.dumps(_frame_map(request), separators=(",", ":"))
    source_record_id = f"window-{_request_id(request)[:16]}"
    if request.target_field is None:
        user = render_user_prompt(
            "window_observer",
            window_id=source_record_id,
            source_start_ms=min(item.source_start_ms for item in request.evidence),
            source_end_ms=max(item.source_end_ms for item in request.evidence),
            frame_timestamp_map_json=frame_map_json,
            declared_metadata_json="{}",
        )
        return PROMPTS["window_observer"][0], user, source_record_id

    user = render_user_prompt(
        "targeted_followup",
        target_field=request.target_field,
        target_question=f"What does the supplied evidence support for {request.target_field}?",
        window_id=source_record_id,
        frame_timestamp_map_json=frame_map_json,
        disputed_claims_json=json.dumps(
            [item.claim.model_dump(mode="json") for item in request.disputed_claims],
            separators=(",", ":"),
        ),
    )
    return PROMPTS["targeted_followup"][0], user, source_record_id


def parse_observer_output(
    raw_response: str,
    request: ObserverRequest,
    source_record_id: str,
    call_id: str,
) -> ObserverOutput:
    try:
        record = VLMRecord.model_validate_json(raw_response)
    except ValidationError as error:
        raise SemanticOutputError(str(error)) from error
    if record.scope is not RecordScope.WINDOW or record.source_record_id != source_record_id:
        raise SemanticOutputError("model output identity does not match the request")

    claim_ids = [claim.claim_id for claim in record.claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise SemanticOutputError("claim IDs must be unique")

    fields = {claim.field.value for claim in record.claims}
    expected_fields = WINDOW_FIELDS if request.target_field is None else {request.target_field}
    if not expected_fields.issubset(fields):
        raise SemanticOutputError("model output omitted required fields")
    if request.target_field is not None and fields != expected_fields:
        raise SemanticOutputError("follow-up output contains unrelated fields")

    frame_map = _frame_map(request)
    evidence_by_point = {
        (frame["frame_id"], frame["timestamp_ms"]): evidence
        for frame, evidence in zip(frame_map, request.evidence, strict=True)
    }
    visible_ids = {claim.claim_id for claim in record.claims if isinstance(claim, VisibleFact)}
    output_claims = []
    for claim in record.claims:
        text = " ".join(
            (
                getattr(claim, "value", ""),
                getattr(claim, "reason", ""),
                *getattr(claim, "alternatives", ()),
            )
        ).casefold()
        if any(phrase in text for phrase in FORBIDDEN_AUTHORITY_PHRASES):
            raise SemanticOutputError("model output claims forbidden authority")

        evidence_refs = ()
        if isinstance(claim, VisibleFact):
            refs = []
            for point in claim.evidence:
                evidence = evidence_by_point.get((point.frame_id, point.timestamp_ms))
                if evidence is None:
                    raise SemanticOutputError("visible fact cites unknown evidence")
                refs.append(evidence.model_copy(update={"bbox_xyxy": point.region}))
            evidence_refs = tuple(refs)
        elif isinstance(claim, Inference):
            if not set(claim.based_on_claim_ids).issubset(visible_ids):
                raise SemanticOutputError("inference cites a non-visible or unknown claim")

        output_claims.append(
            SemanticClaim(
                call_id=call_id,
                claim=claim,
                evidence_refs=evidence_refs,
                status=(
                    ClaimStatus.UNKNOWN
                    if isinstance(claim, UnknownClaim)
                    else ClaimStatus.MODEL_PROPOSED
                ),
            )
        )
    return ObserverOutput(record, tuple(output_claims))


class TransformersVLMProvider:
    def __init__(
        self,
        models: Mapping[str, ModelSpec],
        image_loader: Callable[[EvidenceRef], Image],
    ) -> None:
        assert models
        self.models = dict(models)
        self.image_loader = image_loader

    @property
    def decoding_backend_version(self) -> str:
        return version("lm-format-enforcer")

    @property
    def output_schema_hash(self) -> str:
        return vlm_record_schema_hash()

    def generate(self, observer_id: str, request: ObserverRequest) -> GeneratedResponse:
        import torch
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        from transformers import AutoModelForImageTextToText, AutoProcessor

        spec = self.models[observer_id]
        system, user, source_record_id = build_observer_prompt(request)
        images = [self.image_loader(item) for item in request.evidence]
        processor = AutoProcessor.from_pretrained(spec.model_id, revision=spec.revision)
        model = AutoModelForImageTextToText.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": user},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        parser = JsonSchemaParser(vlm_record_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            processor.tokenizer,
            parser,
        )
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    prefix_allowed_tokens_fn=prefix_function,
                )
            raw_response = processor.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            call_id = f"{observer_id}-{_request_id(request)[:16]}"
            return GeneratedResponse(raw_response, source_record_id, call_id)
        finally:
            del model, processor, inputs
            gc.collect()
            torch.cuda.empty_cache()

    def observe(self, observer_id: str, request: ObserverRequest) -> ObserverOutput:
        generated = self.generate(observer_id, request)
        return parse_observer_output(
            generated.raw_response,
            request,
            generated.source_record_id,
            generated.call_id,
        )
