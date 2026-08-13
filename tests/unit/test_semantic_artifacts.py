from datetime import datetime, timedelta, timezone

import pytest

from actuate.config import Bucket
from actuate.io.backends import LocalBackend
from actuate.language.semantic_artifacts import (
    SemanticArtifactError,
    read_calls,
    read_claims,
    read_manifest,
    write_calls,
    write_claims,
    write_manifest,
)
from actuate.schema.semantic import ModelCall, SemanticClaim, SemanticRunManifest

CAPTURE_ID = "a" * 64
RUN_ID = "run-1"
CREATED_AT = datetime(2026, 8, 14, 9, 8, 7, 654321, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def _visible_fact() -> dict:
    return {
        "kind": "visible_fact",
        "claim_id": "claim-1",
        "field": "activity",
        "value": "A person handles a tool.",
        "evidence": [{"frame_id": "frame-37", "timestamp_ms": 1234}],
        "clarity": "clear",
    }


def _call(run_id: str = RUN_ID) -> ModelCall:
    return ModelCall.model_validate(
        {
            "call_id": "call-1",
            "parent_call_id": None,
            "trigger": "initial_window",
            "run_id": run_id,
            "capture_id": CAPTURE_ID,
            "source_start_ms": 1234,
            "source_end_ms": 8125,
            "provider": "local",
            "model": "Qwen3-VL-4B-Instruct",
            "model_version": "sha-123",
            "prompt_id": "corpus-b-window-observer",
            "prompt_hash": "b" * 64,
            "output_schema_version": "v1",
            "output_schema_hash": "e" * 64,
            "decoding_backend": "lm-format-enforcer",
            "decoding_backend_version": "0.11.2",
            "decode_parameters": {"do_sample": False, "max_new_tokens": 1024},
            "request_hash": "c" * 64,
            "response_hash": "d" * 64,
            "raw_response_uri": "work://raw-responses/call-1.json",
            "input_tokens": 123,
            "output_tokens": 45,
            "video_seconds": 8.125,
            "cost_usd": 0.00123,
            "latency_ms": 987,
            "status": "succeeded",
            "created_at": CREATED_AT,
            "model_output": {
                "scope": "window",
                "source_record_id": "window-1",
                "claims": [_visible_fact()],
                "review_flags": [],
            },
        }
    )


def _claim() -> SemanticClaim:
    return SemanticClaim.model_validate(
        {
            "call_id": "call-1",
            "claim": _visible_fact(),
            "evidence_refs": [
                {
                    "capture_id": CAPTURE_ID,
                    "source_content_hash": CAPTURE_ID,
                    "source_start_ms": 123456789,
                    "source_end_ms": 812500000,
                    "frame_indices": [37, 244],
                    "preview_uri": "work://previews/claim-1.jpg",
                    "bbox_xyxy": [0.125, 0.25, 0.875, 0.9375],
                }
            ],
            "status": "model_proposed",
        }
    )


def _manifest() -> SemanticRunManifest:
    return SemanticRunManifest(
        run_id=RUN_ID,
        capture_id=CAPTURE_ID,
        source_uri="raw://captures/source.mp4",
        provider_models=("local/Qwen3-VL-4B-Instruct@sha-123",),
        prompt_revision="v1",
        output_schema_version="v1",
        trigger_policy_version="v1",
        artifact_uris=("work://calls.jsonl", "work://claims.jsonl"),
        input_window_count=1,
        output_claim_count=1,
        actual_cost_usd=0.00123,
        stop_reason="resolved",
        unresolved_contradiction_count=0,
        created_at=CREATED_AT,
    )


def test_roundtrip_preserves_hashes_timestamps_and_raw_source(tmp_path):
    backend = LocalBackend(tmp_path)
    raw_key = f"captures/{CAPTURE_ID}/source.mp4"
    raw_bytes = b"immutable raw source"
    backend.put_bytes(Bucket.RAW, raw_key, raw_bytes)

    call = _call()
    claim = _claim()
    write_calls(backend, CAPTURE_ID, RUN_ID, [call])
    write_claims(backend, CAPTURE_ID, RUN_ID, [claim])

    loaded_call = read_calls(backend, CAPTURE_ID, RUN_ID, expected_count=1)[0]
    loaded_claim = read_claims(backend, CAPTURE_ID, RUN_ID, expected_count=1)[0]
    assert loaded_call == call
    assert loaded_call.created_at.isoformat() == CREATED_AT.isoformat()
    assert loaded_call.request_hash == "c" * 64
    assert loaded_claim == claim
    assert loaded_claim.evidence_refs[0].source_start_ms == 123456789
    assert loaded_claim.evidence_refs[0].source_content_hash == CAPTURE_ID
    assert backend.get_bytes(Bucket.RAW, raw_key) == raw_bytes


def test_writes_are_deterministic_and_never_overwritten(tmp_path):
    backend = LocalBackend(tmp_path)
    write_calls(backend, CAPTURE_ID, "run-a", [_call("run-a")])
    write_calls(backend, CAPTURE_ID, "run-b", [_call("run-b")])
    key_a = f"raw_delivery/{CAPTURE_ID}/semantic/run-a/calls.jsonl"
    key_b = f"raw_delivery/{CAPTURE_ID}/semantic/run-b/calls.jsonl"
    first = backend.get_bytes(Bucket.WORK, key_a)
    second = backend.get_bytes(Bucket.WORK, key_b)
    assert first.replace(b"run-a", b"run-b") == second

    with pytest.raises(SemanticArtifactError, match="immutable artifact already exists"):
        write_calls(backend, CAPTURE_ID, "run-a", [_call("run-a")])
    assert backend.get_bytes(Bucket.WORK, key_a) == first


def test_empty_artifact_is_valid_only_when_zero_rows_are_expected(tmp_path):
    backend = LocalBackend(tmp_path)
    write_claims(backend, CAPTURE_ID, RUN_ID, [])
    assert read_claims(backend, CAPTURE_ID, RUN_ID, expected_count=0) == ()
    with pytest.raises(SemanticArtifactError, match="expected 1, found 0"):
        read_claims(backend, CAPTURE_ID, RUN_ID, expected_count=1)


def test_missing_artifact_fails_loudly(tmp_path):
    backend = LocalBackend(tmp_path)
    with pytest.raises(SemanticArtifactError, match="missing semantic artifact"):
        read_calls(backend, CAPTURE_ID, RUN_ID, expected_count=1)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"call_id":"a"', "truncated semantic artifact"),
        (b"not-json\n", "invalid row"),
        (b'{"call_id":"a","call_id":"b"}\n', "duplicate JSON key"),
        (b"\n", "blank row"),
    ],
)
def test_corrupted_artifact_fails_loudly(tmp_path, payload, message):
    backend = LocalBackend(tmp_path)
    key = f"raw_delivery/{CAPTURE_ID}/semantic/{RUN_ID}/calls.jsonl"
    backend.put_bytes(Bucket.WORK, key, payload)
    with pytest.raises(SemanticArtifactError, match=message):
        read_calls(backend, CAPTURE_ID, RUN_ID, expected_count=1)


def test_missing_row_fails_loudly(tmp_path):
    backend = LocalBackend(tmp_path)
    write_calls(backend, CAPTURE_ID, RUN_ID, [_call()])
    with pytest.raises(SemanticArtifactError, match="expected 2, found 1"):
        read_calls(backend, CAPTURE_ID, RUN_ID, expected_count=2)


def test_call_identity_must_match_artifact_path(tmp_path):
    backend = LocalBackend(tmp_path)
    with pytest.raises(SemanticArtifactError, match="identity does not match"):
        write_calls(backend, CAPTURE_ID, "wrong-run", [_call()])


def test_claim_evidence_identity_must_match_artifact_path(tmp_path):
    backend = LocalBackend(tmp_path)
    value = _claim().model_dump(mode="json")
    value["evidence_refs"][0]["capture_id"] = "b" * 64
    value["evidence_refs"][0]["source_content_hash"] = "b" * 64
    with pytest.raises(SemanticArtifactError, match="identity does not match"):
        write_claims(
            backend,
            CAPTURE_ID,
            RUN_ID,
            [SemanticClaim.model_validate(value)],
        )


def test_manifest_roundtrip_is_exact_and_immutable(tmp_path):
    backend = LocalBackend(tmp_path)
    manifest = _manifest()
    write_manifest(backend, manifest)
    assert read_manifest(backend, CAPTURE_ID, RUN_ID) == manifest
    with pytest.raises(SemanticArtifactError, match="immutable artifact already exists"):
        write_manifest(backend, manifest)


def test_manifest_identity_mismatch_fails_loudly(tmp_path):
    backend = LocalBackend(tmp_path)
    manifest = _manifest()
    key = f"raw_delivery/{CAPTURE_ID}/semantic/wrong-run/run_manifest.json"
    backend.put_bytes(Bucket.WORK, key, manifest.model_dump_json().encode("utf-8"))
    with pytest.raises(SemanticArtifactError, match="identity mismatch"):
        read_manifest(backend, CAPTURE_ID, "wrong-run")


def test_corrupted_manifest_fails_loudly(tmp_path):
    backend = LocalBackend(tmp_path)
    key = f"raw_delivery/{CAPTURE_ID}/semantic/{RUN_ID}/run_manifest.json"
    backend.put_bytes(Bucket.WORK, key, b'{"run_id":"duplicate","run_id":"values"}')
    with pytest.raises(SemanticArtifactError, match="invalid semantic manifest"):
        read_manifest(backend, CAPTURE_ID, RUN_ID)
