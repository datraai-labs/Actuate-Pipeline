"""Append-only JSONL artifacts for semantic model calls and claims."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from actuate.config import Bucket
from actuate.io.backends import StorageBackend, StorageError
from actuate.schema.semantic import ModelCall, SemanticClaim, SemanticRunManifest

Record = TypeVar("Record", bound=BaseModel)


class SemanticArtifactError(RuntimeError):
    pass


def semantic_prefix(capture_id: str, run_id: str) -> str:
    assert capture_id
    assert run_id
    assert "/" not in capture_id
    assert "/" not in run_id
    return f"raw_delivery/{capture_id}/semantic/{run_id}"


def _key(capture_id: str, run_id: str, name: str) -> str:
    return f"{semantic_prefix(capture_id, run_id)}/{name}.jsonl"


def _manifest_key(capture_id: str, run_id: str) -> str:
    return f"{semantic_prefix(capture_id, run_id)}/run_manifest.json"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in pairs:
        if key in row:
            raise ValueError(f"duplicate JSON key {key!r}")
        row[key] = value
    return row


def _encode(records: Iterable[Record]) -> bytes:
    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for record in records
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    return text.encode("utf-8")


def _write(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
    name: str,
    records: Iterable[Record],
) -> str:
    key = _key(capture_id, run_id, name)
    if backend.exists(Bucket.WORK, key):
        raise SemanticArtifactError(f"immutable artifact already exists: {key}")
    return backend.put_bytes(Bucket.WORK, key, _encode(records))


def _read(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
    name: str,
    record_type: type[Record],
    expected_count: int,
) -> tuple[Record, ...]:
    assert expected_count >= 0
    key = _key(capture_id, run_id, name)
    try:
        payload = backend.get_bytes(Bucket.WORK, key)
    except StorageError as exc:
        raise SemanticArtifactError(f"missing semantic artifact: {key}") from exc

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SemanticArtifactError(f"invalid UTF-8 in {key}") from exc
    if text and not text.endswith("\n"):
        raise SemanticArtifactError(f"truncated semantic artifact: {key}")

    rows: list[Record] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise SemanticArtifactError(f"blank row in {key} at line {line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            rows.append(record_type.model_validate(value))
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise SemanticArtifactError(
                f"invalid row in {key} at line {line_number}: {exc}"
            ) from exc

    if len(rows) != expected_count:
        raise SemanticArtifactError(
            f"row count mismatch in {key}: expected {expected_count}, found {len(rows)}"
        )
    return tuple(rows)


def write_calls(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
    calls: Iterable[ModelCall],
) -> str:
    rows = tuple(calls)
    if any(call.capture_id != capture_id or call.run_id != run_id for call in rows):
        raise SemanticArtifactError("model call identity does not match artifact path")
    return _write(backend, capture_id, run_id, "calls", rows)


def read_calls(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
    expected_count: int,
) -> tuple[ModelCall, ...]:
    rows = _read(backend, capture_id, run_id, "calls", ModelCall, expected_count)
    if any(call.capture_id != capture_id or call.run_id != run_id for call in rows):
        raise SemanticArtifactError("model call identity does not match artifact path")
    return rows


def write_claims(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
    claims: Iterable[SemanticClaim],
) -> str:
    rows = tuple(claims)
    if any(evidence.capture_id != capture_id for claim in rows for evidence in claim.evidence_refs):
        raise SemanticArtifactError("claim evidence identity does not match artifact path")
    return _write(backend, capture_id, run_id, "claims", rows)


def read_claims(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
    expected_count: int,
) -> tuple[SemanticClaim, ...]:
    rows = _read(backend, capture_id, run_id, "claims", SemanticClaim, expected_count)
    if any(evidence.capture_id != capture_id for claim in rows for evidence in claim.evidence_refs):
        raise SemanticArtifactError("claim evidence identity does not match artifact path")
    return rows


def write_manifest(backend: StorageBackend, manifest: SemanticRunManifest) -> str:
    key = _manifest_key(manifest.capture_id, manifest.run_id)
    if backend.exists(Bucket.WORK, key):
        raise SemanticArtifactError(f"immutable artifact already exists: {key}")
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return backend.put_bytes(Bucket.WORK, key, payload)


def read_manifest(
    backend: StorageBackend,
    capture_id: str,
    run_id: str,
) -> SemanticRunManifest:
    key = _manifest_key(capture_id, run_id)
    try:
        payload = backend.get_bytes(Bucket.WORK, key)
    except StorageError as exc:
        raise SemanticArtifactError(f"missing semantic artifact: {key}") from exc
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        manifest = SemanticRunManifest.model_validate(value)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise SemanticArtifactError(f"invalid semantic manifest {key}: {exc}") from exc
    if manifest.capture_id != capture_id or manifest.run_id != run_id:
        raise SemanticArtifactError(f"semantic manifest identity mismatch: {key}")
    return manifest
