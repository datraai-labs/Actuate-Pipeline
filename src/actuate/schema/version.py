"""Canonical schema versioning — Master Spec §3.

`schema_version = 1`. The schema is FROZEN: `frozen/canonical_v1.schema.json` is the
generated JSON Schema, checked in, and `tests/unit/test_schema_frozen.py` fails if the
models drift from it without a version bump.

This is what makes "freeze and version it before building L4–L7" a mechanism rather than
a sentence. Exporters compile against a specific version of this contract; if the models
move underneath them, they break silently against data they believe they understand.

Bump rules:
  - Additive, optional field           -> new version, old readers still work.
  - Removed / renamed / retyped field  -> new version, and every exporter must be
                                          re-verified against the load+train gate.
  - New enum member a consumer branches on -> new version.
Nothing changes without a bump. That is the point.
"""

from __future__ import annotations

from pathlib import Path

#: v2 (2026-07-14) — content-addressed provenance.
#:   + CanonicalEpisode.source_content_hash: SHA-256 of the raw capture bytes.
#:   + capture_id is now defined AS that hash (actuate.ingest.content_address).
#: Bumped now rather than later on purpose: no exporter has been built against v1 yet, so
#: the change is free today and expensive the moment one is (Master Spec §3).
SCHEMA_VERSION = 2

FROZEN_DIR = Path(__file__).parent / "frozen"


def frozen_schema_path(version: int = SCHEMA_VERSION) -> Path:
    return FROZEN_DIR / f"canonical_v{version}.schema.json"
