"""L0 -- Ingestion & Sync.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L0

Built so far: **content-addressed capture provenance** (content_address.py). The capture id
is the SHA-256 of the raw bytes, so metadata cannot drift from the payload it describes and
a duplicate upload is a dedup hit rather than a conflict found later. This removes the
three integrity failure classes found in the real corpus in Increment 1.

Not built (later increments): MCAP container, PyAV decode, Polars sync, the six RigAdapters
(incl. DexUMI exoskeleton), aligned-capture mode, SLAM/IMU ego-motion.
"""

from __future__ import annotations

from actuate.ingest.content_address import (
    CaptureManifest,
    IntegrityError,
    build_manifest,
    check_legacy_claims,
    hash_file,
    read_manifest,
    write_manifest,
)

__all__ = [
    "CaptureManifest",
    "IntegrityError",
    "build_manifest",
    "check_legacy_claims",
    "hash_file",
    "read_manifest",
    "write_manifest",
]
