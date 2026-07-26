"""L0 -- Ingestion & Sync.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L0

Built so far: **content-addressed capture provenance** (content_address.py). The capture id
is the SHA-256 of the raw bytes, so metadata cannot drift from the payload it describes and
a duplicate upload is a dedup hit rather than a conflict found later. This removes the
three integrity failure classes found in the real corpus in Increment 1.

Phase 5 Part E adds a MINIMAL `run` for the already-processed session layout, with
aligned-capture verification (stage2_anchor is EARNED by matching calibrated intrinsics,
never defaulted; unverifiable claims FLAG).

Raw JSON/CSV IMU sidecars are synchronized to the video frame axis in ``session.h5`` for
the modern VIO/SLAM path. Still not built: MCAP containers, the six RigAdapters (including
DexUMI exoskeleton), or a general multi-clock/Polars synchronization layer.
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
from actuate.ingest.imu import IMUSyncResult, sync_imu
from actuate.ingest.run import IngestResult
from actuate.ingest.run import run as run_ingest

__all__ = [
    "CaptureManifest",
    "IMUSyncResult",
    "IngestResult",
    "IntegrityError",
    "build_manifest",
    "check_legacy_claims",
    "hash_file",
    "read_manifest",
    "run_ingest",
    "sync_imu",
    "write_manifest",
]
