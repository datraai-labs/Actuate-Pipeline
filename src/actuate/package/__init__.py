"""L7 -- Packaging & Delivery.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L7

LeRobot v3 exporter, written THROUGH LeRobot's own writer so schema correctness is
inherited rather than re-implemented. Verified by the load+train gate (load with LeRobot's
own loader, run one real training step), never by asserting schema-correctness.

Phase 5 Part D: tier filtering, dual-space (human + per-embodiment robot actions),
export-time co-training transforms, full-percentile norm stats, dataset manifest.

RLDS/Open-X secondary exporter (Part C): written through tfds's own ad-hoc
builder, gated by tfds.load + iterate.
"""

from __future__ import annotations

from actuate.package.lerobot_export import (
    ExportRefused,
    ExportResult,
    compute_norm_stats,
    denormalize_p01_p99,
    export_lerobot_v3,
    normalize_p01_p99,
)
from actuate.package.manifest import DatasetManifest
from actuate.package.manifest import generate as generate_manifest
from actuate.package.normalize import compute_field_stats, verify_round_trip
from actuate.package.rlds_export import RldsExportResult, export_rlds

__all__ = [
    "DatasetManifest",
    "ExportRefused",
    "ExportResult",
    "RldsExportResult",
    "compute_field_stats",
    "compute_norm_stats",
    "denormalize_p01_p99",
    "export_lerobot_v3",
    "export_rlds",
    "generate_manifest",
    "normalize_p01_p99",
    "verify_round_trip",
]
