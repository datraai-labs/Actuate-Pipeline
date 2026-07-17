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
#: v3 (2026-07-15) — MANO pose widened from 15-PCA to the full 45 axis-angle.
#:   * MANOParams.theta_pca (15)  ->  MANOParams.theta (45).
#:   Measured on the real capture with a correct least-squares projection, restricting to the
#:   top-15 PCA subspace loses a median 10.3 deg / p99 22.5 deg per joint -- material for
#:   retargeting (§L5), and carrying the full 45 is free and exactly lossless. The schema now
#:   carries the full 45; a consumer wanting the compressed form projects it down itself. (An
#:   earlier 31 deg figure came from a transpose-inverse bug on a non-orthonormal basis.)
#:   RETYPED field, so
#:   every exporter is re-verified against the load+train gate.
#:   * observation.state / action gain 45 MANO columns (cols 8-52) -- see canonical.build.
#: v4 (2026-07-17) — the certificate becomes real (Phase 5 Parts B+D, one bump not two).
#:   + EpisodeMeta.components: CertificateComponents (sync_integrity,
#:     calibration_completeness, perception_confidence, contact_consistency,
#:     ik_convergence_rate — each Unit|None, None = NOT MEASURED, never zero).
#:   + CanonicalEpisode.retarget_eligibility: dict[embodiment, bool] from L5 sim validation.
#:   + FieldStats.p02/p05/p25/p50/p75/p95/p98 (optional) — full raw-percentile set so
#:     customers on 2/98-per-timestep (TRI LBM) or z-score (EgoMimic) re-derive without
#:     recomputing over the raw dataset.
#:   All additive-optional: v3 payloads validate unchanged.
#: v5 (2026-07-17) — fine-grained action labeling (Master Spec v1 §10.3-10.4).
#:   + CanonicalEpisode.action_intervals: tuple[ActionInterval, ...] -- atomic actions
#:     from the CLOSED 20-verb vocabulary (ActionVerb), each attributed to an Actor,
#:     joinable to per-frame data by [start_frame, end_frame]. Overlapping across actors.
#:   + enums ActionVerb (20) and Actor (8) -- closed vocabularies.
#:   Additive-optional: v4 payloads validate unchanged.
SCHEMA_VERSION = 5

FROZEN_DIR = Path(__file__).parent / "frozen"


def frozen_schema_path(version: int = SCHEMA_VERSION) -> Path:
    return FROZEN_DIR / f"canonical_v{version}.schema.json"
