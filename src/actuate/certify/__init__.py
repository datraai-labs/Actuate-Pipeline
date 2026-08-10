"""L4 -- Certification: the quality certificate, computed from real sources.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §4 L4.

`certify.score(episode, embodiment=None, ...)` computes every certificate component
(sync_integrity, calibration_completeness, perception_confidence, contact_consistency,
ik_convergence_rate), the 1-5 quality composite, binned speed, per-segment mistake flags,
and -- when L5 results are passed in -- strategy_alignment and retarget_eligibility.

The fail-closed consent/PII gate is ALWAYS-ON and lives in actuate.io.consent. Quality never
substitutes for consent.

Not built: the LLM-as-judge caption gate (that is L6, actuate.language).
"""

from __future__ import annotations

from actuate.certify.score import (
    QUALITY_WEIGHTS,
    CertificationReport,
    calibration_completeness,
    composite_quality,
    contact_consistency,
    find_mistakes,
    perception_confidence,
    score,
    speed_bin,
    sync_integrity,
)

__all__ = [
    "QUALITY_WEIGHTS",
    "CertificationReport",
    "calibration_completeness",
    "composite_quality",
    "contact_consistency",
    "find_mistakes",
    "perception_confidence",
    "score",
    "speed_bin",
    "sync_integrity",
]
