"""L1 -- hand pose. WiLoR -> MANO (Master Spec §L1).

LICENCE: WiLoR's published models are CC-BY-NC-ND-4.0 and MANO's standard grant is
non-commercial. INTERNAL RESEARCH ONLY unless separate commercial rights have been signed.
HaMeR's code is MIT, but its MANO model dependency remains separately restricted.
See docs/COMMERCIAL_LICENSE_READINESS.md.
"""

from __future__ import annotations

from actuate.perception.hands.wilor import (
    HandFrame,
    HandResult,
    WiLoREstimator,
    estimator_for,
    from_theta_pca,
    mediapipe_hand_presence,
    run,
    to_theta_pca,
)

__all__ = [
    "HandFrame",
    "HandResult",
    "WiLoREstimator",
    "estimator_for",
    "from_theta_pca",
    "mediapipe_hand_presence",
    "run",
    "to_theta_pca",
]
