"""L2 -- Sensor-Fused Interaction Refinement. Master Spec §L2.

Trust-weighted arbiter:
  measured_robotspace > measured_human > gripper_aperture > vision_primary > vision_fallback
plus a Schmitt-gated interaction-state machine. `fusion.run(hands, rig=..., objects=...)` fuses
per-frame perception into temporally-coherent interaction states and arbitrated,
provenance-tagged per-finger contact. On a bare-hand rig every channel resolves to
vision_fallback; a glove/DexUMI's measured readings override vision through the same arbiter.
The broken-priority test (tests/unit/test_fusion_arbiter.py) fails if the ordering is inverted.
"""

from __future__ import annotations

from actuate.fusion.arbiter import (
    TRUST_RANK,
    Candidate,
    arbitrate,
    resolve_channel,
)
from actuate.fusion.fusion import (
    ContactPoint,
    FrameFusion,
    FusionReport,
    HardwareSources,
    run,
)
from actuate.fusion.states import (
    SchmittTrigger,
    StateClassifier,
    enforce_min_dwell,
    finger_curl,
    grasp_signal,
)

__all__ = [
    "TRUST_RANK",
    "Candidate",
    "arbitrate",
    "resolve_channel",
    "ContactPoint",
    "FrameFusion",
    "FusionReport",
    "HardwareSources",
    "run",
    "SchmittTrigger",
    "StateClassifier",
    "enforce_min_dwell",
    "finger_curl",
    "grasp_signal",
]
