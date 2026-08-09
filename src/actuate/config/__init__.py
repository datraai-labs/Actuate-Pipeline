"""Config, rig registry, embodiment registry (Master Spec §2.3).

Bottom of the import stack: every other package may import this; it imports nothing
from Actuate. Enforced by the import-linter contract in `.importlinter`.
"""

from __future__ import annotations

from actuate.config.capabilities import product_capabilities
from actuate.config.embodiments import (
    CANONICAL_REFERENCE_HAND,
    EMBODIMENT_REGISTRY,
    EmbodimentSpec,
    HandSpec,
    get_embodiment,
    retarget_ready_embodiments,
)
from actuate.config.enums import (
    MEASURED_PROVENANCES,
    PROVENANCE_TRUST_ORDER,
    ActionVerb,
    Actor,
    Channel,
    ConsentStatus,
    ControlMode,
    EgoMotionMethod,
    Finger,
    FingerActionRepr,
    InteractionState,
    PiiStatus,
    Provenance,
    RigType,
    Side,
    Tier,
    trust_rank,
)
from actuate.config.rigs import (
    RIG_REGISTRY,
    RigSpec,
    SensorStreamSpec,
    all_sensor_patterns,
    get_rig,
)
from actuate.config.settings import (
    Bucket,
    Env,
    Settings,
    StorageBackendKind,
    load_settings,
)

__all__ = [
    "CANONICAL_REFERENCE_HAND",
    "EMBODIMENT_REGISTRY",
    "MEASURED_PROVENANCES",
    "PROVENANCE_TRUST_ORDER",
    "RIG_REGISTRY",
    "ActionVerb",
    "Actor",
    "Bucket",
    "Channel",
    "ConsentStatus",
    "ControlMode",
    "EgoMotionMethod",
    "EmbodimentSpec",
    "Env",
    "Finger",
    "FingerActionRepr",
    "HandSpec",
    "InteractionState",
    "PiiStatus",
    "Provenance",
    "RigSpec",
    "RigType",
    "SensorStreamSpec",
    "Settings",
    "Side",
    "StorageBackendKind",
    "Tier",
    "all_sensor_patterns",
    "get_embodiment",
    "get_rig",
    "load_settings",
    "product_capabilities",
    "retarget_ready_embodiments",
    "trust_rank",
]

from actuate.config import auth  # noqa: F401  (user auth + defaults)
