"""The frozen canonical schema — Master Spec §3. THE FREEZE POINT.

Everything downstream of L3 (certify, retarget, language, package, the service) compiles
against this and nothing else. Import from here, not from the submodules.

The current frozen version is exported as ``SCHEMA_VERSION``. See version.py for the
freeze mechanism; never duplicate a numeric version in documentation.
"""

from __future__ import annotations

import json

from actuate.schema.episode import (
    ActionInterval,
    CanonicalEpisode,
    Diversity,
    EpisodeMeta,
    FieldStats,
    HumanAction,
    NormStats,
    ReferenceHandAction,
    RobotAction,
    StrategyAlignment,
    SubgoalFrame,
    Subtask,
)
from actuate.schema.frame import (
    PROVENANCE_REQUIRED_FIELDS,
    SE3,
    CanonicalFrame,
    ContactReading,
    DepthRef,
    HandState,
    ImageRef,
    MANOParams,
    MaskRef,
    ObjectState,
)
from actuate.schema.version import SCHEMA_VERSION, frozen_schema_path

__all__ = [
    "PROVENANCE_REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "SE3",
    "ActionInterval",
    "CanonicalEpisode",
    "CanonicalFrame",
    "ContactReading",
    "DepthRef",
    "Diversity",
    "EpisodeMeta",
    "FieldStats",
    "HandState",
    "HumanAction",
    "ImageRef",
    "MANOParams",
    "MaskRef",
    "NormStats",
    "ObjectState",
    "ReferenceHandAction",
    "RobotAction",
    "StrategyAlignment",
    "SubgoalFrame",
    "Subtask",
    "dump_json_schema",
    "frozen_schema_path",
    "generate_json_schema",
]


def generate_json_schema() -> dict:
    """The versioned JSON Schema for CanonicalEpisode.

    `actuate schema freeze` writes this to frozen_schema_path(); a test diffs the two, so
    the models cannot drift from the published contract without a version bump.
    """
    schema = CanonicalEpisode.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ActuateCanonicalEpisode"
    schema["x-actuate-schema-version"] = SCHEMA_VERSION
    return schema


def dump_json_schema() -> str:
    return json.dumps(generate_json_schema(), indent=2, sort_keys=True) + "\n"
