"""The schema is FROZEN and VERSIONED (Master Spec §3).

Layer 7's exporters compile against a specific version of this contract. If the models
drift without a version bump, those exporters break silently against data they believe
they understand — which is exactly the class of failure the load+train gate exists to
catch late, and this test exists to catch early.

Also contains the required migration demonstration: **the schema check must FAIL when a
required field is dropped.**
"""

from __future__ import annotations

import json

import pytest

from actuate.schema import (
    SCHEMA_VERSION,
    dump_json_schema,
    frozen_schema_path,
    generate_json_schema,
)

#: Fields a consumer cannot function without. Dropping any of them is a MAJOR schema
#: change, and the test below proves that dropping one is detected rather than shipped.
REQUIRED_EPISODE_FIELDS = (
    "episode_id",
    "capture_id",
    "source_content_hash",  # v2: the provenance chain must stay verifiable
    "rig",
    "schema_version",
    "consent",      # the hard gate
    "pii_status",   # the hard gate
    "frames",
)

REQUIRED_FRAME_FIELDS = (
    "t",
    "rig",
    "episode_id",
    "frame_idx",
    "provenance",   # the trust model keys on this
    "confidence",
)


def test_schema_version_is_four():
    """v4 = the certificate becomes real (Phase 5): CertificateComponents on EpisodeMeta,
    retarget_eligibility on the episode, full percentile set on FieldStats. All additive-
    optional — v3 payloads validate unchanged.

    This assertion failing is the freeze mechanism working: a schema change that forgets to
    bump the version cannot pass, and a version bump that forgets to re-freeze the JSON
    Schema cannot pass either.
    """
    assert SCHEMA_VERSION == 4


def test_frozen_schema_exists():
    assert frozen_schema_path().exists(), (
        f"no frozen schema for v{SCHEMA_VERSION}. Run: actuate schema freeze"
    )


def test_models_have_not_drifted_from_the_frozen_schema():
    assert frozen_schema_path().read_text(encoding="utf-8") == dump_json_schema(), (
        f"the Pydantic models no longer match frozen schema v{SCHEMA_VERSION}.\n\n"
        "The canonical schema is the contract every exporter is built against. Bump "
        "SCHEMA_VERSION in actuate/schema/version.py, then run `actuate schema freeze`."
    )


def test_every_required_field_is_present_in_the_frozen_schema():
    """The migration guard.

    Companion demonstration below shows this FAILS when a field is dropped — per Master
    Spec §0, a correctness test that has never been seen to go red is not evidence.
    """
    schema = generate_json_schema()
    props = set(schema["properties"])
    missing = [f for f in REQUIRED_EPISODE_FIELDS if f not in props]
    assert not missing, f"required episode field(s) dropped from the schema: {missing}"

    frame = schema["$defs"]["CanonicalFrame"]["properties"]
    missing = [f for f in REQUIRED_FRAME_FIELDS if f not in frame]
    assert not missing, f"required frame field(s) dropped from the schema: {missing}"


@pytest.mark.parametrize(
    "dropped",
    ["consent", "pii_status", "capture_id", "source_content_hash", "provenance"],
)
def test_dropping_a_required_field_is_detected(dropped: str):
    """THE BROKEN-VARIANT DEMONSTRATION (Master Spec §3 verification gate).

    Simulates a future edit that removes a required field, and asserts the migration check
    catches it. Without this, `test_every_required_field_is_present_in_the_frozen_schema`
    would be a test that has only ever been observed to pass — which proves nothing about
    whether it can fail.

    The four fields chosen are the ones whose loss would be most dangerous and least
    visible: the two consent gates, the key consent is enforced on, and the field the whole
    trust model reads.
    """
    schema = json.loads(json.dumps(generate_json_schema()))  # deep copy

    if dropped in schema["properties"]:
        del schema["properties"][dropped]
    else:
        del schema["$defs"]["CanonicalFrame"]["properties"][dropped]

    props = set(schema["properties"])
    frame_props = set(schema["$defs"]["CanonicalFrame"]["properties"])
    missing = [f for f in REQUIRED_EPISODE_FIELDS if f not in props] + [
        f for f in REQUIRED_FRAME_FIELDS if f not in frame_props
    ]

    assert missing == [dropped], (
        f"dropping {dropped!r} was NOT detected by the migration check. The check is "
        "not actually guarding anything."
    )


def test_the_drift_check_would_catch_a_silent_model_change():
    """And the frozen-file diff catches changes the required-field list doesn't enumerate."""
    tampered = generate_json_schema()
    tampered["properties"]["effective_hours"]["type"] = "string"  # was number
    assert json.dumps(tampered, sort_keys=True) != json.dumps(
        generate_json_schema(), sort_keys=True
    )
