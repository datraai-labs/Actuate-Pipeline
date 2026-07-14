"""The migrated capture must be unreachable by customers — attempted, not assumed.

The real capture is now in `actuate-raw-dev` and registered in the catalog. Its consent
resolves to `pending` (three source records said `granted`, one said `pending`, and
least-permissive wins) and its PII check has not been re-run under this pipeline. It is
therefore **not deliverable**, and this test tries to ship it anyway, against all three
layers, and asserts each one refuses.

This is the end-to-end form of the boundary tests: not a synthetic episode constructed to
be un-consented, but the actual footage sitting in the actual bucket with the actual
conflicted consent record that the actual migration produced.

Requires the deployed environment (AWS + the Aurora tunnel). Skipped otherwise, and
reported as skipped — never counted as a pass.
"""

from __future__ import annotations

import os

import pytest

from actuate.config import (
    ConsentStatus,
    Env,
    PiiStatus,
    RigType,
    StorageBackendKind,
    load_settings,
)
from actuate.io import ConsentViolation, DeliveryWriter, S3Backend
from actuate.schema import CanonicalEpisode

pytestmark = [pytest.mark.needs_aws, pytest.mark.real_data]

#: The SHA-256 of the one surviving real capture. It is a content address, so it is stable:
#: the same 190.7 MB recording will always hash to this, and any other bytes will not.
CAPTURE_HASH = "4cff6acb16f7f390a35034eb7ddc088d76e13f1e814214fb8e49180fc1d6bb83"
EPISODE_ID = f"{CAPTURE_HASH[:16]}_ep00"
DELIVERY_KEY = "acme/pretend-dataset/v1/episode.parquet"
PAYLOAD = b"if this reaches the delivery bucket, un-consented human data can reach a customer"


@pytest.fixture(scope="module")
def settings():
    return load_settings(
        env=Env.DEV,
        storage_backend=StorageBackendKind.S3,
        aws_profile=os.environ.get("AWS_PROFILE_ACTUATE", "datraai-admin"),
    )


@pytest.fixture(scope="module")
def catalog_session():
    url = os.environ.get("ACTUATE_DATABASE_URL")
    if not url:
        pytest.skip("no ACTUATE_DATABASE_URL (needs the SSM tunnel to Aurora)")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    eng = create_engine(url, connect_args={"connect_timeout": 60}, pool_pre_ping=True)
    with sessionmaker(bind=eng)() as s:
        yield s


@pytest.fixture(scope="module")
def migrated(catalog_session):
    from actuate.catalog import Capture

    cap = catalog_session.get(Capture, CAPTURE_HASH)
    if cap is None:
        pytest.skip("the real capture has not been migrated into this catalog")
    return cap


# --- the capture is present, and honestly recorded -----------------------------------------


def test_the_capture_is_content_addressed(migrated):
    """Its id IS the hash of its bytes. Identity derived from data, not assigned to it."""
    assert migrated.id == migrated.content_hash == CAPTURE_HASH
    assert migrated.size_bytes > 100_000_000
    assert CAPTURE_HASH in migrated.raw_uri, "the S3 key must be the content address"


def test_consent_resolved_to_pending_not_granted(catalog_session, migrated):
    """Three sources said `granted`, one said `pending`. The answer is `pending`.

    A majority vote here would be a consent gate that a duplicate upload can outvote.
    """
    from actuate.catalog import Consent

    row = catalog_session.get(Consent, CAPTURE_HASH)
    assert row.status is ConsentStatus.PENDING
    assert "CONFLICT" in row.conflict_note
    assert "Needs human review" in row.conflict_note


def test_the_conflict_is_logged_as_a_duplicate_not_a_revocation(catalog_session, migrated):
    """Nobody withdrew consent. We uploaded the same footage twice and the copies disagreed.

    Both block. They are not the same event, and an audit must be able to say which.
    """
    from actuate.catalog import ConsentEventType, consent_history

    events = consent_history(catalog_session, CAPTURE_HASH)
    kinds = {e.event_type for e in events}

    assert ConsentEventType.DUPLICATE_CONFLICT in kinds
    assert ConsentEventType.RECONCILED in kinds
    assert ConsentEventType.REVOKED not in kinds, (
        "a duplicate-upload artifact was logged as a revocation — that would make a filing "
        "error indistinguishable from a person withdrawing consent"
    )

    sources = {e.source for e in events if e.source}
    assert len(sources) >= 2, "the disagreeing sources must be named so a human can look"


# --- THE POINT: try to ship it. All three layers must refuse. -------------------------------


def test_LAYER_2_the_catalog_cannot_even_return_it(catalog_session, migrated):
    from actuate.catalog import deliverable_episodes

    assert deliverable_episodes(catalog_session) == [], (
        "the catalog returned an un-consented episode as deliverable"
    )


def test_LAYER_1_the_code_guard_refuses_the_write(settings, migrated):
    episode = CanonicalEpisode(
        episode_id=EPISODE_ID,
        capture_id=CAPTURE_HASH,
        source_content_hash=CAPTURE_HASH,
        rig=RigType.HEAD_MOUNTED,
        consent=ConsentStatus.PENDING,
        pii_status=PiiStatus.PENDING,
    )
    with pytest.raises(ConsentViolation, match="consent=pending"):
        DeliveryWriter(S3Backend(settings)).put_bytes(episode, DELIVERY_KEY, PAYLOAD)


def test_LAYER_3_IAM_refuses_even_when_all_our_code_is_bypassed(settings, migrated):
    """The last line. Assume every guard above has a bug and write directly, as admin."""
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.Session(
        profile_name=settings.aws_profile, region_name=settings.aws_region
    ).client("s3")

    with pytest.raises(ClientError) as exc:
        s3.put_object(Bucket="actuate-delivery-dev", Key=DELIVERY_KEY, Body=PAYLOAD)

    assert exc.value.response["Error"]["Code"] == "AccessDenied"


def test_the_delivery_bucket_is_still_empty(settings, migrated):
    """The assertion that actually matters. Everything else is mechanism."""
    import boto3

    s3 = boto3.Session(
        profile_name=settings.aws_profile, region_name=settings.aws_region
    ).client("s3")
    objects = s3.list_objects_v2(Bucket="actuate-delivery-dev").get("Contents", [])
    assert not objects, (
        f"{len(objects)} object(s) reached the customer-facing delivery bucket: "
        f"{[o['Key'] for o in objects[:5]]}"
    )
