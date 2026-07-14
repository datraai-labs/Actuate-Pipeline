"""The consent guard — and the demonstration that it is load-bearing.

Master Spec §L4 names the bug class explicitly: **the auto-consent workaround**. A test
that merely passes with the guard in place proves nothing about the guard. So
`test_removing_the_guard_leaks_data` reimplements the delivery write *without* the check
and asserts that un-consented data reaches the delivery bucket — i.e. it proves the guard
is the only thing standing between us and a leak.

If someone later "simplifies" DeliveryWriter by dropping the check, the other tests here
go red. If someone deletes these tests instead, the leak test is the one that documents
what was lost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from actuate.config import Bucket, ConsentStatus, PiiStatus, RigType
from actuate.io import ConsentViolation, DeliveryWriter, LocalBackend, check_deliverable
from actuate.schema import CanonicalEpisode

PAYLOAD = b"customer-facing training data"
KEY = "acme/dataset-v1/episode.parquet"


def _episode(consent: ConsentStatus, pii: PiiStatus) -> CanonicalEpisode:
    return CanonicalEpisode(
        episode_id="ep0",
        capture_id="cap0",
        rig=RigType.HEAD_MOUNTED,
        consent=consent,
        pii_status=pii,
    )


@pytest.fixture
def backend(tmp_path: Path) -> LocalBackend:
    return LocalBackend(tmp_path)


# --- the gate opens only on an explicit, complete allow ---------------------------------


def test_granted_and_passed_may_ship(backend: LocalBackend):
    ep = _episode(ConsentStatus.GRANTED, PiiStatus.PASSED)
    DeliveryWriter(backend).put_bytes(ep, KEY, PAYLOAD)
    assert backend.get_bytes(Bucket.DELIVERY, KEY) == PAYLOAD


@pytest.mark.parametrize(
    "consent,pii",
    [
        (ConsentStatus.PENDING, PiiStatus.PASSED),
        (ConsentStatus.DENIED, PiiStatus.PASSED),
        (ConsentStatus.REVOKED, PiiStatus.PASSED),
        (ConsentStatus.GRANTED, PiiStatus.PENDING),
        (ConsentStatus.GRANTED, PiiStatus.FAILED),
        (ConsentStatus.PENDING, PiiStatus.PENDING),
    ],
)
def test_anything_short_of_both_gates_is_blocked(backend, consent, pii):
    ep = _episode(consent, pii)
    with pytest.raises(ConsentViolation):
        DeliveryWriter(backend).put_bytes(ep, KEY, PAYLOAD)
    assert not backend.exists(Bucket.DELIVERY, KEY), "un-consented bytes reached delivery"


def test_a_missing_record_blocks_rather_than_defaults_open():
    """Absence is not permission. An episode the catalog has never heard of is the most
    suspicious kind, not the most innocent."""
    with pytest.raises(ConsentViolation, match="NO consent record"):
        check_deliverable("ep0", None, PiiStatus.PASSED)
    with pytest.raises(ConsentViolation, match="NO pii_status record"):
        check_deliverable("ep0", ConsentStatus.GRANTED, None)


def test_default_episode_is_not_deliverable():
    """The schema's defaults must not open the gate. A CanonicalEpisode constructed with
    no consent argument at all is PENDING, not GRANTED."""
    ep = CanonicalEpisode(episode_id="e", capture_id="c", rig=RigType.HEAD_MOUNTED)
    assert not ep.is_deliverable
    assert "consent=pending" in ep.delivery_block_reason()


# --- THE DEMONSTRATION: remove the guard, and data leaks --------------------------------


def test_removing_the_guard_leaks_data(backend: LocalBackend):
    """PROOF that the guard is what prevents the leak, not something else.

    This is the "broken variant" the Master Spec requires: it writes to the delivery
    bucket exactly as DeliveryWriter does, minus the consent check — the shape a
    well-meaning refactor or an "auto-consent workaround" would take — and shows that
    un-consented data lands in the customer-facing bucket.

    The assertion is deliberately inverted: we assert the LEAK HAPPENS. If someone makes
    the raw backend itself consent-aware, this test goes red and should be re-examined,
    not deleted.
    """
    ep = _episode(ConsentStatus.DENIED, PiiStatus.FAILED)

    # The bug: reaching for the backend directly instead of going through DeliveryWriter.
    backend.put_bytes(Bucket.DELIVERY, KEY, PAYLOAD)

    assert backend.get_bytes(Bucket.DELIVERY, KEY) == PAYLOAD, (
        "Without the guard, data whose consent is DENIED and whose PII check FAILED is "
        "now sitting in the customer-facing delivery bucket. This is the leak."
    )

    # And with the guard, the same write is refused.
    backend.delete(Bucket.DELIVERY, KEY)
    with pytest.raises(ConsentViolation):
        DeliveryWriter(backend).put_bytes(ep, KEY, PAYLOAD)
    assert not backend.exists(Bucket.DELIVERY, KEY)


def test_settings_cannot_be_configured_to_bypass_consent():
    """There is no supported way to turn the gate off — the config validator refuses.

    An override flag that exists 'just for testing' is how the auto-consent workaround
    gets into production.
    """
    from pydantic import ValidationError

    from actuate.config import load_settings

    with pytest.raises(ValidationError, match="not a supported configuration"):
        load_settings(allow_unconsented_delivery=True)
