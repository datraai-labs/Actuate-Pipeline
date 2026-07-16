"""Part F.1 gates on the LOCAL backend (the delivery bucket was never deployed):
consent blocks regardless of quality, quality floor blocks regardless of consent,
unscored fails closed, and the happy path goes through DeliveryWriter."""

from __future__ import annotations

import json

import pytest

from actuate.config import ConsentStatus, PiiStatus, RigType
from actuate.io.backends import LocalBackend
from actuate.io.consent import ConsentViolation
from actuate.package.deliver import DeliveryRefused, deliver
from actuate.schema import CanonicalEpisode
from actuate.schema.episode import EpisodeMeta


def _episode(consent=ConsentStatus.GRANTED, pii=PiiStatus.PASSED, quality=3):
    return CanonicalEpisode(
        episode_id="ep", capture_id="c" * 64, rig=RigType.HEAD_MOUNTED,
        consent=consent, pii_status=pii,
        episode_meta=EpisodeMeta(quality=quality, speed=2))


@pytest.fixture()
def dataset(tmp_path):
    d = tmp_path / "ds"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "actuate_manifest.json").write_text("{}")
    (d / "data.parquet").write_bytes(b"x" * 64)
    return d


@pytest.fixture()
def backend(tmp_path):
    return LocalBackend(tmp_path / "buckets")


def test_consent_blocks_even_at_quality_5(dataset, backend):
    """quality never substitutes for consent — the Part B gate, at the delivery door."""
    with pytest.raises(ConsentViolation):
        deliver(dataset, "acme", [_episode(consent=ConsentStatus.PENDING, quality=5)],
                backend=backend)


def test_pii_blocks_too(dataset, backend):
    with pytest.raises(ConsentViolation):
        deliver(dataset, "acme", [_episode(pii=PiiStatus.PENDING, quality=5)],
                backend=backend)


def test_quality_floor_blocks_a_consented_episode(dataset, backend):
    with pytest.raises(DeliveryRefused, match="quality 1 < floor"):
        deliver(dataset, "acme", [_episode(quality=1)], backend=backend)


def test_unscored_fails_closed(dataset, backend):
    """Unscored is not 'low quality', it is UNKNOWN — and unknown blocks."""
    with pytest.raises(DeliveryRefused, match="no quality score"):
        deliver(dataset, "acme", [_episode(quality=None)], backend=backend)


def test_one_bad_episode_blocks_the_whole_delivery(dataset, backend):
    """Every NAMED episode must be deliverable, not just the one whose bytes move."""
    good = _episode()
    bad = _episode(consent=ConsentStatus.PENDING)
    with pytest.raises(ConsentViolation):
        deliver(dataset, "acme", [good, bad], backend=backend)


def test_happy_path_writes_via_delivery_writer_and_ledgers(dataset, backend):
    rec = deliver(dataset, "acme", [_episode()], backend=backend)
    assert len(rec.uris) == 2
    assert all("delivery" in u for u in rec.uris)         # landed in the DELIVERY bucket
    assert rec.url is None                                # local: no presigned URL exists
    assert any("never deployed" in n for n in rec.notes)  # and it says so
    ledger = backend.root / "delivery_ledger.jsonl"
    entry = json.loads(ledger.read_text().strip().splitlines()[-1])
    assert entry["customer"] == "acme" and entry["episodes"] == ["ep"]


def test_refuses_an_unpackaged_path(tmp_path, backend):
    with pytest.raises(DeliveryRefused, match="not a directory"):
        deliver(tmp_path / "missing", "acme", [_episode()], backend=backend)
