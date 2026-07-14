"""The consent-aware delivery write guard — FAIL-CLOSED.

AWS Architecture §2: writes to `actuate-delivery-*` are permitted only for the packaging
role, and *"the packaging job refuses to run on any episode whose consent/pii_status is
not passed."* Master Spec §L4 names the bug class to defend against by name: **the
"auto-consent workaround"**.

This module is the code half of that boundary. The IAM half (bucket policy denying
PutObject from every principal but the packaging role) lives in `infra/`. Both exist on
purpose: a bug in app code must not be able to leak un-consented data, and neither must a
mistake in IAM. Either alone is a single point of failure.

Design rules, each of which is a bug this guard is built to make impossible:

  1. **Only an explicit allow passes.** The check is `consent is GRANTED and pii_status is
     PASSED`, never `!= DENIED`. A ConsentStatus member added later (say `EXPIRED`) must
     default to *blocked*, not silently inherit permission.
  2. **Missing is not permissive.** No consent record at all is an error, not PENDING and
     certainly not GRANTED.
  3. **There is no override flag.** `Settings.allow_unconsented_delivery` exists solely so
     that a validator can reject it — see config/settings.py. There is no code path that
     honours it.
  4. **The guard wraps the write, it does not merely advise.** A caller cannot "check
     then write"; they write *through* `DeliveryWriter`, so forgetting the check is not
     an available mistake.
"""

from __future__ import annotations

from pathlib import Path

from actuate.config import Bucket, ConsentStatus, PiiStatus
from actuate.io.backends import StorageBackend
from actuate.schema import CanonicalEpisode


class ConsentViolation(RuntimeError):
    """Raised when something tried to put un-consented data on the delivery path.

    This is not a validation warning. If it fires in production, the pipeline tried to
    ship data it had no right to ship, and the attempt must be loud.
    """


def check_deliverable(
    episode_id: str,
    consent: ConsentStatus | None,
    pii_status: PiiStatus | None,
) -> None:
    """Raise unless this episode is cleared for delivery. Fail-closed.

    `None` for either argument means *no record exists*, which blocks. An episode the
    catalog has never heard of is the most suspicious kind, not the most innocent.
    """
    if consent is None:
        raise ConsentViolation(
            f"episode {episode_id}: NO consent record. Absence is not permission — "
            "the delivery gate is fail-closed (AWS Architecture §2)."
        )
    if pii_status is None:
        raise ConsentViolation(
            f"episode {episode_id}: NO pii_status record. Absence is not permission."
        )
    if consent is not ConsentStatus.GRANTED:
        raise ConsentViolation(
            f"episode {episode_id}: consent={consent.value}, required=granted. "
            "Refusing to write to the delivery bucket."
        )
    if pii_status is not PiiStatus.PASSED:
        raise ConsentViolation(
            f"episode {episode_id}: pii_status={pii_status.value}, required=passed. "
            "Refusing to write to the delivery bucket."
        )


def check_episode_deliverable(episode: CanonicalEpisode) -> None:
    check_deliverable(episode.episode_id, episode.consent, episode.pii_status)


class DeliveryWriter:
    """The ONLY sanctioned way to write to the delivery bucket.

    Every write is preceded by the consent check on the *same call*, so there is no
    window in which a caller has permission but has not verified it, and no way to write
    without verifying. Layers must never reach for the raw backend when the target is
    `Bucket.DELIVERY`.
    """

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    def put_bytes(self, episode: CanonicalEpisode, key: str, data: bytes) -> str:
        check_episode_deliverable(episode)
        return self._backend.put_bytes(Bucket.DELIVERY, key, data)

    def put_file(self, episode: CanonicalEpisode, key: str, src: Path) -> str:
        check_episode_deliverable(episode)
        return self._backend.put_file(Bucket.DELIVERY, key, src)
