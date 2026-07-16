"""L7/L8 delivery -- presigned URLs for consent-passed, quality-above-floor datasets.

Every byte goes THROUGH `DeliveryWriter` -- never `S3Backend` directly. That is the
load-bearing detail: the consent check runs on the same call as the write, and
`tests/unit/test_consent_guard.py` exists precisely because "reach for the backend
directly" is the shape of the bug this module must not have. Quality is a SECOND,
independent floor on top of consent: a consented episode below the floor still does not
ship, and a quality-5 episode without consent still does not ship (quality never
substitutes for consent -- proven in Part B's gate 4).

**Deployment honesty:** the delivery bucket has never been deployed (no datraai-admin AWS
profile; StorageStack is synth-only). Against a LocalBackend this module is fully
exercised -- refusals, the writer path, the delivery record; the presigned-URL leg is
implemented but can only be verified once real S3 exists. WRITTEN-ONLY, and says so.
The catalog delivery record is likewise written to a JSONL ledger next to the data
(no Postgres on this machine).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from actuate.io.backends import Bucket, LocalBackend, S3Backend, StorageBackend
from actuate.io.consent import ConsentViolation, DeliveryWriter
from actuate.schema import CanonicalEpisode

#: Episodes below this quality do not ship. §L8 routes honestly-graded LOW-quality data as
#: metadata-labelled robustness data -- but an UNSCORED episode (quality None) is not "low
#: quality", it is unknown, and unknown fails closed.
QUALITY_FLOOR = 2


class DeliveryRefused(RuntimeError):
    """The dataset cannot honestly ship. Not a warning."""


@dataclass
class DeliveryRecord:
    customer: str
    dataset: str
    episode_ids: list[str]
    uris: list[str]
    url: str | None                    # presigned URL (S3 only); None on a local backend
    expires_at: str | None
    created_at: str
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"delivery -> {self.customer}/{self.dataset}: "
                 f"{len(self.uris)} object(s), {len(self.episode_ids)} episode(s)"]
        lines.append(f"  url: {self.url or 'NONE — local backend, nothing publicly reachable'}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _quality_gate(episodes: list[CanonicalEpisode], floor: int) -> None:
    for ep in episodes:
        q = ep.episode_meta.quality
        if q is None:
            raise DeliveryRefused(
                f"episode {ep.episode_id} has no quality score. Unscored is not 'low "
                "quality', it is UNKNOWN -- run `actuate certify run` first. Fail-closed.")
        if q < floor:
            raise DeliveryRefused(
                f"episode {ep.episode_id} quality {q} < floor {floor}. §L8 ships "
                "honestly-graded low-quality data as labelled robustness data through a "
                "separate lane -- not through a customer delivery URL.")


def deliver(
    dataset_dir: Path,
    customer: str,
    episodes: list[CanonicalEpisode],
    *,
    backend: StorageBackend | None = None,
    expires_s: int = 7 * 24 * 3600,
    quality_floor: int = QUALITY_FLOOR,
    dataset_name: str | None = None,
) -> DeliveryRecord:
    """Ship one packaged dataset directory to a customer, gated twice, via DeliveryWriter.

    Raises DeliveryRefused / ConsentViolation rather than shipping anything questionable.
    `backend=None` builds an S3Backend from settings (the production path -- currently
    undeployed); tests pass a LocalBackend.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DeliveryRefused(f"{dataset_dir} is not a directory; package first, then deliver")
    if not episodes:
        raise DeliveryRefused("no episodes given; a delivery must name what it ships")

    # gate 1: consent/PII — enforced again on EVERY write by DeliveryWriter below;
    # checking up front just fails before any byte moves rather than mid-upload.
    # gate 2: quality floor.
    _quality_gate(episodes, quality_floor)

    if backend is None:
        from actuate.config.settings import Settings

        backend = S3Backend(Settings())   # fails loudly without deployed infra — honest
    writer = DeliveryWriter(backend)

    dataset = dataset_name or dataset_dir.name
    prefix = f"{customer}/{dataset}/{datetime.now(timezone.utc):%Y%m%d}"
    uris: list[str] = []
    anchor = episodes[0]               # every write re-checks; anchor carries the consent
    for ep in episodes:
        # every named episode must be deliverable, not just the one whose bytes move
        writer.put_bytes(ep, f"{prefix}/.consent_check/{ep.episode_id}", b"")
    for f in sorted(p for p in dataset_dir.rglob("*") if p.is_file()):
        key = f"{prefix}/{f.relative_to(dataset_dir).as_posix()}"
        uris.append(writer.put_file(anchor, key, f))

    notes: list[str] = []
    url: str | None = None
    expires_at: str | None = None
    if isinstance(backend, S3Backend):
        url = backend._s3.generate_presigned_url(   # the backend's own authed client
            "get_object",
            Params={"Bucket": backend._bucket_name(Bucket.DELIVERY),
                    "Key": f"{prefix}/actuate_manifest.json"},
            ExpiresIn=expires_s)
        expires_at = datetime.fromtimestamp(time.time() + expires_s,
                                            tz=timezone.utc).isoformat()
    else:
        notes.append("local backend: no presigned URL exists (the delivery bucket was "
                     "never deployed). WRITTEN-ONLY until real S3 exists.")

    record = DeliveryRecord(
        customer=customer, dataset=dataset,
        episode_ids=[e.episode_id for e in episodes], uris=uris,
        url=url, expires_at=expires_at,
        created_at=datetime.now(timezone.utc).isoformat(), notes=notes,
    )
    # catalog leg: a JSONL ledger (no Postgres on this machine — written-only, stated)
    ledger = (backend.root / "delivery_ledger.jsonl"
              if isinstance(backend, LocalBackend) else Path("delivery_ledger.jsonl"))
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "customer": customer, "dataset": dataset, "episodes": record.episode_ids,
            "n_objects": len(uris), "created_at": record.created_at,
            "expires_at": expires_at}) + "\n")
    return record


__all__ = ["deliver", "DeliveryRecord", "DeliveryRefused", "ConsentViolation",
           "QUALITY_FLOOR"]
