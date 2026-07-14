"""The catalog query surface.

AWS Architecture §3: Postgres is the source of truth for *status, consent, provenance*,
and for the query packaging and the dashboard actually need —

    "give me episodes where task=X and embodiment=Y and consent=passed and quality>=4"

`deliverable_episodes()` is that query, and it is the second line of the consent defence:
the delivery path asks the catalog which episodes may ship, and the catalog answers with
a JOIN that cannot return an un-consented one. The `io.consent.DeliveryWriter` guard then
re-checks per write, and IAM refuses the PutObject regardless. Three independent layers,
because any one of them can have a bug.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from actuate.catalog.models import (
    Capture,
    Certification,
    Consent,
    ConsentEvent,
    ConsentEventType,
    Episode,
)
from actuate.config import ConsentStatus, PiiStatus


class ConsentConflict(RuntimeError):
    """Two sources disagree about consent for the same capture.

    Never resolved by picking a winner. The conflict is recorded and the capture is
    blocked until a human decides — "vision says grasped, encoder says open" applied to
    the one decision where being wrong is not a data-quality problem.
    """


def get_consent(session: Session, capture_id: str) -> ConsentStatus | None:
    """None means NO RECORD — which is not the same as PENDING, and blocks."""
    row = session.get(Consent, capture_id)
    return row.status if row else None


def record_consent(
    session: Session,
    capture_id: str,
    status: ConsentStatus,
    conflict_note: str | None = None,
) -> Consent:
    """Upsert consent for a capture.

    **Downgrades are always allowed; upgrades to GRANTED are not, if a conflict is
    recorded.** Revocation must be sticky: once a capture is REVOKED or DENIED, a later
    ingest of the same footage under a new session id must not quietly re-grant it. That
    is precisely the shape of the bug in the local v1 data.
    """
    existing = session.get(Consent, capture_id)
    now = datetime.now(timezone.utc)

    if existing is None:
        row = Consent(
            capture_id=capture_id,
            status=status,
            granted_at=now if status is ConsentStatus.GRANTED else None,
            conflict_note=conflict_note,
        )
        session.add(row)
        return row

    sticky = (ConsentStatus.REVOKED, ConsentStatus.DENIED)
    if existing.status in sticky and status is ConsentStatus.GRANTED:
        raise ConsentConflict(
            f"capture {capture_id}: consent is {existing.status.value}; refusing to "
            f"re-grant. A revocation must survive re-ingestion of the same footage."
        )

    if existing.status is not status:
        note = (
            f"{existing.status.value} -> {status.value}"
            if conflict_note is None
            else conflict_note
        )
        existing.conflict_note = note

    existing.status = status
    if status is ConsentStatus.GRANTED:
        existing.granted_at = now
    if status in sticky:
        existing.revoked_at = now
        # A withdrawal of consent is a DECISION and is logged as one -- never conflated
        # with the duplicate-upload conflicts logged by reconcile_consent().
        log_consent_event(
            session, capture_id, ConsentEventType.REVOKED, status,
            detail="consent withdrawn or denied; this is permanent and cannot be re-granted",
        )
    return existing


#: Least permissive first. The order IS the policy.
_PERMISSIVENESS = [
    ConsentStatus.DENIED,
    ConsentStatus.REVOKED,
    ConsentStatus.PENDING,
    ConsentStatus.GRANTED,
]


def log_consent_event(
    session: Session,
    capture_id: str,
    event_type: ConsentEventType,
    status: ConsentStatus,
    source: str | None = None,
    detail: str | None = None,
) -> ConsentEvent:
    """Append to the consent audit log. Never updated, never deleted."""
    ev = ConsentEvent(
        capture_id=capture_id,
        event_type=event_type,
        status=status,
        source=source,
        detail=detail,
    )
    session.add(ev)
    return ev


def reconcile_consent(
    session: Session,
    capture_id: str,
    observed: list[ConsentStatus],
    sources: list[str] | None = None,
) -> ConsentStatus:
    """Collapse several source records for one capture into a single decision.

    **The least permissive value wins, always.** The four session directories of the same
    real recording carry `pending` on one and `granted` on three; the answer is `pending`,
    not "3 out of 4 said yes". A majority vote here would be a consent gate that a
    duplicate upload can defeat.

    Every observation is logged as a distinct event, and a disagreement is logged as
    DUPLICATE_CONFLICT rather than REVOKED. Both resolve the same safe way, but they are
    not the same thing: one is a filing artifact of uploading the same footage twice, the
    other is a person withdrawing consent. An audit that cannot tell them apart cannot
    answer "why is this blocked?", and that is the only question anyone will ever ask of
    this table.
    """
    if not observed:
        raise ConsentConflict(
            f"capture {capture_id}: no consent records at all. Absence is not permission."
        )

    decision = min(observed, key=_PERMISSIVENESS.index)
    srcs = sources or [None] * len(observed)

    conflicted = len(set(observed)) > 1
    if conflicted:
        for status, src in zip(observed, srcs):
            log_consent_event(
                session,
                capture_id,
                ConsentEventType.DUPLICATE_CONFLICT,
                status,
                source=src,
                detail=(
                    "the same capture arrived under multiple sources carrying different "
                    "consent values; this is a duplicate-upload artifact, NOT a withdrawal "
                    "of consent"
                ),
            )
        note = (
            f"CONFLICT: sources reported {sorted({s.value for s in observed})}; "
            f"resolved to least-permissive {decision.value}. Needs human review."
        )
    else:
        note = None

    record_consent(session, capture_id, decision, conflict_note=note)
    log_consent_event(
        session,
        capture_id,
        ConsentEventType.RECONCILED if conflicted else ConsentEventType.RECORDED,
        decision,
        detail=(
            f"resolved from {sorted(s.value for s in observed)} by least-permissive-wins"
            if conflicted
            else None
        ),
    )
    return decision


def consent_history(session: Session, capture_id: str) -> list[ConsentEvent]:
    """Why is this capture in the state it's in? The only question this table answers."""
    return list(
        session.scalars(
            select(ConsentEvent)
            .where(ConsentEvent.capture_id == capture_id)
            .order_by(ConsentEvent.occurred_at, ConsentEvent.id)
        )
    )


def deliverable_episodes(
    session: Session,
    task_id: str | None = None,
    min_quality: int | None = None,
) -> list[Episode]:
    """Episodes that may actually ship. The fail-closed query.

    An INNER JOIN through consent and certifications, not a filter over a LEFT JOIN: an
    episode whose capture has no consent row, or which has no certification row, simply
    does not appear. It cannot be missed by a forgotten `WHERE ... IS NOT NULL`.
    """
    stmt = (
        select(Episode)
        .join(Capture, Episode.capture_id == Capture.id)
        .join(Consent, Consent.capture_id == Capture.id)
        .join(Certification, Certification.episode_id == Episode.id)
        .where(Consent.status == ConsentStatus.GRANTED)
        .where(Certification.pii_status == PiiStatus.PASSED)
    )
    if task_id is not None:
        stmt = stmt.where(Episode.task_id == task_id)
    if min_quality is not None:
        stmt = stmt.where(Certification.quality >= min_quality)

    return list(session.scalars(stmt))
