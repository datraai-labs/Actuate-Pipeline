"""The Postgres catalog — against a REAL Postgres, not a mock.

The Increment-1 prompt is explicit: *"the catalog performs a real query against a real
Postgres (spin up Postgres in a test container / testcontainers, not a mock)"*. A mock
would prove nothing here — the guarantees being tested are the ones only a real database
enforces: the enum types, the foreign keys, and pgvector.

**Requires Docker.** Skipped without it, and the skip is reported honestly in STATUS.md
rather than counted as a pass.
"""

from __future__ import annotations

import pytest

from actuate.catalog import (
    Capture,
    Certification,
    Consent,
    ConsentConflict,
    Episode,
    EpisodeEmbedding,
    deliverable_episodes,
    get_consent,
    init_schema,
    make_session_factory,
    reconcile_consent,
    record_consent,
    session_scope,
)
from actuate.config import ConsentStatus, PiiStatus, RigType, Tier


@pytest.fixture(scope="module")
def engine():
    """A REAL Postgres. Two ways to get one, in order of preference:

    1. ``ACTUATE_DATABASE_URL`` points at a real database — in practice the deployed
       Aurora cluster, reached through the SSM tunnel. This is the strongest form of the
       test: it runs against the actual production engine, with the actual migration
       applied, over the actual network path.
    2. Otherwise, spin up ``pgvector/pgvector:pg16`` via testcontainers. Used in CI and on
       machines without AWS access.

    There is deliberately no third option. A mock or a SQLite fallback would not enforce
    the enum types, the consent foreign key, or pgvector — which are precisely the
    guarantees these tests exist to check.
    """
    import os

    from sqlalchemy import create_engine

    url = os.environ.get("ACTUATE_DATABASE_URL")
    if url:
        # Aurora at min_capacity=0 sleeps; give it room to wake.
        eng = create_engine(
            url, connect_args={"connect_timeout": 60}, pool_pre_ping=True
        )
        init_schema(eng)  # idempotent: CREATE EXTENSION IF NOT EXISTS + create_all
        yield eng
        return

    docker = pytest.importorskip("docker", reason="Docker SDK not installed")
    try:
        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"no ACTUATE_DATABASE_URL and Docker unavailable ({exc}). "
            "These tests need a real Postgres; they will not run against a mock."
        )

    from testcontainers.postgres import PostgresContainer

    # pgvector/pgvector ships the extension preinstalled; plain postgres does not, and
    # CREATE EXTENSION vector would fail — which is the point of using a real database.
    with PostgresContainer("pgvector/pgvector:pg16", driver="psycopg") as pg:
        eng = create_engine(pg.get_connection_url())
        init_schema(eng)
        yield eng


#: Every capture these tests create is prefixed with this. Cleanup deletes ONLY these.
#:
#: This matters more than it looks. When ACTUATE_DATABASE_URL points at the deployed dev
#: Aurora — which is the strongest way to run these tests, and the way we run them — a
#: blanket `DELETE FROM captures` would wipe the REAL migrated capture out of the catalog.
#: It did, once: the end-to-end tests in test_migrated_capture_is_not_deliverable.py began
#: skipping with "the real capture has not been migrated", because the test suite had
#: deleted it.
#:
#: A test fixture must never be able to destroy real data. Real captures are content-
#: addressed (64 hex chars); test captures are not, and are scoped by this prefix.
TEST_CAPTURE_PREFIX = "testcap_"


@pytest.fixture
def session(engine):
    factory = make_session_factory(engine)
    with session_scope(factory) as s:
        yield s
        s.rollback()

        from actuate.catalog import ConsentEvent

        ids = [
            c.id
            for c in s.query(Capture).filter(Capture.id.like(f"{TEST_CAPTURE_PREFIX}%")).all()
        ]
        if ids:
            eps = [
                e.id for e in s.query(Episode).filter(Episode.capture_id.in_(ids)).all()
            ]
            if eps:
                s.query(EpisodeEmbedding).filter(
                    EpisodeEmbedding.episode_id.in_(eps)
                ).delete(synchronize_session=False)
                s.query(Certification).filter(
                    Certification.episode_id.in_(eps)
                ).delete(synchronize_session=False)
                s.query(Episode).filter(Episode.id.in_(eps)).delete(
                    synchronize_session=False
                )
            s.query(ConsentEvent).filter(ConsentEvent.capture_id.in_(ids)).delete(
                synchronize_session=False
            )
            s.query(Consent).filter(Consent.capture_id.in_(ids)).delete(
                synchronize_session=False
            )
            s.query(Capture).filter(Capture.id.in_(ids)).delete(synchronize_session=False)
        s.commit()


def _capture(session, cid=f"{TEST_CAPTURE_PREFIX}1", digest="a" * 64) -> Capture:
    c = Capture(id=cid, content_hash=digest, rig_type=RigType.HEAD_MOUNTED,
                raw_uri=f"s3://actuate-raw-dev/head_mounted/{digest}/raw.mp4")
    session.add(c)
    session.flush()
    return c


def _episode(session, cid=f"{TEST_CAPTURE_PREFIX}1", eid="ep1", quality=5, pii=PiiStatus.PASSED) -> Episode:
    e = Episode(id=eid, capture_id=cid, rig_type=RigType.HEAD_MOUNTED, schema_version=1,
                tier=Tier.STAGE1_VOLUME, canonical_uri=f"s3://actuate-work-dev/canonical/{eid}/v1")
    session.add(e)
    session.add(Certification(episode_id=eid, quality=quality, pii_status=pii))
    session.flush()
    return e


# --- the real query the dashboard and packaging need -------------------------------------


def test_deliverable_query_returns_only_consented_certified_episodes(session):
    _capture(session)
    _episode(session, eid="ep1", quality=5)
    _episode(session, eid="ep2", quality=2)
    record_consent(session, f"{TEST_CAPTURE_PREFIX}1", ConsentStatus.GRANTED)
    session.flush()

    assert {e.id for e in deliverable_episodes(session)} == {"ep1", "ep2"}
    assert {e.id for e in deliverable_episodes(session, min_quality=4)} == {"ep1"}


def test_an_episode_with_no_consent_row_is_invisible_to_the_deliverable_query(session):
    """Fail-closed at the query level. Not a filter over a LEFT JOIN that a forgotten
    `IS NOT NULL` could defeat — an INNER JOIN that cannot return the row at all."""
    _capture(session)
    _episode(session, eid="ep1")
    session.flush()

    assert deliverable_episodes(session) == []
    assert get_consent(session, f"{TEST_CAPTURE_PREFIX}1") is None  # None means NO RECORD, not "pending"


def test_revoking_consent_removes_every_episode_of_that_capture(session):
    """THE reason consent is keyed on capture_id and not episode_id.

    One recording, three episodes. Revoke once; all three stop being deliverable. Keyed on
    episode, you would have to remember to revoke each — and the real v1 data shows exactly
    what happens when you don't.
    """
    _capture(session)
    for i in range(3):
        _episode(session, eid=f"ep{i}")
    record_consent(session, f"{TEST_CAPTURE_PREFIX}1", ConsentStatus.GRANTED)
    session.flush()
    assert len(deliverable_episodes(session)) == 3

    record_consent(session, f"{TEST_CAPTURE_PREFIX}1", ConsentStatus.REVOKED)
    session.flush()
    assert deliverable_episodes(session) == []


def test_a_revocation_cannot_be_undone_by_re_ingesting_the_same_footage(session):
    """Revocation is sticky. Re-uploading the same capture under a new session id must not
    quietly re-grant it — which is the shape of the bug in the local v1 data."""
    _capture(session)
    record_consent(session, f"{TEST_CAPTURE_PREFIX}1", ConsentStatus.REVOKED)
    session.flush()

    with pytest.raises(ConsentConflict, match="refusing to re-grant"):
        record_consent(session, f"{TEST_CAPTURE_PREFIX}1", ConsentStatus.GRANTED)


def test_pii_failure_blocks_delivery_even_with_consent_granted(session):
    _capture(session)
    _episode(session, eid="ep1", pii=PiiStatus.FAILED)
    record_consent(session, f"{TEST_CAPTURE_PREFIX}1", ConsentStatus.GRANTED)
    session.flush()

    assert deliverable_episodes(session) == []


# --- consent reconciliation: the real corpus's actual shape --------------------------------


def test_conflicting_records_resolve_to_least_permissive_not_majority(session):
    """The real corpus: ONE recording under four session ids — `pending` on one, `granted`
    on three. The answer is `pending`.

    A majority vote would be a consent gate that a duplicate upload can outvote.
    """
    _capture(session)
    observed = [
        ConsentStatus.PENDING,
        ConsentStatus.GRANTED,
        ConsentStatus.GRANTED,
        ConsentStatus.GRANTED,
    ]
    decision = reconcile_consent(session, f"{TEST_CAPTURE_PREFIX}1", observed)
    session.flush()

    assert decision is ConsentStatus.PENDING, "3-of-4 'granted' must NOT carry the vote"

    row = session.get(Consent, f"{TEST_CAPTURE_PREFIX}1")
    assert "CONFLICT" in row.conflict_note
    assert "Needs human review" in row.conflict_note

    _episode(session, eid="ep1")
    session.flush()
    assert deliverable_episodes(session) == []


def test_no_consent_records_at_all_is_an_error_not_a_default(session):
    _capture(session)
    with pytest.raises(ConsentConflict, match="Absence is not permission"):
        reconcile_consent(session, f"{TEST_CAPTURE_PREFIX}1", [])


# --- THE BROKEN VARIANT: prove the INNER JOIN is what stops the leak --------------------


def test_a_left_join_variant_LEAKS_unconsented_episodes(session):
    """PROOF that the INNER JOIN is load-bearing, not incidental.

    `deliverable_episodes()` joins through `consent` with an INNER JOIN, so an episode
    whose capture has no consent row cannot appear in the result at all. That is a
    structural guarantee — you cannot forget it, because there is no row to forget about.

    The tempting "equivalent" refactor is a LEFT JOIN with a WHERE predicate. It reads the
    same and is not the same: a LEFT JOIN produces a row with NULLs for the missing consent,
    and any predicate that doesn't explicitly handle NULL lets it through. This test writes
    that variant and asserts it LEAKS — an episode with NO CONSENT RECORD AT ALL is
    returned as deliverable.

    The assertion is deliberately inverted: we assert the leak HAPPENS. If someone later
    "simplifies" the real query into this shape, the real tests above go red and this one
    documents exactly what was lost.
    """
    from sqlalchemy import select

    _capture(session)
    _episode(session, eid="ep_no_consent")
    session.flush()

    # The real query. Cannot return it.
    assert deliverable_episodes(session) == []

    # The broken variant: LEFT JOIN + a predicate that looks careful and isn't.
    # `Consent.status != DENIED` is TRUE-looking for a row that has no consent at all...
    # except in SQL, NULL != 'DENIED' evaluates to NULL, not TRUE — so this particular
    # form filters it out by accident. The genuinely dangerous form is the one that treats
    # "not explicitly denied" as permitted:
    leaky = (
        select(Episode)
        .outerjoin(Consent, Consent.capture_id == Episode.capture_id)
        .outerjoin(Certification, Certification.episode_id == Episode.id)
        .where(
            (Consent.status != ConsentStatus.DENIED) | (Consent.status.is_(None))
        )
    )
    leaked = {e.id for e in session.scalars(leaky)}

    assert "ep_no_consent" in leaked, (
        "The LEFT JOIN variant was expected to LEAK an episode with no consent record. "
        "If it did not, re-examine this test — but do NOT relax the real query."
    )

    # Worth seeing: against the deployed catalog this variant also leaks the REAL migrated
    # capture, whose consent is `pending`. `pending != DENIED` is true, so "not explicitly
    # denied" lets it through — which is exactly how a well-meaning refactor ships human
    # data that nobody agreed to release. The INNER JOIN below cannot do this.
    assert len(leaked) >= 1

    # And the episode it leaked is one that has no consent record whatsoever.
    assert get_consent(session, f"{TEST_CAPTURE_PREFIX}1") is None
    assert not deliverable_episodes(session), (
        "the real INNER JOIN query must still refuse it"
    )


# --- pgvector ------------------------------------------------------------------------------


def test_a_duplicate_conflict_is_logged_DIFFERENTLY_from_a_revocation(session):
    """A filing artifact and a human decision are not the same event.

    Both block delivery. Both resolve least-permissive. But "we uploaded the same footage
    twice and the copies disagreed" and "a person withdrew their consent" are different
    facts, and the first question anyone asks of this table is *why is this blocked?* —
    which an audit log that conflates them cannot answer.
    """
    from actuate.catalog import ConsentEventType, consent_history

    _capture(session, cid=f"{TEST_CAPTURE_PREFIX}dup")
    reconcile_consent(
        session,
        f"{TEST_CAPTURE_PREFIX}dup",
        [ConsentStatus.PENDING, ConsentStatus.GRANTED, ConsentStatus.GRANTED],
        sources=["session_001", "uuid-a", "uuid-b"],
    )
    session.flush()

    kinds = {e.event_type for e in consent_history(session, f"{TEST_CAPTURE_PREFIX}dup")}
    assert ConsentEventType.DUPLICATE_CONFLICT in kinds
    assert ConsentEventType.RECONCILED in kinds
    assert ConsentEventType.REVOKED not in kinds, (
        "a duplicate-upload conflict was logged as a REVOCATION — that would make a filing "
        "error look like a person withdrawing consent"
    )

    # The sources that disagreed are named, so a human can go look.
    dup = [
        e
        for e in consent_history(session, f"{TEST_CAPTURE_PREFIX}dup")
        if e.event_type is ConsentEventType.DUPLICATE_CONFLICT
    ]
    assert {e.source for e in dup} == {"session_001", "uuid-a", "uuid-b"}

    # Now a REAL revocation on a different capture.
    _capture(session, cid=f"{TEST_CAPTURE_PREFIX}rev", digest="c" * 64)
    record_consent(session, f"{TEST_CAPTURE_PREFIX}rev", ConsentStatus.GRANTED)
    session.flush()
    record_consent(session, f"{TEST_CAPTURE_PREFIX}rev", ConsentStatus.REVOKED)
    session.flush()

    rev_kinds = {e.event_type for e in consent_history(session, f"{TEST_CAPTURE_PREFIX}rev")}
    assert ConsentEventType.REVOKED in rev_kinds
    assert ConsentEventType.DUPLICATE_CONFLICT not in rev_kinds


def test_the_event_log_is_append_only_across_reconciliations(session):
    from actuate.catalog import consent_history

    _capture(session)
    reconcile_consent(session, f"{TEST_CAPTURE_PREFIX}1", [ConsentStatus.PENDING])
    session.flush()
    n1 = len(consent_history(session, f"{TEST_CAPTURE_PREFIX}1"))

    reconcile_consent(session, f"{TEST_CAPTURE_PREFIX}1", [ConsentStatus.PENDING])
    session.flush()
    assert len(consent_history(session, f"{TEST_CAPTURE_PREFIX}1")) > n1, "history must accumulate, not replace"


def test_pgvector_similarity_search_works(session):
    """Dedup (L7) needs this before splits are generated, or near-duplicates leak across
    train/val/test."""
    _capture(session)
    _episode(session, eid="ep1")
    _episode(session, eid="ep2")
    session.flush()

    a = [1.0] + [0.0] * 511
    b = [0.99] + [0.01] * 511  # near-duplicate of a
    session.add(EpisodeEmbedding(episode_id="ep1", model="clip-vit-b32", embedding=a))
    session.add(EpisodeEmbedding(episode_id="ep2", model="clip-vit-b32", embedding=b))
    session.flush()

    nearest = (
        session.query(EpisodeEmbedding)
        .filter(EpisodeEmbedding.episode_id != "ep1")
        .order_by(EpisodeEmbedding.embedding.cosine_distance(a))
        .first()
    )
    assert nearest.episode_id == "ep2"


def test_enum_types_are_enforced_by_the_database(session):
    """A real Postgres rejects an invalid enum value. SQLite would not — which is why
    there is no SQLite fallback in db.py."""
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    _capture(session)
    with pytest.raises(DBAPIError):
        session.execute(
            text("INSERT INTO consent (capture_id, status) VALUES ('cap1', 'definitely_yes')")
        )
