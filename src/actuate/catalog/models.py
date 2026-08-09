"""Postgres catalog — SQLAlchemy models mirroring AWS Architecture §3.

**Rule: blobs in S3, everything queryable in Postgres with an S3 URI pointer.**
No large arrays in the DB. A column here holds an `s3://...` string, never a point cloud.

The catalog is the source of truth for *status, consent, provenance*, and the query the
dashboard and packaging actually need:

    "give me episodes where task=X and embodiment=Y and consent=passed and quality>=4"

### The one place this diverges from the doc, and why

The doc's table list has no `captures` table, but it keys `consent` on `capture_id`. That
is correct and load-bearing — **one physical recording yields many episodes, and revoking
consent must revoke all of them at once** — but it needs something to own capture
identity. Hence `captures`, with a content hash of the raw video as the natural key.

This is not theoretical. The local v1 data has the *same* 95-second recording under four
session ids: `session_001` records consent `pending`, its three UUID copies record
`granted`. Keyed on session, revoking one would leave three shippable. Keyed on capture,
they are one row and one decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from actuate.config import ConsentStatus, PiiStatus, RigType, Tier

# The catalog is part of the `aws` extra. Importing actuate.catalog without it installed
# is a configuration error, not something to silently degrade around — a fallback column
# type would let a dedup index be created that isn't actually a vector index.
try:
    from pgvector.sqlalchemy import Vector
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "actuate.catalog requires the aws extra (pgvector, sqlalchemy, psycopg): "
        "pip install -e '.[aws]'"
    ) from exc


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# --------------------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------------------


class Rig(Base, TimestampMixin):
    __tablename__ = "rigs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rig_type: Mapped[RigType] = mapped_column(SAEnum(RigType, name="rig_type"))
    description: Mapped[str | None] = mapped_column(Text)


class Embodiment(Base, TimestampMixin):
    __tablename__ = "embodiments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    arm_dof: Mapped[int] = mapped_column(Integer)
    hand_dof: Mapped[int | None] = mapped_column(Integer)
    urdf_uri: Mapped[str | None] = mapped_column(Text)
    #: True only after a real sim replay passed (Master Spec §L5 gate). Default False —
    #: an embodiment is not retarget-eligible because someone added a row for it.
    sim_validated: Mapped[bool] = mapped_column(Boolean, default=False)


class Scene(Base, TimestampMixin):
    __tablename__ = "scenes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)


class Demonstrator(Base, TimestampMixin):
    """Reported SEPARATELY from scene (EgoVerse): collapsing scene and demonstrator into
    one 'diversity' number hides which axis is actually thin."""

    __tablename__ = "demonstrators"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    instruction: Mapped[str] = mapped_column(Text)
    family: Mapped[str | None] = mapped_column(String(64))


# --------------------------------------------------------------------------------------
# Capture + consent — the boundary
# --------------------------------------------------------------------------------------


class Capture(Base, TimestampMixin):
    """One physical recording. The unit consent is granted or revoked on."""

    __tablename__ = "captures"

    #: == content_hash. The capture id IS the SHA-256 of the raw bytes (schema v2,
    #: actuate.ingest.content_address). Identity is derived from the data, not assigned to
    #: it -- so the same footage uploaded twice is one row, not two that can disagree.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #: Bytes on disk. Recorded so a payload/metadata mismatch is expressible and therefore
    #: checkable -- the 1.49 MB video filed under 181 MB metadata could not be detected in
    #: Increment 1 because nothing held both facts.
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    rig_type: Mapped[RigType] = mapped_column(SAEnum(RigType, name="rig_type"))
    raw_uri: Mapped[str] = mapped_column(Text)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)

    consent: Mapped[Consent] = relationship(back_populates="capture", uselist=False)
    episodes: Mapped[list[Episode]] = relationship(back_populates="capture")


class Consent(Base, TimestampMixin):
    """FAIL-CLOSED. Keyed on capture, not episode (AWS Architecture §3).

    `status` has **no server default**. A capture with no consent row is not `pending` —
    it is a capture we have no record for, and the delivery guard blocks it. Absence is
    never permission.
    """

    __tablename__ = "consent"

    capture_id: Mapped[str] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[ConsentStatus] = mapped_column(SAEnum(ConsentStatus, name="consent_status"))
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: Set when two source records disagreed about the same capture (this happened in the
    #: real v1 data). An unresolved conflict resolves to BLOCKED, never to the permissive
    #: value — and it is recorded here rather than silently reconciled.
    conflict_note: Mapped[str | None] = mapped_column(Text)

    capture: Mapped[Capture] = relationship(back_populates="consent")


# --------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------


class ConsentEventType(str, PyEnum):
    """What happened to a capture's consent, and why.

    These all *resolve* the same safe way -- least-permissive wins -- but they are not the
    same event, and an audit that cannot tell them apart is not an audit:

    RECORDED           first consent decision for this capture.
    DUPLICATE_CONFLICT the same footage arrived again carrying a DIFFERENT consent value.
                       This is a data-hygiene artifact of duplicate uploads. It says
                       nothing about what the demonstrator wants.
    REVOKED            a human withdrew consent. This is a decision, and it is permanent.
    RECONCILED         several source records for one capture were collapsed to the
                       least-permissive value.

    Collapsing DUPLICATE_CONFLICT into REVOKED would make a filing error look like a
    withdrawal of consent; collapsing it the other way would let a stale duplicate
    resurrect data someone asked us to stop using. Neither is acceptable, so they are
    stored distinctly.
    """

    RECORDED = "recorded"
    DUPLICATE_CONFLICT = "duplicate_conflict"
    REVOKED = "revoked"
    RECONCILED = "reconciled"


class ConsentEvent(Base):
    """Append-only consent audit log. Never updated, never deleted.

    `consent` holds the current state; this holds how it got there. The distinction matters
    the first time somebody asks "why is this capture blocked?" -- and the answer is either
    "a person withdrew consent" or "we uploaded it twice and the second copy disagreed",
    which are very different answers.
    """

    __tablename__ = "consent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[ConsentEventType] = mapped_column(
        SAEnum(ConsentEventType, name="consent_event_type")
    )
    #: The status this event resulted in.
    status: Mapped[ConsentStatus] = mapped_column(SAEnum(ConsentStatus, name="consent_status"))
    #: Where the claim came from -- e.g. the session directory that carried it.
    source: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Episode(Base, TimestampMixin):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    capture_id: Mapped[str] = mapped_column(ForeignKey("captures.id", ondelete="CASCADE"), index=True)
    rig_type: Mapped[RigType] = mapped_column(SAEnum(RigType, name="rig_type"))

    scene_id: Mapped[str | None] = mapped_column(ForeignKey("scenes.id"), index=True)
    demonstrator_id: Mapped[str | None] = mapped_column(ForeignKey("demonstrators.id"), index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), index=True)

    #: -> S3. The catalog points; it does not carry.
    canonical_uri: Mapped[str | None] = mapped_column(Text)
    schema_version: Mapped[int] = mapped_column(Integer)

    tier: Mapped[Tier | None] = mapped_column(SAEnum(Tier, name="tier"))
    effective_hours: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)

    capture: Mapped[Capture] = relationship(back_populates="episodes")
    certification: Mapped[Certification] = relationship(back_populates="episode", uselist=False)

    __table_args__ = (
        Index("ix_episodes_task_tier", "task_id", "tier"),
    )


class Certification(Base, TimestampMixin):
    """L4 output. π0.7 metadata + the PII half of the delivery gate."""

    __tablename__ = "certifications"

    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True
    )
    quality: Mapped[int | None] = mapped_column(Integer)
    speed: Mapped[int | None] = mapped_column(Integer)
    mistakes: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    strategy_alignment: Mapped[str | None] = mapped_column(Text)

    #: The other half of the fail-closed gate. No default: absence blocks.
    pii_status: Mapped[PiiStatus] = mapped_column(SAEnum(PiiStatus, name="pii_status"))

    episode: Mapped[Episode] = relationship(back_populates="certification")

    __table_args__ = (
        CheckConstraint("quality IS NULL OR (quality >= 1 AND quality <= 5)", name="ck_quality_1_5"),
    )


class Annotation(Base, TimestampMixin):
    __tablename__ = "annotations"
    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True
    )
    paraphrases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    subtasks_uri: Mapped[str | None] = mapped_column(Text)
    subgoal_frames: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))


class RetargetResult(Base, TimestampMixin):
    """L5 output. Empty until L5 exists — which is the honest state today."""

    __tablename__ = "retarget_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    embodiment_id: Mapped[str] = mapped_column(ForeignKey("embodiments.id"), index=True)
    joint_traj_uri: Mapped[str | None] = mapped_column(Text)
    ee_traj_uri: Mapped[str | None] = mapped_column(Text)
    #: NULL == never sim-validated. Not the same as False (validated, failed).
    sim_validation_passed: Mapped[bool | None] = mapped_column(Boolean)
    sim_validation_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("episode_id", "embodiment_id", name="uq_episode_embodiment"),)


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    tier: Mapped[Tier | None] = mapped_column(SAEnum(Tier, name="tier"))
    manifest_uri: Mapped[str] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("customer", "name", "version", name="uq_dataset_version"),)


class Job(Base, TimestampMixin):
    """The job ledger. AWS Architecture §4.1: status lives HERE, persistently — not in a
    worker's memory, and not regex-scraped out of a log file (which is what v1's FastAPI
    service does today, and why it loses state on restart)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stage: Mapped[str] = mapped_column(String(32), index=True)
    episode_id: Mapped[str | None] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(24), index=True)
    provenance: Mapped[str | None] = mapped_column(Text)
    checkpoint_uri: Mapped[str | None] = mapped_column(Text)
    logs_uri: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)


# --------------------------------------------------------------------------------------
# Vectors (pgvector) — an addition beyond the AWS doc, flagged as such
# --------------------------------------------------------------------------------------

#: CLIP/video-embedding dimension. 512 is CLIP ViT-B/32; change requires a migration.
EMBEDDING_DIM = 512


class EpisodeEmbedding(Base, TimestampMixin):
    """Near-duplicate detection before splits are generated (L7 dedup).

    NOT in the AWS Architecture doc — added because L7 needs embedding similarity to stop
    near-duplicates leaking across train/val/test, and pgvector keeps that in the catalog
    we already run rather than adding a second datastore (FAISS) to operate.

    This is the one place the "no large arrays in the DB" rule is deliberately bent: a
    512-float vector is an index, not a blob, and the whole point is to query it.
    """

    __tablename__ = "episode_embeddings"

    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True
    )
    #: Which model produced it. Two embeddings from different models are not comparable,
    #: and a dedup that silently compares across them would drop non-duplicates.
    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
