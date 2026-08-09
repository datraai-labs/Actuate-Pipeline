"""Postgres catalog (AWS Architecture §3).

Blobs in S3; everything queryable here, with an S3 URI pointer. Requires the `aws` extra.
"""

from __future__ import annotations

from actuate.catalog.db import (
    CatalogNotConfigured,
    init_schema,
    make_engine,
    make_session_factory,
    resolve_database_url,
    session_scope,
)
from actuate.catalog.models import (
    EMBEDDING_DIM,
    Annotation,
    Base,
    Capture,
    Certification,
    Consent,
    ConsentEvent,
    ConsentEventType,
    Dataset,
    Demonstrator,
    Embodiment,
    Episode,
    EpisodeEmbedding,
    Job,
    RetargetResult,
    Rig,
    Scene,
    Task,
)
from actuate.catalog.repository import (
    ConsentConflict,
    consent_history,
    deliverable_episodes,
    get_consent,
    log_consent_event,
    reconcile_consent,
    record_consent,
)

__all__ = [
    "EMBEDDING_DIM",
    "Annotation",
    "Base",
    "Capture",
    "CatalogNotConfigured",
    "Certification",
    "Consent",
    "ConsentConflict",
    "ConsentEvent",
    "ConsentEventType",
    "Dataset",
    "Demonstrator",
    "Embodiment",
    "Episode",
    "EpisodeEmbedding",
    "Job",
    "RetargetResult",
    "Rig",
    "Scene",
    "Task",
    "consent_history",
    "deliverable_episodes",
    "get_consent",
    "init_schema",
    "log_consent_event",
    "make_engine",
    "make_session_factory",
    "reconcile_consent",
    "record_consent",
    "resolve_database_url",
    "session_scope",
]
