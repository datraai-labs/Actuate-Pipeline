"""The FastAPI app. Stub — Increment 1.

Every route below is a thin read over the library. None of them contain logic that cannot
be reached by importing `actuate` directly; if one ever does, the CLI/API-first rule has
been broken and the import-linter will not catch it (it only checks direction), so this
is the place to be disciplined by hand.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from actuate.config import Bucket, load_settings
from actuate.schema import SCHEMA_VERSION


def create_app() -> FastAPI:
    app = FastAPI(
        title="Actuate",
        version="2.0.0-dev",
        description="Multimodal capture -> VLA-training-ready robot data.",
    )
    settings = load_settings()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "env": settings.env.value,
            "storage_backend": settings.storage_backend.value,
        }

    @app.get("/buckets")
    def buckets() -> dict:
        return {b.value: settings.bucket(b) for b in Bucket}

    @app.get("/episodes")
    def list_episodes() -> None:
        raise HTTPException(
            501,
            "not implemented — the catalog query surface lands with the pipeline layers "
            "(Increment 2+). Increment 1 builds schema, io, catalog, and infra only.",
        )

    @app.post("/episodes/{episode_id}/export")
    def export(episode_id: str) -> None:
        raise HTTPException(
            501,
            "not implemented — the LeRobot v3 / RLDS exporters are Increment 2, and are "
            "not 'done' until LeRobot's own loader reads the output and a real training "
            "step runs (Master Spec §L7).",
        )

    return app


app = create_app()
