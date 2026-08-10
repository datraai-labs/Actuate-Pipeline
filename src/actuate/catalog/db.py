"""Engine + session for the Postgres catalog.

Credentials come from Settings, which sources them from AWS Secrets Manager (or an env
var in local dev). **There is no default connection string.** A silent fallback to SQLite
would be worse than an error: the catalog's guarantees — enum constraints, the consent
foreign key, pgvector — do not all exist in SQLite, so a test that "passed" against it
would prove nothing about production. Fail loudly instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from actuate.catalog.models import Base
from actuate.config import Settings


class CatalogNotConfigured(RuntimeError):
    pass


def resolve_database_url(settings: Settings) -> str:
    """Explicit URL, or fetch from Secrets Manager. Never a default."""
    if settings.database_url is not None:
        return settings.database_url.get_secret_value()

    if settings.db_secret_arn:
        import json
        from urllib.parse import quote

        import boto3

        session = boto3.Session(
            profile_name=settings.aws_profile, region_name=settings.aws_region
        )
        secret = session.client("secretsmanager").get_secret_value(
            SecretId=settings.db_secret_arn
        )
        s = json.loads(secret["SecretString"])
        host, port = s["host"], s.get("port", 5432)
        # Local escape hatch: reach the private endpoint through an SSM tunnel without
        # copying the password out of Secrets Manager. Unset in prod (see Settings).
        if settings.db_endpoint_override:
            h, _, p = settings.db_endpoint_override.partition(":")
            host = h or host
            port = p or port
        # Encode credentials: a generated password can contain characters (`:`, `%`, ...)
        # that would otherwise corrupt the URL. quote() with an empty safe set is exact.
        user = quote(s["username"], safe="")
        pw = quote(s["password"], safe="")
        return f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{s.get('dbname', 'actuate')}"

    raise CatalogNotConfigured(
        "no database configured. Set ACTUATE_DATABASE_URL, or ACTUATE_DB_SECRET_ARN to "
        "read credentials from AWS Secrets Manager. There is deliberately no default — "
        "a fallback datastore would not enforce the consent constraints this catalog exists for."
    )


#: Aurora Serverless v2 runs at min_capacity=0 in dev, which means it AUTO-PAUSES when
#: idle and takes ~15-30s to resume. psycopg's default connect timeout gives up long before
#: that, so the first query after an idle period fails with a bare ConnectionTimeout that
#: looks like a networking fault and isn't. A generous connect timeout is not a workaround
#: here; it is the correct setting for a database that is allowed to sleep.
_RESUME_TIMEOUT_SEC = 60


def make_engine(settings: Settings, echo: bool = False) -> Engine:
    return create_engine(
        resolve_database_url(settings),
        echo=echo,
        # Recycle stale connections silently: a paused cluster drops them.
        pool_pre_ping=True,
        connect_args={"connect_timeout": _RESUME_TIMEOUT_SEC},
    )


def init_schema(engine: Engine) -> None:
    """Create the pgvector extension and all tables.

    Alembic owns migrations in production (see catalog/migrations/); this exists for
    tests and first-time local bring-up.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
