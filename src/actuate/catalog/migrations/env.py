"""Alembic environment. URL comes from Settings — never from alembic.ini.

The initial migration is NOT hand-written. Run, against a real Postgres:

    ACTUATE_DATABASE_URL=postgresql+psycopg://... alembic revision --autogenerate -m "initial"

A hand-written migration that has never been executed against a real database is exactly
the kind of artifact that looks correct and isn't — and this schema depends on Postgres
specifics (enum types, pgvector) that only a real Postgres will validate.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from actuate.catalog.models import Base
from actuate.config import load_settings

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    from actuate.catalog.db import resolve_database_url

    return resolve_database_url(load_settings())


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    config.set_main_option("sqlalchemy.url", _url())
    # Same resume tolerance as actuate.catalog.db: Aurora at min_capacity=0 sleeps.
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 60},
    )
    with connectable.connect() as connection:
        from sqlalchemy import text

        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
