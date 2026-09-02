"""Typed, validated settings — Master Spec §2.3, AWS Architecture §4.3.

Replaces v1's `config.py` module-level constants. Settings here are an object you pass,
not global state you reassign, so the separate dashboard API can remain multi-tenant.

**No secrets in code, ever.** DB credentials come from AWS Secrets Manager (or an env var
in local dev). There is no default password and no default connection string; absence is
an error, not something to paper over.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(str, Enum):
    DEV = "dev"
    PROD = "prod"


class StorageBackendKind(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class Bucket(str, Enum):
    """The four buckets of AWS Architecture §2. The separation IS the consent boundary."""

    RAW = "raw"
    WORK = "work"
    DELIVERY = "delivery"
    ARTIFACTS = "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACTUATE_",
        # `.env.local` holds the developer's real DB URL / AWS profile and takes
        # precedence over a committed `.env`. Real process env still wins over both.
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Env = Env.DEV

    # --- storage -----------------------------------------------------------------
    #: LOCAL is the default so that nothing accidentally writes to AWS. Opting into S3
    #: is deliberate.
    storage_backend: StorageBackendKind = StorageBackendKind.LOCAL

    #: Root for the local backend. Mirrors the S3 bucket layout exactly, so the same
    #: code paths are exercised in tests as in production.
    local_root: Path = Path("./.actuate-store")

    aws_region: str = "eu-north-1"
    #: DatraAI's own account. See docs/architecture/AWS_ARCHITECTURE.md and note that
    #: a developer machine may have partner-account credentials configured by default.
    aws_profile: str | None = None
    aws_account_id: str | None = None

    bucket_prefix: str = "actuate"

    # --- catalog (Postgres) ------------------------------------------------------
    #: Full SQLAlchemy URL. In AWS this is assembled from Secrets Manager; locally it
    #: is set via ACTUATE_DATABASE_URL. There is deliberately NO default — a silent
    #: fallback to sqlite would make the catalog's guarantees untestable.
    database_url: SecretStr | None = None
    db_secret_arn: str | None = None

    #: Local-only escape hatch. A private RDS is unreachable from a laptop except through
    #: an SSM tunnel, but the secret's `host` is the private cluster endpoint. This
    #: overrides host[:port] (e.g. "127.0.0.1:15432") so `db_secret_arn` can be used from a
    #: developer machine without ever copying the password out of Secrets Manager. Unset in
    #: prod, where in-VPC compute reaches the endpoint directly.
    db_endpoint_override: str | None = None

    # --- gates -------------------------------------------------------------------
    #: Exists to be read, never to be set True in a config file. See the validator.
    #: The Master Spec §L4 names the "auto-consent workaround" as a bug class to test
    #: against; this field is here so that any attempt to add such a bypass has to go
    #: through a validator that refuses it, rather than being quietly added later.
    allow_unconsented_delivery: bool = False

    @model_validator(mode="after")
    def _no_consent_bypass(self) -> Settings:
        if self.allow_unconsented_delivery:
            raise ValueError(
                "allow_unconsented_delivery=True is not a supported configuration. "
                "The consent/PII gate is fail-closed and hard-blocking (Master Spec §L4; "
                "AWS Architecture §2). If you are trying to deliver an episode, fix its "
                "consent record — do not disable the gate."
            )
        return self

    def bucket(self, which: Bucket) -> str:
        """`actuate-raw-dev`, `actuate-delivery-prod`, ... (AWS Architecture §2)."""
        return f"{self.bucket_prefix}-{which.value}-{self.env.value}"


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
