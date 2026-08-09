"""Storage I/O — the single interface every layer reads and writes through.

Never raw boto3 in a layer. This package owns:
  - `backends`  — LocalBackend / S3Backend against the 4 buckets (AWS Architecture §2)
  - `store`     — canonical episode <-> Parquet + Zarr (Master Spec §L3)
  - `consent`   — the FAIL-CLOSED delivery write guard (AWS Architecture §2; Master Spec §L4)
  - `geometry`  — SE(3) helpers

Writing to `Bucket.DELIVERY` goes through `DeliveryWriter` and nothing else.
"""

from __future__ import annotations

from actuate.io.backends import (
    LocalBackend,
    S3Backend,
    StorageBackend,
    StorageError,
    get_backend,
)
from actuate.io.consent import (
    ConsentViolation,
    DeliveryWriter,
    check_deliverable,
    check_episode_deliverable,
)
from actuate.io.geometry import hand_frame_quaternion


def __getattr__(name: str):
    """Load the Parquet/Zarr store only when one of its entry points is used.

    ``pyarrow`` belongs to the storage extra. Importing ``actuate.io.geometry`` (and thus
    running ``actuate --help``) must not make it a core dependency.
    """
    if name in {"canonical_prefix", "read_episode", "write_episode"}:
        from actuate.io import store

        return getattr(store, name)
    raise AttributeError(name)

__all__ = [
    "ConsentViolation",
    "DeliveryWriter",
    "LocalBackend",
    "S3Backend",
    "StorageBackend",
    "StorageError",
    "canonical_prefix",
    "check_deliverable",
    "check_episode_deliverable",
    "get_backend",
    "hand_frame_quaternion",
    "read_episode",
    "write_episode",
]
