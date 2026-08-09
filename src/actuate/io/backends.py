"""Storage backends — one interface, never raw boto3 scattered around the codebase.

AWS Architecture §2 defines four buckets whose *separation is the consent boundary*.
Every layer reads and writes through `StorageBackend`, so:

  - the delivery-write path has exactly one place to guard (see `consent.py`), and
  - the same code runs against S3 in production and a local directory in tests, which is
    what lets the abstraction be proven before any bucket exists.

`LocalBackend` mirrors the S3 bucket layout exactly — `<root>/<bucket>/<key>` — rather
than inventing a friendlier local scheme. A test that exercises a different path shape
than production is a test that has not exercised production.

boto3/s3fs are imported lazily and only by `S3Backend`, so `actuate` core installs and
runs with no AWS dependency at all (Master Spec §5, GPU/dep boundary).
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from actuate.config import Bucket, Settings, StorageBackendKind


class StorageError(RuntimeError):
    pass


class StorageBackend(ABC):
    """Blob store. Keys are POSIX-style paths within a logical bucket."""

    @abstractmethod
    def uri(self, bucket: Bucket, key: str) -> str:
        """The durable pointer stored in Postgres (AWS Architecture §3: 'blobs in S3,
        everything queryable in Postgres with an S3 URI pointer')."""

    @abstractmethod
    def exists(self, bucket: Bucket, key: str) -> bool: ...

    @abstractmethod
    def put_bytes(self, bucket: Bucket, key: str, data: bytes) -> str: ...

    @abstractmethod
    def get_bytes(self, bucket: Bucket, key: str) -> bytes: ...

    @abstractmethod
    def put_file(self, bucket: Bucket, key: str, src: Path) -> str: ...

    @abstractmethod
    def open(self, bucket: Bucket, key: str, mode: str = "rb") -> BinaryIO:
        """Streaming handle. Zarr/Parquet read through this — chunked, partial, never a
        full-episode download (AWS Architecture §2, 'Access patterns')."""

    @abstractmethod
    def list(self, bucket: Bucket, prefix: str = "") -> Iterator[str]: ...

    @abstractmethod
    def delete(self, bucket: Bucket, key: str) -> None: ...


class LocalBackend(StorageBackend):
    """Filesystem backend mirroring the S3 layout. Dev, tests, and offline work."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, bucket: Bucket, key: str) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise StorageError(f"unsafe key {key!r}")
        return self.root / bucket.value / key

    def uri(self, bucket: Bucket, key: str) -> str:
        return self._path(bucket, key).as_posix()

    def exists(self, bucket: Bucket, key: str) -> bool:
        return self._path(bucket, key).exists()

    def put_bytes(self, bucket: Bucket, key: str, data: bytes) -> str:
        p = self._path(bucket, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return self.uri(bucket, key)

    def get_bytes(self, bucket: Bucket, key: str) -> bytes:
        p = self._path(bucket, key)
        if not p.exists():
            raise StorageError(f"no such object: {self.uri(bucket, key)}")
        return p.read_bytes()

    def put_file(self, bucket: Bucket, key: str, src: Path) -> str:
        p = self._path(bucket, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, p)
        return self.uri(bucket, key)

    def open(self, bucket: Bucket, key: str, mode: str = "rb") -> BinaryIO:
        p = self._path(bucket, key)
        if "w" in mode or "a" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p.open(mode)  # type: ignore[return-value]

    def list(self, bucket: Bucket, prefix: str = "") -> Iterator[str]:
        base = self.root / bucket.value
        start = base / prefix if prefix else base
        if not start.exists():
            return
        for p in sorted(start.rglob("*")):
            if p.is_file():
                yield p.relative_to(base).as_posix()

    def delete(self, bucket: Bucket, key: str) -> None:
        self._path(bucket, key).unlink(missing_ok=True)


class S3Backend(StorageBackend):
    """S3 via boto3/s3fs. Requires the `aws` extra.

    Bucket names come from Settings (`actuate-<bucket>-<env>`), never hardcoded — so
    which AWS account this touches stays a credentials decision, not a code change.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise StorageError(
                "S3Backend needs the aws extra: pip install -e '.[aws]'"
            ) from exc
        import boto3

        session = boto3.Session(
            profile_name=settings.aws_profile, region_name=settings.aws_region
        )
        self._s3 = session.client("s3")
        self._settings = settings

    def _bucket_name(self, bucket: Bucket) -> str:
        return self._settings.bucket(bucket)

    def uri(self, bucket: Bucket, key: str) -> str:
        return f"s3://{self._bucket_name(bucket)}/{key}"

    def exists(self, bucket: Bucket, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self._bucket_name(bucket), Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "403"):
                return False
            raise

    def put_bytes(self, bucket: Bucket, key: str, data: bytes) -> str:
        self._s3.put_object(Bucket=self._bucket_name(bucket), Key=key, Body=data)
        return self.uri(bucket, key)

    def get_bytes(self, bucket: Bucket, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket_name(bucket), Key=key)
        return obj["Body"].read()

    def put_file(self, bucket: Bucket, key: str, src: Path) -> str:
        self._s3.upload_file(str(src), self._bucket_name(bucket), key)
        return self.uri(bucket, key)

    def open(self, bucket: Bucket, key: str, mode: str = "rb") -> BinaryIO:
        import s3fs

        fs = s3fs.S3FileSystem(
            profile=self._settings.aws_profile, client_kwargs={"region_name": self._settings.aws_region}
        )
        return fs.open(f"{self._bucket_name(bucket)}/{key}", mode)  # type: ignore[return-value]

    def list(self, bucket: Bucket, prefix: str = "") -> Iterator[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket_name(bucket), Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def delete(self, bucket: Bucket, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket_name(bucket), Key=key)


def get_backend(settings: Settings) -> StorageBackend:
    """The only place a backend is chosen. Defaults to LOCAL so nothing writes to AWS
    by accident — opting into S3 is deliberate."""
    if settings.storage_backend is StorageBackendKind.S3:
        return S3Backend(settings)
    return LocalBackend(settings.local_root)
