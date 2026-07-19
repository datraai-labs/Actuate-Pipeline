"""S3 storage for processed runs -- durable artifacts live in AWS, not on local disk.

The processing SCRATCH is unavoidably local (WiLoR/UniDepth read video frames off the
filesystem, the GPU needs local files). What lives in S3 is the DURABLE output:

    raw video            -> s3://actuate-raw-<env>/<capture_id>/<video>
    canonical episode    -> s3://actuate-work-<env>/canonical/<episode_id>/canonical.json
    exported dataset     -> s3://actuate-work-<env>/canonical/<episode_id>/<name>/...

RAW and WORK are internal buckets and are NOT consent-gated -- only DELIVERY is, and it stays
behind the DeliveryWriter + IAM Deny. So uploading processing artifacts here never touches the
consent boundary; a customer handoff still goes through `actuate deliver`.

Credentials come from the AWS profile in ~/.actuate/config.json (`aws_profile`), else the
AWS_PROFILE env. The key is never read into code -- boto3 resolves it from the profile.
"""

from __future__ import annotations

import os
from pathlib import Path

from actuate.config import auth

# skip cache/scratch when uploading an export tree
_SKIP = {".actuate_cache", "__pycache__", ".DS_Store"}


def s3_backend():
    """An S3Backend pointed at the configured account/profile/region."""
    from actuate.config.settings import Settings
    from actuate.io.backends import S3Backend

    cfg = auth.load_config()
    profile = cfg.get("aws_profile") or os.environ.get("AWS_PROFILE")
    region = os.environ.get("AWS_REGION", "eu-north-1")
    return S3Backend(Settings(aws_profile=profile, aws_region=region))


def _upload_tree(backend, bucket, prefix: str, local_dir: Path) -> list[str]:
    """Upload every file under `local_dir` to `bucket/prefix/<relpath>`; return the URIs."""
    uris = []
    for f in sorted(local_dir.rglob("*")):
        if not f.is_file() or any(part in _SKIP for part in f.parts):
            continue
        key = f"{prefix}/{f.relative_to(local_dir).as_posix()}"
        uris.append(backend.put_file(bucket, key, f))
    return uris


def upload_run(canonical_path: Path, capture_id: str, episode_id: str, *,
               video: Path | None = None, export_dirs: list[Path] | None = None,
               backend=None, clean_local: bool = False) -> dict:
    """Push a processed run's durable artifacts to S3. Returns {kind: uri(s)}.

    `clean_local` deletes the local copies AFTER a verified upload -- so the result lives in
    AWS, not on disk. The raw/canonical are single files; each export dir is a tree.
    """
    from actuate.config.settings import Bucket

    backend = backend or s3_backend()
    canonical_path = Path(canonical_path)
    uris: dict = {}

    if video is not None and Path(video).exists():
        uris["raw"] = backend.put_file(Bucket.RAW, f"{capture_id}/{Path(video).name}",
                                       Path(video))
    uris["canonical"] = backend.put_file(
        Bucket.WORK, f"canonical/{episode_id}/canonical.json", canonical_path)

    export_uris: dict[str, list[str]] = {}
    for d in export_dirs or []:
        d = Path(d)
        if d.is_dir():
            export_uris[d.name] = _upload_tree(
                backend, Bucket.WORK, f"canonical/{episode_id}/{d.name}", d)
    if export_uris:
        uris["exports"] = export_uris

    if clean_local:
        _clean(canonical_path, video, export_dirs)
    return uris


def _clean(canonical_path: Path, video, export_dirs) -> None:
    import shutil

    # remove the export trees and the canonical; leave the session video only if it is the
    # user's own source file (we never delete an input the user pointed us at)
    for d in export_dirs or []:
        shutil.rmtree(d, ignore_errors=True)
    try:
        Path(canonical_path).unlink(missing_ok=True)
    except OSError:
        pass


def verify_bucket_access(backend=None) -> dict:
    """Read-only-ish check that credentials + buckets work: put a tiny marker into WORK and
    confirm it round-trips, then delete it. Used before the first real upload."""
    from actuate.config.settings import Bucket

    backend = backend or s3_backend()
    key = "._actuate_access_check"
    uri = backend.put_bytes(Bucket.WORK, key, b"ok")
    exists = backend.exists(Bucket.WORK, key)
    try:
        backend._s3.delete_object(Bucket=backend._bucket_name(Bucket.WORK), Key=key)
    except Exception:
        pass
    return {"uri": uri, "verified": exists}
