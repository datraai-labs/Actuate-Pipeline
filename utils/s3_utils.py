"""
DatraAI Pipeline — S3 Utilities
boto3 upload, presigned URL generation, and credential validation.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None  # type: ignore


def _get_client(region: Optional[str] = None):
    """Get an S3 client, raising clear errors if boto3 or credentials are missing."""
    if boto3 is None:
        raise ImportError(
            "boto3 is not installed. Install with: pip install boto3"
        )

    # Import config lazily to avoid circular imports
    import config as cfg

    region = region or cfg.AWS_REGION

    try:
        client = boto3.client("s3", region_name=region)
        return client
    except NoCredentialsError:
        print(
            "[s3_utils] ERROR: AWS credentials not configured.\n"
            "  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars,\n"
            "  or configure ~/.aws/credentials\n"
            "  See: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html",
            file=sys.stderr,
        )
        raise


def check_credentials() -> bool:
    """
    Test boto3 connection by calling STS GetCallerIdentity.
    Returns True if credentials are valid, False otherwise.
    Prints helpful error message on failure.
    """
    if boto3 is None:
        print("[s3_utils] boto3 is not installed.", file=sys.stderr)
        return False

    # Import config lazily to avoid circular imports; must match the region
    # used by _get_client()/uploads, or this check can pass/fail
    # inconsistently with the actual upload path in region-restricted
    # accounts.
    import config as cfg

    try:
        sts = boto3.client("sts", region_name=cfg.AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"[s3_utils] AWS credentials valid. Account: {identity['Account']}")
        return True
    except NoCredentialsError:
        print(
            "[s3_utils] No AWS credentials found.\n"
            "  Set environment variables:\n"
            "    export AWS_ACCESS_KEY_ID=your_key\n"
            "    export AWS_SECRET_ACCESS_KEY=your_secret\n"
            "  Or configure: aws configure",
            file=sys.stderr,
        )
        return False
    except ClientError as e:
        print(f"[s3_utils] AWS credential check failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[s3_utils] Unexpected error checking credentials: {e}", file=sys.stderr)
        return False


def _sanitize_s3_prefix(prefix: str) -> str:
    """Strip path-traversal segments from a caller-supplied S3 key prefix."""
    parts = [p for p in prefix.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts)


def upload_folder(
    local_path: Path,
    bucket: Optional[str] = None,
    s3_prefix: str = "",
    region: Optional[str] = None,
) -> int:
    """
    Upload all files in local_path (recursively) to S3.

    Args:
        local_path: Local directory to upload.
        bucket: S3 bucket name. Defaults to config.S3_BUCKET.
        s3_prefix: S3 key prefix (e.g., "deliveries/batch_001/").
        region: AWS region override.

    Returns:
        Number of files uploaded.
    """
    import config as cfg

    bucket = bucket or cfg.S3_BUCKET
    client = _get_client(region)

    local_path = Path(local_path)
    if not local_path.is_dir():
        raise NotADirectoryError(f"Local path is not a directory: {local_path}")

    safe_prefix = _sanitize_s3_prefix(s3_prefix)

    uploaded = 0
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_path)
            s3_key = f"{safe_prefix}/{relative}".replace("\\", "/").lstrip("/")

            try:
                client.upload_file(
                    str(file_path),
                    bucket,
                    s3_key,
                    ExtraArgs={"ServerSideEncryption": "AES256"},
                )
                print(f"[s3_utils] Uploaded: s3://{bucket}/{s3_key}")
                uploaded += 1
            except ClientError as e:
                print(
                    f"[s3_utils] ERROR uploading {file_path}: {e}",
                    file=sys.stderr,
                )
                raise

    print(f"[s3_utils] Upload complete: {uploaded} files to s3://{bucket}/{safe_prefix}")
    return uploaded


def generate_presigned_url(
    bucket: Optional[str] = None,
    key: str = "",
    expiry_seconds: int = 604800,
    region: Optional[str] = None,
) -> str:
    """
    Generate a presigned URL for an S3 object.

    Args:
        bucket: S3 bucket name. Defaults to config.S3_BUCKET.
        key: S3 object key.
        expiry_seconds: URL expiry in seconds (default 7 days).
        region: AWS region override.

    Returns:
        Presigned URL string.
    """
    import config as cfg

    bucket = bucket or cfg.S3_BUCKET
    client = _get_client(region)

    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )
        return url
    except ClientError as e:
        print(f"[s3_utils] ERROR generating presigned URL: {e}", file=sys.stderr)
        raise
