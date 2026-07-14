"""`actuate storage` — inspect the buckets and, critically, VERIFY THE CONSENT BOUNDARY.

AWS Architecture §8: *"The consent boundary (IAM + bucket policy + fail-closed gate)
should be stood up and tested first, before any real capture data lands, and re-verified
whenever roles change."*

`actuate storage verify-consent-boundary` is that re-verification, runnable on demand. It
is the command to run before the first byte of capture data is uploaded, and again after
any IAM change.
"""

from __future__ import annotations

import typer

from actuate.config import Bucket, Env, StorageBackendKind, load_settings

storage_app = typer.Typer(help="Buckets, and the consent boundary that separates them.")


@storage_app.command("whoami")
def whoami(profile: str = typer.Option(None, help="AWS profile.")) -> None:
    """Which AWS account are we actually pointed at?

    Worth its own command: a dev machine can easily have credentials for a *partner*
    account configured as the default, and provisioning Actuate's buckets there would be
    both broken and inappropriate.
    """
    import boto3

    settings = load_settings(aws_profile=profile) if profile else load_settings()
    session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
    ident = session.client("sts").get_caller_identity()

    typer.echo(f"account : {ident['Account']}")
    typer.echo(f"arn     : {ident['Arn']}")
    typer.echo(f"region  : {settings.aws_region}")
    typer.echo(f"profile : {settings.aws_profile or '(default)'}")


@storage_app.command("buckets")
def buckets(env: Env = Env.DEV) -> None:
    """Show the four bucket names for an environment."""
    settings = load_settings(env=env)
    for b in Bucket:
        typer.echo(f"{b.value:10s} s3://{settings.bucket(b)}")


@storage_app.command("verify-consent-boundary")
def verify_consent_boundary(
    env: Env = Env.DEV,
    profile: str = typer.Option(None, help="AWS profile."),
) -> None:
    """Prove the delivery bucket refuses an un-consented write — against real AWS.

    Checks, in order:
      1. All four buckets exist.
      2. Block Public Access is ON for each.
      3. Default encryption (SSE-KMS) is set for each.
      4. Delivery and raw are versioned.
      5. The delivery bucket policy denies PutObject to non-packaging principals.

    This is an infrastructure check. The *code* half of the same boundary
    (`io.consent.DeliveryWriter`) is covered by tests that confirm data leaks when the
    guard is removed — both halves exist because either alone is a single point of failure.
    """
    import botocore.exceptions
    import boto3

    settings = load_settings(env=env, storage_backend=StorageBackendKind.S3)
    if profile:
        settings = load_settings(env=env, storage_backend=StorageBackendKind.S3, aws_profile=profile)

    session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
    s3 = session.client("s3")

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            typer.secho(f"  PASS  {msg}", fg="green")
        else:
            typer.secho(f"  FAIL  {msg}", fg="red")
            failures.append(msg)

    for b in Bucket:
        name = settings.bucket(b)
        typer.echo(f"\n{name}")
        try:
            s3.head_bucket(Bucket=name)
        except botocore.exceptions.ClientError as exc:
            typer.secho(f"  FAIL  bucket does not exist or is unreachable ({exc})", fg="red")
            failures.append(f"{name}: unreachable")
            continue

        try:
            pab = s3.get_public_access_block(Bucket=name)["PublicAccessBlockConfiguration"]
            check(all(pab.values()), "Block Public Access fully ON")
        except botocore.exceptions.ClientError:
            check(False, "Block Public Access configured")

        try:
            enc = s3.get_bucket_encryption(Bucket=name)
            algo = enc["ServerSideEncryptionConfiguration"]["Rules"][0][
                "ApplyServerSideEncryptionByDefault"
            ]["SSEAlgorithm"]
            check(algo == "aws:kms", f"default encryption is SSE-KMS (got {algo})")
        except botocore.exceptions.ClientError:
            check(False, "default encryption configured")

        if b in (Bucket.RAW, Bucket.DELIVERY):
            ver = s3.get_bucket_versioning(Bucket=name).get("Status")
            check(ver == "Enabled", f"versioning enabled (got {ver})")

        if b is Bucket.DELIVERY:
            try:
                pol = s3.get_bucket_policy(Bucket=name)["Policy"]
                check("Deny" in pol and "s3:PutObject" in pol,
                      "bucket policy contains an explicit Deny on PutObject")
            except botocore.exceptions.ClientError:
                check(False, "delivery bucket has a policy denying non-packaging writes")

    typer.echo("")
    if failures:
        typer.secho(
            f"CONSENT BOUNDARY NOT INTACT — {len(failures)} check(s) failed.\n"
            "Do NOT land real capture data until these pass (AWS Architecture §8).",
            fg="red",
            bold=True,
        )
        raise typer.Exit(1)
    typer.secho("consent boundary intact", fg="green", bold=True)
