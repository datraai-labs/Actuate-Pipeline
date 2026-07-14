"""The IAM half of the consent boundary — ATTEMPTED against real AWS, not asserted.

AWS Architecture §2 promises:

    "the delivery bucket's policy denies writes from any principal except the packaging
     role, and the raw/work roles have no PutObject on delivery. So a bug in app code
     cannot leak un-consented data into the customer-facing bucket — IAM stops it too."

Checking that the policy *exists* proves nothing. A policy can exist and be wrong: scoped
to the wrong ARN, shadowed by an Allow, or attached to the wrong bucket. The only evidence
that counts is a real `PutObject` that is really refused.

So these tests do both directions against the live bucket:

  - A principal that is NOT the packaging role attempts a write and MUST be denied. The
    principal used is the current caller — in practice an **administrator**. That makes it
    the strongest possible form of the test: an explicit `Deny` in a bucket policy cannot
    be overridden by *any* Allow, so even full admin cannot write there.
  - The packaging role assumes, writes, and MUST succeed. Without this half we would only
    have proven the bucket rejects everything, which is not a boundary — it is a brick.

Requires real AWS credentials for the Actuate account. Skipped otherwise, and reported as
skipped rather than counted as passed.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.needs_aws

BUCKET = os.environ.get("ACTUATE_DELIVERY_BUCKET", "actuate-delivery-dev")
REGION = os.environ.get("AWS_REGION", "eu-north-1")
PROFILE = os.environ.get("AWS_PROFILE_ACTUATE", "datraai-admin")
ACTUATE_ACCOUNT = os.environ.get("ACTUATE_AWS_ACCOUNT", "")

PAYLOAD = b"if this lands in the delivery bucket, un-consented data can reach a customer"


@pytest.fixture(scope="module")
def session():
    boto3 = pytest.importorskip("boto3")
    try:
        s = boto3.Session(profile_name=PROFILE, region_name=REGION)
        ident = s.client("sts").get_caller_identity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no AWS credentials for profile {PROFILE!r}: {exc}")

    if not ACTUATE_ACCOUNT:
        pytest.skip(
            "ACTUATE_AWS_ACCOUNT is not set. These tests write to real buckets, so the "
            "target account must be named explicitly -- never inferred from ambient "
            "credentials, which on a dev machine may belong to a PARTNER account."
        )
    if ident["Account"] != ACTUATE_ACCOUNT:
        pytest.skip(
            f"credentials point at account {ident['Account']}, not the one named in "
            "ACTUATE_AWS_ACCOUNT. Refusing to run boundary tests against another account."
        )
    return s


@pytest.fixture
def key() -> str:
    return f"_boundary_probe/{uuid.uuid4().hex}.bin"


def _packaging_role_arn(session) -> str:
    return f"arn:aws:iam::{ACTUATE_ACCOUNT}:role/actuate-packaging-dev"


# --- the negative half: even an ADMIN cannot write to delivery ---------------------------


def test_a_non_packaging_principal_is_DENIED_a_real_putobject(session, key):
    """Attempt a real write with the current (administrator) identity. It must fail.

    This is the assertion that matters. An explicit Deny in a bucket policy beats every
    Allow in IAM, so if this write succeeds, the consent boundary does not exist — no
    matter what the policy document says.
    """
    from botocore.exceptions import ClientError

    s3 = session.client("s3")
    caller = session.client("sts").get_caller_identity()["Arn"]
    assert "actuate-packaging" not in caller, "this test must NOT run as the packaging role"

    with pytest.raises(ClientError) as exc:
        s3.put_object(Bucket=BUCKET, Key=key, Body=PAYLOAD)

    code = exc.value.response["Error"]["Code"]
    assert code in ("AccessDenied", "AccessDeniedException"), (
        f"expected AccessDenied, got {code}. The delivery bucket accepted a write from "
        f"{caller} — the IAM consent boundary is NOT enforced."
    )

    # And nothing landed.
    with pytest.raises(ClientError) as head:
        s3.head_object(Bucket=BUCKET, Key=key)
    assert head.value.response["Error"]["Code"] in ("404", "403")


def test_the_deny_beats_administrator_privilege(session, key):
    """Spell out what the previous test proves, because it is easy to under-read.

    The caller is an administrator with `s3:*` on everything. It is still refused. That is
    the property we want: no future over-broad grant to any role can open a write path to
    the delivery bucket, because an explicit Deny cannot be overridden.
    """
    from botocore.exceptions import ClientError

    iam = session.client("iam")
    user = session.client("sts").get_caller_identity()["Arn"].rsplit("/", 1)[-1]
    try:
        policies = [
            p["PolicyName"]
            for p in iam.list_attached_user_policies(UserName=user)["AttachedPolicies"]
        ]
    except ClientError:
        policies = []

    is_admin = "AdministratorAccess" in policies
    if not is_admin:
        pytest.skip(f"caller {user} is not AdministratorAccess; nothing extra to prove")

    with pytest.raises(ClientError):
        session.client("s3").put_object(Bucket=BUCKET, Key=key, Body=PAYLOAD)


# --- the positive half: the packaging role CAN write -------------------------------------


def test_the_packaging_role_CAN_write_to_delivery(session, key):
    """Without this, "the bucket rejects everything" would look like a working boundary.

    A boundary that nothing can cross is not a boundary; it is an outage. The packaging
    role must actually be able to deliver.
    """
    from botocore.exceptions import ClientError

    sts = session.client("sts")
    try:
        creds = sts.assume_role(
            RoleArn=_packaging_role_arn(session),
            RoleSessionName="actuate-boundary-proof",
        )["Credentials"]
    except ClientError as exc:
        pytest.skip(
            f"cannot assume the packaging role ({exc.response['Error']['Code']}). "
            "In dev its trust policy includes the account root; redeploy StorageStack."
        )

    import boto3

    packaging = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )

    packaging.put_object(Bucket=BUCKET, Key=key, Body=PAYLOAD)
    got = packaging.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    assert got == PAYLOAD, "the packaging role could not write to the delivery bucket"

    packaging.delete_object(Bucket=BUCKET, Key=key)


# --- and the raw/work buckets are NOT similarly locked ------------------------------------


def test_the_pipeline_can_still_write_to_work(session, key):
    """Sanity: the Deny is scoped to delivery, not sprayed across every bucket.

    If this failed, the boundary would be indiscriminate and the pipeline could not
    function — which is a different bug that would look, from the delivery bucket alone,
    exactly like success.
    """
    s3 = session.client("s3")
    s3.put_object(Bucket="actuate-work-dev", Key=key, Body=b"intermediate artifact")
    assert s3.get_object(Bucket="actuate-work-dev", Key=key)["Body"].read()
    s3.delete_object(Bucket="actuate-work-dev", Key=key)
