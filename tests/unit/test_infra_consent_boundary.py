"""The IAM half of the consent boundary, asserted against the synthesized CloudFormation.

AWS Architecture §2 promises that *"a bug in app code cannot leak un-consented data into
the customer-facing bucket — IAM stops it too."* That promise is only real if the template
actually contains the Deny. These tests read the synthesized template and check.

`io/consent.py` is the code half, and `tests/unit/test_consent_guard.py` proves it is
load-bearing by removing it and watching data leak. Both halves, because either alone is a
single point of failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[2] / "infra"

aws_cdk = pytest.importorskip("aws_cdk", reason="aws-cdk-lib not installed")


@pytest.fixture(scope="module")
def template() -> dict:
    """Synthesize StorageStack in-process. No AWS calls, no credentials."""
    sys.path.insert(0, str(INFRA))
    from aws_cdk import App
    from aws_cdk.assertions import Template
    from stacks.storage_stack import StorageStack

    app = App(context={"env": "dev"})
    stack = StorageStack(app, "TestStorage", env_name="dev")
    return Template.from_stack(stack).to_json()


def _buckets(template: dict) -> dict[str, dict]:
    return {
        v["Properties"]["BucketName"]: v["Properties"]
        for v in template["Resources"].values()
        if v["Type"] == "AWS::S3::Bucket"
    }


def test_the_four_buckets_exist(template):
    names = set(_buckets(template))
    assert {
        "actuate-raw-dev",
        "actuate-work-dev",
        "actuate-delivery-dev",
        "actuate-artifacts-dev",
    } <= names


def test_every_data_bucket_blocks_public_access_and_uses_sse_kms(template):
    for name, props in _buckets(template).items():
        if name == "actuate-access-logs-dev":
            continue  # S3 log delivery cannot write to a CMK-encrypted bucket
        pab = props["PublicAccessBlockConfiguration"]
        assert all(pab.values()), f"{name}: Block Public Access is not fully ON"

        algo = props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
            "ServerSideEncryptionByDefault"
        ]["SSEAlgorithm"]
        assert algo == "aws:kms", f"{name}: encryption is {algo}, expected aws:kms"


def test_raw_and_delivery_are_versioned(template):
    b = _buckets(template)
    for name in ("actuate-raw-dev", "actuate-delivery-dev"):
        assert b[name]["VersioningConfiguration"]["Status"] == "Enabled", f"{name} not versioned"


def test_raw_lifecycles_to_deep_archive(template):
    """Video dominates cost (AWS Architecture §1, §7)."""
    rules = _buckets(template)["actuate-raw-dev"]["LifecycleConfiguration"]["Rules"]
    classes = {
        t["StorageClass"] for r in rules for t in r.get("Transitions", [])
    }
    assert "DEEP_ARCHIVE" in classes


# --- THE CONSENT BOUNDARY -----------------------------------------------------------------


def _delivery_policy_statements(template: dict) -> list[dict]:
    for v in template["Resources"].values():
        if v["Type"] != "AWS::S3::BucketPolicy":
            continue
        if "Delivery" not in json.dumps(v["Properties"]["Bucket"]):
            continue
        return v["Properties"]["PolicyDocument"]["Statement"]
    pytest.fail("the delivery bucket has NO bucket policy at all")


def test_delivery_bucket_explicitly_denies_writes_from_non_packaging_principals(template):
    """The safety-critical assertion.

    An explicit Deny in IAM cannot be overridden by any Allow, anywhere — so even if
    someone later attaches an over-broad policy to another role, writes to the delivery
    bucket still fail unless the caller IS the packaging role.
    """
    denies = [
        s
        for s in _delivery_policy_statements(template)
        if s["Effect"] == "Deny" and s.get("Sid") == "DenyDeliveryWritesExceptPackagingRole"
    ]
    assert denies, "no explicit Deny on delivery writes — the consent boundary is NOT enforced"

    stmt = denies[0]
    actions = stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]]
    assert "s3:PutObject" in actions
    assert "s3:DeleteObject" in actions

    cond = stmt["Condition"]
    assert "ArnNotEquals" in cond, (
        "the Deny is unconditional or conditioned on the wrong thing — it must apply to "
        "every principal EXCEPT the packaging role"
    )
    assert "aws:PrincipalArn" in cond["ArnNotEquals"]


def test_delivery_bucket_enforces_tls(template):
    denies = [
        s
        for s in _delivery_policy_statements(template)
        if s["Effect"] == "Deny"
        and s.get("Condition", {}).get("Bool", {}).get("aws:SecureTransport") == "false"
    ]
    assert denies, "delivery bucket does not deny non-TLS access"


def test_the_pipeline_role_is_never_granted_putobject_on_delivery(template):
    """The pipeline role processes raw/work. It must have no write path to the customer
    bucket at all — so that a bug in pipeline code cannot reach it even to try."""
    for v in template["Resources"].values():
        if v["Type"] != "AWS::IAM::Policy":
            continue
        roles = json.dumps(v["Properties"].get("Roles", []))
        if "PipelineRole" not in roles:
            continue
        doc = json.dumps(v["Properties"]["PolicyDocument"])
        if "Delivery" not in doc:
            continue
        for stmt in v["Properties"]["PolicyDocument"]["Statement"]:
            if stmt["Effect"] != "Allow":
                continue
            resources = json.dumps(stmt.get("Resource", []))
            if "Delivery" not in resources:
                continue
            actions = stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]]
            writes = [a for a in actions if "Put" in a or "Delete" in a]
            assert not writes, (
                f"the pipeline role is granted {writes} on the delivery bucket. It must "
                "have read access at most — writing there is the packaging role's job alone."
            )
