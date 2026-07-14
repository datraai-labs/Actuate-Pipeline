#!/usr/bin/env python3
"""Actuate CDK app — StorageStack + DataStack (Increment 1).

AWS Architecture §6 lists four stacks; ComputeStack (Batch + Step Functions) and
ServiceStack (Fargate + ALB) are later increments. §8 is explicit about the order:

    S3 + KMS + IAM (consent boundary) -> RDS/Postgres catalog -> containerize -> compute
    -> API -> dashboard

    "The consent boundary should be stood up and tested FIRST, before any real capture
     data lands."

Usage:
    cdk synth  -c env=dev
    cdk deploy -c env=dev --profile datraai-admin --all

No account ID is hardcoded. The target account comes from the credentials you deploy
with, which is deliberate — a dev machine may have credentials for a *partner* account
configured as its default, and Actuate infra must never be provisioned there.
"""

from __future__ import annotations

import os

import aws_cdk as cdk

from stacks.budget_stack import BudgetStack
from stacks.data_stack import DataStack
from stacks.storage_stack import StorageStack

app = cdk.App()

env_name = app.node.try_get_context("env") or "dev"
if env_name not in ("dev", "prod"):
    raise SystemExit(f"env must be 'dev' or 'prod', got {env_name!r}")

# Cost guardrail. Overridable per-env via context, e.g. -c budget_usd=100
budget_usd = float(app.node.try_get_context("budget_usd") or 50)
alert_email = app.node.try_get_context("alert_email") or "pushkar@datraai.com"

# Environment-agnostic UNLESS an account is EXPLICITLY named in ACTUATE_AWS_ACCOUNT.
#
# Note we deliberately do NOT read CDK_DEFAULT_ACCOUNT. The CDK CLI injects that variable
# automatically from whatever credentials are ambient — and on a dev machine here, those
# belong to a *partner* AWS account (a `vendor-upload-only` user in someone else's org).
# Trusting CDK_DEFAULT_ACCOUNT would silently bind these stacks to that account and make
# `cdk synth` perform live AZ lookups against it.
#
# Requiring an explicit variable makes the target account a decision, never an accident:
#
#     ACTUATE_AWS_ACCOUNT=<your-account-id> AWS_REGION=eu-north-1 \
#         cdk deploy -c env=dev --profile datraai-admin --all
#
# With it unset, `cdk synth` is a pure offline operation and cannot touch any account.
account = os.environ.get("ACTUATE_AWS_ACCOUNT")
region = os.environ.get("AWS_REGION", "eu-north-1")
env = cdk.Environment(account=account, region=region) if account else None

# Deployed FIRST. Costs nothing, and it is the only resource here that can tell you the
# others have gone wrong before the invoice does.
#
# PINNED TO us-east-1. AWS Budgets is a global service whose CloudFormation resource type
# only exists in us-east-1 — deploying it to eu-north-1 fails with "Unrecognized resource
# types: [AWS::Budgets::Budget]". Billing data is account-wide regardless of the region the
# budget resource lives in, so this watches eu-north-1 spend perfectly well from us-east-1.
budget_env = (
    cdk.Environment(account=account, region="us-east-1") if account else None
)
BudgetStack(
    app, f"Actuate-Budget-{env_name}", env_name=env_name, env=budget_env,
    monthly_limit_usd=budget_usd, alert_email=alert_email,
    description=f"Actuate cost guardrail: ${budget_usd:.0f}/mo, alerts to {alert_email}.",
)

storage = StorageStack(
    app, f"Actuate-Storage-{env_name}", env_name=env_name, env=env,
    description="Actuate S3 buckets, KMS CMK, and the IAM consent boundary.",
)

# The catalog sits in private isolated subnets with no ingress -- correct, and also
# unreachable from a laptop. `-c bastion=true` adds an SSM bastion INSIDE this stack
# (see stacks/bastion.py): no inbound ports open, the DB stays private. Off by default;
# it is the only always-on compute in this environment.
with_bastion = app.node.try_get_context("bastion") in ("true", "1", True)

data = DataStack(
    app, f"Actuate-Data-{env_name}", env_name=env_name, key=storage.key, env=env,
    with_bastion=with_bastion,
    description="Actuate Postgres catalog (Aurora Serverless v2) + Secrets Manager.",
)

cdk.Tags.of(app).add("project", "actuate")
cdk.Tags.of(app).add("env", env_name)

app.synth()
