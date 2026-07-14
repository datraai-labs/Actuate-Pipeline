"""StorageStack — the four buckets, the KMS CMK, and THE CONSENT BOUNDARY.

AWS Architecture §2. The bucket separation *is* the consent gate; this stack is the half
of it that IAM enforces.

> "Writes to actuate-delivery-* are permitted only for the packaging role... So a bug in
>  app code cannot leak un-consented data into the customer-facing bucket — IAM stops it
>  too."

Three independent things must all fail for un-consented data to reach a customer:
  1. `io.consent.DeliveryWriter` — the code guard (fail-closed, tested by removal).
  2. `catalog.deliverable_episodes()` — an INNER JOIN that cannot return an unconsented row.
  3. This stack — an explicit `Deny` on `s3:PutObject` to the delivery bucket for every
     principal except the packaging role.

Belt, braces, and a second pair of braces. Any one of them can have a bug.

No account IDs are hardcoded. Environment comes from CDK context (`-c env=dev`).
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from constructs import Construct


class StorageStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, *, env_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.env_name = env_name
        is_prod = env_name == "prod"

        # ---------------------------------------------------------------- KMS
        self.key = kms.Key(
            self,
            "ActuateCmk",
            alias=f"alias/actuate-{env_name}",
            description="Actuate S3 + RDS encryption (AWS Architecture §4.3).",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.RETAIN if is_prod else cdk.RemovalPolicy.DESTROY,
        )

        # ------------------------------------------------------- access logs
        self.access_logs = s3.Bucket(
            self,
            "AccessLogs",
            bucket_name=f"actuate-access-logs-{env_name}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,  # log delivery cannot use a CMK
            enforce_ssl=True,
            removal_policy=cdk.RemovalPolicy.RETAIN if is_prod else cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=not is_prod,
        )

        # ------------------------------------------------------------ buckets
        self.raw = self._bucket(
            "Raw",
            "raw",
            versioned=True,
            lifecycle_rules=[
                # Video dominates cost (AWS Architecture §1, §7). Raw is written once and
                # read rarely — but kept forever, because reprocessing the corpus when a
                # perception model improves is the whole reason raw provenance exists.
                s3.LifecycleRule(
                    id="raw-to-deep-archive",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=cdk.Duration.days(30),
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.DEEP_ARCHIVE,
                            transition_after=cdk.Duration.days(90),
                        ),
                    ],
                )
            ],
        )

        self.work = self._bucket(
            "Work",
            "work",
            versioned=False,  # churny intermediates; versioning here is pure cost
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="work-intelligent-tiering",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=cdk.Duration.days(30),
                        )
                    ],
                    abort_incomplete_multipart_upload_after=cdk.Duration.days(7),
                )
            ],
        )

        self.delivery = self._bucket("Delivery", "delivery", versioned=True)
        self.artifacts = self._bucket("Artifacts", "artifacts", versioned=True)

        # -------------------------------------------------- THE CONSENT BOUNDARY
        # In prod, only an ECS task may assume this. In dev we additionally allow an
        # operator in this account to assume it -- not for convenience, but because the
        # consent boundary is worthless if it cannot be TESTED. Proving the boundary means
        # showing both directions: that a non-packaging principal is denied, and that the
        # packaging role actually succeeds. Without an assumable path, the positive half
        # could only be asserted, never observed. AWS Architecture §8 requires the boundary
        # be "stood up and tested first, and re-verified whenever roles change".
        assumed_by: iam.IPrincipal = iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        if not is_prod:
            assumed_by = iam.CompositePrincipal(
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
                iam.AccountRootPrincipal(),
            )

        self.packaging_role = iam.Role(
            self,
            "PackagingRole",
            role_name=f"actuate-packaging-{env_name}",
            assumed_by=assumed_by,
            description=(
                "The ONLY principal permitted to write to the delivery bucket. "
                "AWS Architecture section 2."
            ),
        )
        self.pipeline_role = iam.Role(
            self,
            "PipelineRole",
            role_name=f"actuate-pipeline-{env_name}",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            # ASCII only. IAM rejects a role description containing characters outside
            # [\t\n\r\x20-\x7e\xa1-\xff] -- an em-dash here failed the whole stack.
            description=(
                "Raw/work processing. Deliberately has NO PutObject on delivery: a bug "
                "in pipeline code must not be able to reach the customer-facing bucket."
            ),
        )

        for b in (self.raw, self.work, self.artifacts):
            b.grant_read_write(self.pipeline_role)
            b.grant_read_write(self.packaging_role)
        self.key.grant_encrypt_decrypt(self.pipeline_role)
        self.key.grant_encrypt_decrypt(self.packaging_role)

        # Packaging may write delivery. Pipeline is granted READ ONLY — note this is not
        # an oversight: nothing in the pipeline role's grant list gives it PutObject here.
        self.delivery.grant_read_write(self.packaging_role)
        self.delivery.grant_read(self.pipeline_role)

        # The explicit Deny. A grant elsewhere cannot override an explicit Deny in IAM, so
        # even if someone later attaches an over-broad policy to another role, writes to
        # the delivery bucket still fail unless the caller IS the packaging role.
        self.delivery.add_to_resource_policy(
            iam.PolicyStatement(
                sid="DenyDeliveryWritesExceptPackagingRole",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["s3:PutObject", "s3:PutObjectAcl", "s3:DeleteObject"],
                resources=[self.delivery.arn_for_objects("*")],
                conditions={
                    "ArnNotEquals": {"aws:PrincipalArn": self.packaging_role.role_arn}
                },
            )
        )

        # ------------------------------------------------------------ outputs
        for name, bucket in [
            ("RawBucket", self.raw),
            ("WorkBucket", self.work),
            ("DeliveryBucket", self.delivery),
            ("ArtifactsBucket", self.artifacts),
        ]:
            cdk.CfnOutput(self, name, value=bucket.bucket_name)
        cdk.CfnOutput(self, "KmsKeyArn", value=self.key.key_arn)
        cdk.CfnOutput(self, "PackagingRoleArn", value=self.packaging_role.role_arn)

    def _bucket(
        self,
        construct_id: str,
        kind: str,
        *,
        versioned: bool,
        lifecycle_rules: list[s3.LifecycleRule] | None = None,
    ) -> s3.Bucket:
        """Every bucket: Block Public Access ON, SSE-KMS, TLS-only, access logging."""
        is_prod = self.env_name == "prod"
        return s3.Bucket(
            self,
            construct_id,
            bucket_name=f"actuate-{kind}-{self.env_name}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.key,
            bucket_key_enabled=True,  # cuts KMS request cost dramatically on many objects
            enforce_ssl=True,  # adds the aws:SecureTransport=false Deny
            versioned=versioned,
            server_access_logs_bucket=self.access_logs,
            server_access_logs_prefix=f"{kind}/",
            lifecycle_rules=lifecycle_rules or [],
            removal_policy=cdk.RemovalPolicy.RETAIN if is_prod else cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=not is_prod,
        )
