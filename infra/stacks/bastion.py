"""SsmBastion -- reach the private catalog WITHOUT opening a single inbound port.

The Aurora catalog lives in private isolated subnets with an empty ingress list, which is
correct and which also makes it unreachable from a developer laptop. The three usual ways
out of that are:

  1. Make the DB publicly accessible with an IP allowlist. Puts a database endpoint on the
     public internet, and breaks whenever a residential IP changes.
  2. RDS Data API. No network exposure, but psycopg and alembic do not speak it.
  3. This: a bastion reached through **SSM Session Manager**.

SSM is the right answer because the bastion has **no inbound security group rules at all**.
The agent on the instance dials *out* to SSM; AWS brokers the session. There is no open
port to scan, no SSH key to leak, no IP allowlist to maintain, and every session is IAM-
authorised and logged in CloudTrail. The database stays private.

    aws ssm start-session --target <instance-id> \\
        --document-name AWS-StartPortForwardingSessionToRemoteHost \\
        --parameters host=<db-endpoint>,portNumber=5432,localPortNumber=5432

...then psycopg/alembic/pytest connect to localhost:5432 exactly as they would locally.

This is a Construct, not a Stack, and lives inside DataStack on purpose: it needs the VPC
*and* it adds an ingress rule to the DB security group, so as a peer stack it forms a
dependency cycle (Data -> Bastion for the SG; Bastion -> Data for the VPC) that CDK
rejects outright.

Cost: a t4g.nano (~$3/mo), stoppable when idle.
"""

from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from constructs import Construct


class SsmBastion(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        vpc: ec2.IVpc,
        db_security_group: ec2.ISecurityGroup,
        db_port: int = 5432,
    ) -> None:
        super().__init__(scope, construct_id)

        # allow_all_outbound=True is required: the SSM agent must reach the SSM endpoints.
        # There are deliberately NO ingress rules -- that is the entire point.
        self.security_group = ec2.SecurityGroup(
            self,
            "Sg",
            vpc=vpc,
            description="SSM bastion. NO inbound rules: sessions arrive via the SSM agent "
            "dialing out, never via an open port.",
            allow_all_outbound=True,
        )

        self.instance = ec2.Instance(
            self,
            "Instance",
            instance_name=f"actuate-bastion-{env_name}",
            vpc=vpc,
            # Public subnet so the SSM agent has a route out. With nat_gateways=0 in dev
            # (saving ~$35/mo), an instance in a private subnet could not reach SSM at all.
            # It still has no inbound rules, so "public subnet" means routable outbound,
            # not reachable inbound.
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T4G, ec2.InstanceSize.NANO
            ),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(
                cpu_type=ec2.AmazonLinuxCpuType.ARM_64
            ),
            security_group=self.security_group,
            # No SSH key pair. There is no way in except SSM, so a key would only be a
            # credential to lose.
            require_imdsv2=True,
        )

        # The only permission the bastion needs.
        self.instance.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
        )

        # THE one ingress rule in this whole environment: the database accepts connections
        # from the bastion's security group, and from nothing else. Not from a CIDR, not
        # from the VPC -- from this security group specifically.
        db_security_group.add_ingress_rule(
            peer=self.security_group,
            connection=ec2.Port.tcp(db_port),
            description="Postgres from the SSM bastion only.",
        )
