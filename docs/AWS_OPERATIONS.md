# Actuate — AWS Storage & Database Operations

**Audience:** whoever owns AWS infra. **Status:** dev environment live as of 2026-07-25.
**Account:** DatraAI-owned **812607971995**, region **eu-north-1** (Stockholm).

This describes how the `actuate` CLI (the `src/actuate` code path) persists data to AWS:
**blobs → S3, catalog metadata → Aurora Postgres.** Everything below is the `dev` env.

---

## 1. The big picture

```
                actuate CLI (laptop / future in-VPC compute)
                          |
        ┌─────────────────┴──────────────────┐
        │                                     │
   blobs (video, canonical            catalog metadata
   episodes, parquet, zarr)           (captures, consent, episodes,
        │                              embeddings, jobs…)
        ▼                                     ▼
   Amazon S3 (4 buckets)          Aurora Serverless v2 Postgres
   actuate-{raw,work,                 (private subnets, pgvector)
   delivery,artifacts}-dev                    ▲
                                              │ private — reach via
                                       SSM bastion port-forward
```

Design rule baked into the code: **blobs live in S3, everything queryable lives in
Postgres with an S3 URI pointer.** A DB row never contains pixels; it points at them.

---

## 2. Storage — Amazon S3

**Four buckets, and their separation IS the consent boundary** (do not collapse them):

| Bucket | Purpose |
|---|---|
| `actuate-raw-dev` | Original captures (video/IMU). Content-addressed. |
| `actuate-work-dev` | Processed/canonical episodes (episode.json, frames.parquet, dense arrays). |
| `actuate-delivery-dev` | Consent-cleared datasets ready to ship. **Writes are consent-gated, fail-closed.** |
| `actuate-artifacts-dev` | Misc build/QA artifacts. |
| `actuate-access-logs-dev` | S3 access logs. |

- Bucket names are always `actuate-<bucket>-<env>` — derived in code, never hardcoded.
- Encrypted with a dedicated KMS key (`Actuate-Storage-dev.KmsKeyArn`).
- The app talks to S3 only through one abstraction (`StorageBackend`); there is no scattered
  boto3. Swapping local⇄S3 is a single config switch, not a code change.
- Deployed by CDK stack **`Actuate-Storage-dev`** (already live).

**Cost:** storage + request costs only (pennies at current scale). Video dominates long-term;
never duplicate or re-encode it.

---

## 3. Database — Aurora Serverless v2 Postgres

Deployed by CDK stack **`Actuate-Data-dev`** on 2026-07-25.

| Property | Value |
|---|---|
| Engine | Aurora PostgreSQL **16.13** (Serverless v2) |
| Cluster endpoint | `actuate-catalog-dev.cluster-cvya6qc649kf.eu-north-1.rds.amazonaws.com:5432` |
| Database / user | `actuate` / `actuate_admin` |
| Credentials | **AWS Secrets Manager** `arn:aws:secretsmanager:eu-north-1:812607971995:secret:actuate/dev/catalog-4Kf6Lb` — password never leaves AWS, never in git |
| Network | **Private isolated subnets** (no public endpoint) in VPC `vpc-0011d1543343afed4` |
| Capacity | `min=0` (auto-pauses when idle → ~\$0), `max=2 ACU` |
| Schema | 16 tables + `pgvector`, created by `alembic upgrade head` |
| Bastion | t4g.nano `i-0be25b8eb18b09e71` (SSM only, no open inbound ports) |

**Why Serverless v2 min=0:** with one recording in the catalog the DB is asleep almost all
the time. Trade-off: the **first query after idle takes ~15–30 s to wake** (the app sets a
60 s connect timeout for this). Fine for a batch catalog; prod would keep a warm floor.

**Cost (dev):** ~\$3/mo bastion + near-\$0 Aurora while idle (up to ~\$45/mo only under
sustained load). NAT-free (isolated subnets). This env's idle bill is basically the bastion.

---

## 4. How connectivity works (important)

Aurora is **private**. Two ways in:

1. **In-VPC compute (production path):** future ECS/Lambda in this VPC reads the secret ARN
   and connects straight to the cluster endpoint. Clean, no tunnel.
2. **A laptop (current dev path):** cannot reach the private DB directly. It opens an **SSM
   port-forward through the bastion** — `localhost:15432` → Aurora:5432 — with no inbound
   ports ever opened. The app is told to use `127.0.0.1:15432` via an override, but still
   pulls the *password* from Secrets Manager (never copied into a file).

Open the tunnel (must stay running during local `actuate` use):

```bash
aws ssm start-session --target i-0be25b8eb18b09e71 \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters host=actuate-catalog-dev.cluster-cvya6qc649kf.eu-north-1.rds.amazonaws.com,portNumber=5432,localPortNumber=15432 \
  --profile datraai-admin --region eu-north-1
```
Requires AWS CLI + Session Manager plugin (both installed on the current dev machine).

---

## 5. Configuration (`.env.local`, git-ignored)

The app reads these (prefix `ACTUATE_`):

```
ACTUATE_STORAGE_BACKEND=s3
ACTUATE_AWS_PROFILE=datraai-admin        # DatraAI account — NOT the default profile
ACTUATE_AWS_REGION=eu-north-1
ACTUATE_ENV=dev
ACTUATE_DB_SECRET_ARN=arn:aws:secretsmanager:...:secret:actuate/dev/catalog-4Kf6Lb
ACTUATE_DB_ENDPOINT_OVERRIDE=127.0.0.1:15432   # laptop-via-tunnel only; unset in prod
```

⚠️ **Profiles / accounts** (the one thing not to get wrong):

| Profile | Account | Use |
|---|---|---|
| `datraai-admin` | 812607971995 (DatraAI) | ✅ deploy + admin |
| `DatraAI` | 812607971995 (DatraAI) | ✅ least-priv local CLI |
| `default` | **985368780855 (PARTNER)** | ❌ never — different org's account |

---

## 6. Deploy / operate (runbook)

```bash
# From infra/. Account must be named explicitly (guards against the partner account).
ACTUATE_AWS_ACCOUNT=812607971995 AWS_REGION=eu-north-1 \
  cdk deploy Actuate-Data-dev -c env=dev -c bastion=true \
  --profile datraai-admin --require-approval never

# Create/upgrade schema (with the tunnel open, from repo root):
alembic upgrade head            # picks up the DB URL from settings automatically

# Stacks: Actuate-Storage-dev (S3+KMS), Actuate-Data-dev (Aurora+VPC+bastion),
#         Actuate-Budget-dev (cost alerts to pushkar@datraai.com).
```

**To pause billing on the bastion:** stop EC2 instance `i-0be25b8eb18b09e71` (start it when
you next need local DB access). Aurora auto-pauses on its own.

---

## 7. Open items / caveats (hand-off notes)

- **v1 pipeline is not S3-aware.** `run_pipeline.py` → `scripts/NN_*.py` still writes
  intermediates to local `processed/` and only uploads at the final `--upload` step (a
  separate bucket var `DATRAAI_S3_BUCKET`). The S3/Aurora setup here governs the **`actuate`
  CLI** path only. Migrating the v1 path is future work.
- **Catalog is empty.** The old local Postgres rows were not migrated (their blob pointers
  were stale/deleted). Fresh start.
- **Delivery is consent-gated (fail-closed)** and the on-hand capture is `consent=pending`,
  so delivery-bucket writes are blocked by design until consent is resolved.
- **No prod env yet.** Everything above is `dev`. Prod would use `-c env=prod` (warm Aurora
  floor, longer backups, no auto-pause) and its own buckets/secret.
- **Secrets Manager rotation** is not configured — worth enabling for prod.

---

## 8. Verified working (2026-07-25)

- S3: `write_episode` round-trips **bit-exact** to `actuate-work-dev`.
- Aurora: capture/consent/episode rows persist and read back; FK + consent constraints
  enforced; episode `canonical_uri` points at the S3 blob; `pgvector` present.
- End-to-end proof cleaned up after itself (no residual test data).
