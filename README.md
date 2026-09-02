# Actuate

Multimodal robot capture → certified, retargeted, **VLA-training-ready** datasets
(LeRobot v3 / RLDS) for frontier robotics labs.

Six capture rigs in — egocentric head-mounted, UMI handheld gripper, stereo, instrumented
glove, teleoperated robot, DexUMI exoskeleton. Training-ready data out, with a
machine-checkable quality certificate and a **fail-closed consent gate** on every episode.

> **Read [`STATUS.md`](STATUS.md) before trusting anything here.** It grades every component
> as tested-against-real-data / unit-tested / written-only. Increment 1 built the
> foundation; most layers are deliberately empty.

## The documents that govern

| Doc | Role |
|---|---|
| [`docs/architecture/MASTER_IMPLEMENTATION_SPEC.md`](docs/architecture/MASTER_IMPLEMENTATION_SPEC.md) | **The document we build from.** §3 canonical schema is the freeze point; §8 is the build order. |
| [`docs/architecture/AWS_ARCHITECTURE.md`](docs/architecture/AWS_ARCHITECTURE.md) | Storage, the Postgres catalog, and the consent boundary **enforced in IAM, not just code**. |
| [`docs/PIPELINE_STATUS.md`](docs/PIPELINE_STATUS.md) | The honest ledger of what is genuinely unvalidated in the v1 pipeline. |

Earlier architecture docs (v1, v2) are kept as history in `docs/architecture/`; the Master
Spec consolidates them. The v1 pipeline's own README is at
[`docs/README_v1_pipeline.md`](docs/README_v1_pipeline.md).

## Two non-negotiables

1. **CLI/API-first.** Everything is a library function first; the Typer CLI and the separate
   `Actuate-dashboard` API are thin consumers, never dependencies. If a capability only works
   through one interface, it is wrong. **Enforced in CI** by an import-linter contract.
2. **Verify against real data.** A component is "done" only when tested against real data,
   and **every correctness test must be confirmed to fail against a broken version.**
   "Looks right" is not a status.

## Install

```bash
pip install -e ".[aws,dev]"   # core is CPU-only and light; heavy deps are extras
```

Extras: `[perception]` (WiLoR, depth models — GPU), `[retarget]` (IK), `[sim]` (MuJoCo),
`[aws]` (S3 + Postgres + pgvector). Dashboard API dependencies live in the separate
`Actuate-dashboard` repository. The core library, schema, certification, and exporters all
run **without a GPU stack** — so value ships before any model does.

## Run the tests

```bash
pytest tests/unit               # fast; no Docker, no AWS
pytest tests/integration        # needs Docker (spins up a real Postgres + pgvector)
pytest                          # everything, incl. the v1 pipeline suite
lint-imports                    # the CLI/API-first dependency contract
actuate schema freeze --check   # fails if the models drifted without a version bump
```

## The CLI

```bash
actuate schema freeze              # emit the versioned JSON Schema
actuate storage whoami             # WHICH AWS ACCOUNT ARE WE POINTED AT?  (see below)
actuate storage buckets            # the four bucket names for an env
actuate storage verify-consent-boundary --env dev --profile datraai-admin
actuate migrate plan               # what migration would do. Reads only, touches nothing.
actuate migrate run --backend s3 --env dev --profile datraai-admin --yes
```

Layer commands (`ingest`, `perceive`, `fuse`, `canonical`, `certify`, `retarget`,
`language`, `package`) parse and honestly print "not implemented" — Increment 1 built the
foundation only.

## AWS

**Actuate's infrastructure lives in account `<ACTUATE_AWS_ACCOUNT>`, region `eu-north-1`, via the
`datraai-admin` profile.**

> ⚠️ A dev machine here may have credentials for a **partner account**
> (`vendor-upload-only` @ `<PARTNER_ACCOUNT>`, holding `northstar-*` / `humanstryde-*` buckets)
> configured as its default. **Never provision Actuate infrastructure there.** Run
> `actuate storage whoami` if you are unsure. The CDK app hardcodes no account ID and
> requires `ACTUATE_AWS_ACCOUNT` to be set explicitly — it deliberately ignores
> `CDK_DEFAULT_ACCOUNT`, which the CDK CLI auto-populates from whatever credentials happen
> to be ambient.

### Bring up the consent boundary FIRST

AWS Architecture §8: *"The consent boundary (IAM + bucket policy + fail-closed gate) should
be stood up and tested first, before any real capture data lands."*

```bash
aws configure --profile datraai-admin                  # <ACTUATE_AWS_ACCOUNT>, eu-north-1
aws sts get-caller-identity --profile datraai-admin    # must print <ACTUATE_AWS_ACCOUNT>

cd infra
pip install -r requirements.txt
npm install -g aws-cdk

cdk synth -c env=dev                                   # offline; needs no credentials

export ACTUATE_AWS_ACCOUNT=<ACTUATE_AWS_ACCOUNT>
export AWS_REGION=eu-north-1
cdk bootstrap aws://$ACTUATE_AWS_ACCOUNT/$AWS_REGION --profile datraai-admin
cdk deploy -c env=dev --profile datraai-admin --all

# Then PROVE the boundary is intact, before any data lands:
actuate storage verify-consent-boundary --env dev --profile datraai-admin
```

Stacks: `StorageStack` (4 buckets + KMS CMK + the IAM consent boundary), `DataStack`
(Aurora Serverless v2 Postgres + Secrets Manager). `ComputeStack` and `ServiceStack` are
later increments.

### The consent boundary, in three independent layers

Un-consented data reaching a customer requires **all three** to fail:

1. **Code** — `io.consent.DeliveryWriter` refuses any write whose episode is not
   `consent=granted` **and** `pii_status=passed`. Fail-closed: a *missing* record blocks.
   Proven load-bearing by a test that removes the guard and watches data leak.
2. **Catalog** — `catalog.deliverable_episodes()` is an INNER JOIN through `consent` and
   `certifications`. An un-consented episode cannot be *returned*, let alone written.
3. **IAM** — the delivery bucket policy `Deny`s `PutObject` from every principal except the
   packaging role. An explicit Deny cannot be overridden by any Allow, anywhere.

Consent is keyed on **`capture_id`**, not episode — one physical recording yields many
episodes, and a revocation must revoke all of them at once.

## Storage layout

```
s3://actuate-raw-<env>/       <rig>/<capture_id>/...      versioned; -> Glacier Deep Archive
s3://actuate-work-<env>/      canonical/<episode_id>/vN/  Intelligent-Tiering
s3://actuate-delivery-<env>/  <customer>/<dataset>/<ver>/ CONSENT-PASSED ONLY
s3://actuate-artifacts-<env>/ checkpoints, URDFs, sim assets
```

Blobs in S3; everything queryable in Postgres with an S3 URI pointer. **Video is never
duplicated or re-encoded** — the canonical representation references chunked MP4, it does
not carry pixels.

## Repo shape

```
src/actuate/        the library — the source of truth
  config/           settings + rig registry (6 rigs) + embodiment registry
  schema/           THE FROZEN CANONICAL CONTRACT (schema_version=1)
  io/               storage backends, Parquet/Zarr store, the consent guard
  catalog/          Postgres (SQLAlchemy + Alembic + pgvector)
  ingest/ perception/ fusion/ canonical/ certify/ retarget/ language/ package/ feedback/
  cli/              Typer (thin)
infra/              AWS CDK: StorageStack + DataStack
scripts/NN_*.py     the 17 v1 stages — still the execution path (run_pipeline.py)
```

Dependency direction is one-way and CI-enforced: `cli/` imports layers; layers import only
`schema/`, `io/`, `config/`, `catalog/`, and each other in pipeline order. The external
dashboard imports the public SDK and is never imported by Core.
