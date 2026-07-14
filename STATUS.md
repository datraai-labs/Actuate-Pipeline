# Actuate — Honest Status

Per Master Spec §0: *"A component is 'done' only when tested against real data, and every
correctness test must be confirmed to fail against a broken version. 'Looks right' is not
a status."*

Graded **tested-against-real-data** / **unit-only** / **written-only**. Nothing is graded
on whether it compiles.

**Increment 2 complete (Parts A, B, C).** 2026-07-14 · `schema_version = 2` ·
**493 passed, 0 skipped, 0 failed** · import-linter 2/2 · schema frozen · ruff clean

---

## ⚠️ Part C proves EXPORTER MECHANICS. It is NOT a shippable milestone.

The LeRobot v3 load+train gate passes. **This does not mean we have training-ready data.**
Three independent reasons, each sufficient on its own:

1. **The capture is NOT DELIVERABLE.** `consent=pending`, `pii_status=pending`. It cannot
   go to a customer, and the three-layer boundary refuses it. Export is nonetheless legal —
   **consent gates *delivery*, not internal processing** — but nothing here ships.

2. **n = 1.** One 95-second clip. That is a functioning exporter, not a dataset. Diversity,
   dedup, stratified splits, and scaling-law value all require a corpus we do not have.

3. **The action is EGO-CONTAMINATED.** There is no `camera_pose` — L1 SLAM (ORB-SLAM3) is
   not built. On this head-mounted rig **the camera moves**, so the wrist delta between
   frames is `hand_motion + head_motion`. A policy trained on it would learn to predict head
   motion as if it were hand motion. Master Spec §L3's stable-frame reprojection is
   implemented (`canonical/reproject.py`) and **refuses to run without ego-motion** rather
   than silently returning the contaminated value. The warning ships inside the dataset
   (`meta/actuate_provenance.json`).

What Part C *does* prove: the exporter's mechanics are correct and verified against the
real library — format, chunking, the delta-timestamp action-chunk path, normalization, and
that a real policy can consume the result.

### The gate (Master Spec §L7 — non-negotiable)

> *"load the exported LeRobot v3 dataset with LeRobot's own loader and run one real
> training step... A schema-valid-but-untrainable export must be caught by this gate — do
> not assert schema-correctness in place of a real load+train."*

```
LOADED with LeRobot's own loader : 2680 frames, 1 episode, fps=30, codebase_version=v3.0
  observation.images.head  (3, 224, 224)   <- per-camera chunked MP4
  observation.state        (8,)            <- [x y z qw qx qy qz grasp]
  action                   (16, 8)         <- action chunk, delta-timestamp native
  task                     'Sort and staple paperwork at the workbench.'

REAL TRAINING STEP (ACT, 51.6M params, CPU, 2.1s)
  loss before : 84.397583
  grad norm   : 1485.85      (finite)
  loss after  : 49.145649
```

Written **through LeRobot's own writer** (`create` → `add_frame` → `save_episode`), so
schema correctness is *inherited, not re-implemented* (§L7).

### The broken variant — the gate actually bites

`test_a_schema_valid_but_UNTRAINABLE_export_is_caught_by_the_gate`: poison 10% of the
actions with NaN — the realistic failure, i.e. "just fill the gaps" instead of dropping
frames with no detected hand.

The result **still loads**. `meta/info.json` is still correct. Every schema-correctness
assertion we could write still passes. **The training step is what catches it** (non-finite
loss / gradients). That is the entire argument for having this gate rather than a schema
test.

### What the exporter REFUSES

- **No `task`** → `ExportRefused`. v1's classifier returns `unknown` and its language
  grounding emits *"Perform unknown task using right hand with power grasp."* — a fluent
  sentence containing no task. Exporting that would launder a failed classification into a
  training label. LeRobot does `frame.pop("task")` and would KeyError three layers down; we
  refuse where a human can read why. **The task used here is operator-supplied**, grounded
  in two independent VLM reads recorded in `docs/PIPELINE_STATUS.md` — explicitly *not* the
  v1 classifier.
- **No video** → `ExportRefused`. A VLA dataset without `observation.images.*` is not a VLA
  dataset, and LeRobot's own policies reject it outright.
- **Frames with no detected hand are DROPPED (170 of 2850), not zero-filled.** Zeroing would
  teach a policy to drive the end-effector to the camera origin every time the hand left
  view.

### Normalization (TRI LBM: normalization dominates)

1st/99th percentile → `[-1, 1]` by default (π0.5/EgoVerse), with **raw percentiles AND
mean/std both shipped** so a customer on 2/98-per-timestep (TRI LBM) or z-score (EgoMimic)
can re-derive without a full pass. Round-trip verified.

---

## Part B — the real capture is in S3, and it cannot ship

**190.7 MB of real human capture data now lives in `actuate-raw-dev`.** It is
**not deliverable**, and that was proven by attempting to ship it and watching all three
layers refuse (`tests/integration/test_migrated_capture_is_not_deliverable.py`):

```
LAYER 2 (catalog INNER JOIN) : deliverable_episodes() -> []
LAYER 1 (code guard)         : ConsentViolation: consent=pending, required=granted
LAYER 3 (IAM, as ADMIN,
         bypassing all our code): AccessDenied
actuate-delivery-dev          : 0 objects
```

### Content-addressed provenance (schema v2)

**The capture id IS the SHA-256 of the raw bytes.** This does not *detect* the Increment-1
failure classes — it removes them:

| Increment-1 bug | Now |
|---|---|
| 1.49 MB video filed under 181.84 MB metadata (frame counts internally consistent, so any check on those waved it through) | The manifest carries the hash of the bytes it describes. A swapped payload is detected by construction; there is nothing to remember to check. |
| Same footage under 4 session ids → conflicting consent | Identical bytes compute an identical id and collapse to **one** capture. There is no second record to disagree with the first. Re-running migrate printed `already present (dedup hit)` and re-uploaded nothing. |
| Metadata drifting from payload | `CanonicalEpisode.source_content_hash` binds every derived artifact back to the capture, and a validator refuses an episode whose hash disagrees with its capture id. |

### The consent conflict is logged as what it actually is

```
DUPLICATE_CONFLICT  GRANTED  source=adf1b750-2e6c-458f-a557-80a3efb1995c
DUPLICATE_CONFLICT  PENDING  source=session_001
RECONCILED          PENDING
```

Three sources said `granted`, one said `pending`. Least-permissive wins: **`pending`**. A
majority vote would be a consent gate that a duplicate upload can outvote.

And it is logged as `DUPLICATE_CONFLICT`, **not `REVOKED`**. Nobody withdrew consent; we
uploaded the same footage twice and the copies disagreed. Both block, but they are not the
same event — and "why is this blocked?" is the only question anyone will ever ask of that
table.

### 6 sessions refused, nothing written for them

Including the two whose `raw.mp4` is 1.49 MB while their metadata describes a 181.84 MB
source — the bug that passes any check looking only at frame counts.

### A test fixture was deleting production data

The catalog integration tests did `DELETE FROM captures` in cleanup. Run against the
deployed Aurora — which is how we run them — **that wiped the real migrated capture**, and
the end-to-end tests began skipping with "the real capture has not been migrated". Cleanup
is now scoped to a `testcap_` prefix; real captures are content-addressed (64 hex chars) and
cannot match it. A test fixture must never be able to destroy real data.

---

## The consent boundary — ALL THREE LAYERS NOW PROVEN

Increment 1 had only the code guard executed. All three are now proven by
**attempt-and-observe against live infrastructure**, each with a broken variant confirmed
to fail.

| Layer | Status | How it was proven |
|---|---|---|
| **1. Code guard** (`io.consent.DeliveryWriter`) | ✅ **PROVEN** | 11 tests. Removing the guard **leaks data into the delivery bucket** — asserted, red→green. `Settings(allow_unconsented_delivery=True)` is refused by a validator, so there is no supported bypass. |
| **2. Catalog INNER JOIN** (`deliverable_episodes`) | ✅ **PROVEN — against real Aurora** | 10 tests on the deployed cluster. A LEFT-JOIN variant **provably leaks an episode with no consent record at all** (`test_a_left_join_variant_LEAKS_unconsented_episodes`); the real INNER JOIN cannot return it. |
| **3. IAM Deny** (delivery bucket policy) | ✅ **PROVEN — real PutObject, really refused** | 4 tests against the live bucket. An **`AdministratorAccess` user attempted a real `PutObject` to `actuate-delivery-dev` and got `AccessDenied`** — an explicit Deny beats full admin. The packaging role assumed and **succeeded**, so the boundary is a boundary and not an outage. `actuate-work-dev` still accepts writes, so the Deny is scoped to delivery and not sprayed. |

`actuate storage verify-consent-boundary --env dev` passes end to end: Block Public Access,
SSE-KMS, versioning on raw+delivery, explicit Deny present.

> The IAM test is the one that could not be faked. Asserting a policy *exists* proves
> nothing — a policy can exist and be scoped to the wrong ARN, shadowed by an Allow, or
> attached to the wrong bucket. Only a refused write is evidence.

---

## Deployed infrastructure (account <ACTUATE_AWS_ACCOUNT>, eu-north-1)

| Stack | Contents | Status |
|---|---|---|
| `Actuate-Budget-dev` (us-east-1) | $50/mo budget; alerts at 80% actual and 100% **forecast** | ✅ live |
| `Actuate-Storage-dev` | 4 buckets + access logs, KMS CMK, packaging/pipeline roles, **the delivery Deny** | ✅ live |
| `Actuate-Data-dev` | Aurora PG 16.4 Serverless v2 (**min 0 ACU — auto-pauses, ~$0 idle**), Secrets Manager, VPC, SSM bastion | ✅ live |

Catalog schema applied via `alembic upgrade head`: **15 tables, `vector` extension, 4 enum
types**, `consent` FK → `captures`.

**The database is not reachable from the internet.** It sits in private isolated subnets
whose security group has exactly one ingress rule: port 5432 from the bastion's security
group. The bastion itself has **zero inbound rules** — access is via SSM Session Manager,
which the agent dials *out* to. No open port, no SSH key, no IP allowlist.

---

## Per-deliverable status

| Deliverable | Status | What actually ran |
|---|---|---|
| Repo skeleton, CLI/API-first | **tested** | `lint-imports` 2/2. Demonstrated failing on a bad import, then fixed. |
| Frozen canonical schema (§3) | **tested-against-real-data** | Round-trips **real session_001 keypoints** → Parquet/Zarr → reload, **bit-exact**. Drop-a-required-field test red→green. |
| `io/` LocalBackend | **tested-against-real-data** | Bit-exact round-trip incl. the ~5% no-hand frames (return empty, not zeroed). |
| `io/` S3Backend | **tested-against-real-AWS** | Real `PutObject`/`GetObject` against `actuate-work-dev`; real `AccessDenied` against `actuate-delivery-dev`. |
| Consent guard | **PROVEN** | See boundary table. |
| `catalog/` Postgres | **tested-against-real-data** | ⬆️ *Was written-only in Increment 1.* 10 tests now pass against the **deployed Aurora**, not a container. |
| `infra/` CDK | **DEPLOYED and verified** | ⬆️ *Was synth-only.* All stacks live; boundary verified by attempted writes. |
| CI | **written-only** | Workflow exists; **has never run** (no push yet). |
| `actuate migrate` | **tested-against-real-data** | Real 190.7 MB capture uploaded to `actuate-raw-dev`, content-addressed; catalog registered; re-run is a dedup hit. |
| **L3 `canonical build`** | **tested-against-real-data** | Builds the frozen v2 schema from the real capture: 2680/2850 valid frames. Stable-frame reprojection implemented and **refuses** without ego-motion. |
| **L7 LeRobot v3 exporter** | **tested-against-real-data** | ⬆️ *Was written-only.* Passes the load+train gate: LeRobot's own loader + a real ACT policy + a real optimizer step. Broken-variant (NaN-poisoned) caught. |
| RLDS / TFDS exporter | **not built** | Deferred. |
| L1 SLAM (ego-motion) | **not built** | **Blocks correct actions on moving-camera rigs.** See the ego-contamination note above. |

---

## Bugs found this increment (all real, all caught before doing damage)

1. **`AWS::Budgets::Budget` does not exist in eu-north-1.** Budgets is a us-east-1-only
   global service. First deploy failed. Stack now pinned to us-east-1.
2. **An em-dash in an IAM role description failed the deploy.** IAM validates descriptions
   against `[\t\n\r\x20-\x7e\xa1-\xff]`; `cdk synth` does **not** catch this — only the AWS
   API does. CloudFormation rolled back cleanly. The `DataStack` security group had the
   identical bug queued behind it. `tests/unit/test_infra_ascii_descriptions.py` now catches
   it offline; demonstrated red→green.
3. **Aurora at `min_capacity=0` auto-pauses, and the first connection after a pause times
   out.** Presents as a bare `ConnectionTimeout` that looks like a network fault and isn't.
   A 60s connect timeout is now the default in `catalog/db.py` and `migrations/env.py` — not
   a workaround, but the correct setting for a database that is allowed to sleep.
4. **`alembic revision --autogenerate` emitted a `pgvector` column without importing
   pgvector.** Would have crashed on `upgrade`. Import added.
5. **Bastion-as-a-peer-stack is a CDK dependency cycle** (Data → Bastion for the SG;
   Bastion → Data for the VPC). It is a Construct inside `DataStack` instead.

---

## Still not done — and not claimed

- **CI has never run.** The workflow is written. That is all.
- **`migrate run` has not touched S3.** No real capture data is in any bucket yet. Part B.
- **`canonical build` (L3) and the LeRobot v3 exporter do not exist.** Part C. The
  load+train gate has not been attempted.
- **The corpus is still one 95-second capture**, and its consent resolves to `pending`.
  Nothing is deliverable. Proving the exporter's mechanics on it is legitimate; calling that
  a "training-ready delivery" milestone would not be.

---

## Security

**Key rotation: DONE.** The access key that was exposed during development has been deleted
from IAM and replaced; only the new key exists. Verified via `iam:ListAccessKeys`.

**Nothing sensitive is in this repository.** No access keys, no secret keys, no database
credentials, no AWS account IDs — not in the working tree and not in git history. The
account is supplied at runtime via `ACTUATE_AWS_ACCOUNT`; the DB URL comes from Secrets
Manager (or a gitignored `.env.local` locally, which is never printed).

**Consent-gated capture data is NOT tracked by git.** `raw/`, `processed/`, `delivery/`,
and any `*.mp4` are gitignored. The three-layer consent boundary guards S3 and the delivery
bucket — **it does not guard git**, so the repository must be kept clean by exclusion. Raw
provenance lives in `s3://actuate-raw-<env>/`, content-addressed, where consent can actually
be enforced and revoked.

## Cost

Idle: **~$4-5/month** — KMS CMK (~$1), Secrets Manager (~$0.40), SSM bastion t4g.nano
(~$3). **Aurora auto-pauses to $0.** Storage is negligible until real data lands.

The bastion is the only always-on compute and is stoppable when not in use.
