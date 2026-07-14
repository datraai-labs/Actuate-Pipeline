# Actuate / datraai-pipeline — Project Context

Multimodal capture → certified, retargeted, **VLA-training-ready** robot data.
Six capture rigs in (egocentric, UMI gripper, stereo, glove, teleop robot, DexUMI exoskeleton); LeRobot v3 + RLDS datasets out.

## The documents that govern — read before designing anything

| Doc | Role |
|---|---|
| [docs/architecture/MASTER_IMPLEMENTATION_SPEC.md](docs/architecture/MASTER_IMPLEMENTATION_SPEC.md) | **THE DOCUMENT WE BUILD FROM.** Consolidates v1 + v2 + v3 + v3.1. Its §3 canonical schema is **the freeze point**. Its §8 is the build order. |
| [docs/architecture/AWS_ARCHITECTURE.md](docs/architecture/AWS_ARCHITECTURE.md) | Storage (4 S3 buckets), the Postgres catalog, and **the consent boundary enforced in IAM, not just code.** |
| [docs/architecture/ARCHITECTURE_V1.md](docs/architecture/ARCHITECTURE_V1.md) · [ARCHITECTURE_V2.md](docs/architecture/ARCHITECTURE_V2.md) · [IMPLEMENTATION_SPEC.md](docs/architecture/IMPLEMENTATION_SPEC.md) | **Historical.** Superseded by the Master Spec, which consolidates them. Useful for the *why* behind a design, not as the plan. |

Original PDFs in [docs/architecture/pdf/](docs/architecture/pdf/), including the Build Increment 1 scope prompt.

The other file that matters: [docs/PIPELINE_STATUS.md](docs/PIPELINE_STATUS.md) — the honest ledger of what is genuinely unvalidated. **Trust it over any ✅ elsewhere.**

## The standing discipline — not boilerplate; it caught every real bug in this project

> **CLI/API-first.** Everything is a library function first. Typer CLI and FastAPI service are thin consumers, never dependencies. **If a capability only works through the CLI or the service, it is wrong.** Enforced by an import-linter contract in CI.
>
> **Verify against real data.** A component is "done" only when tested against real data, and **every correctness test must be confirmed to FAIL against a broken version, not merely pass against the current one.** "Looks right" is not a status.
> — Master Spec §0

Concretely:
- Report status as **tested-against-real-data / unit-only / written-only**. If you didn't run it, say so.
- Schema-correctness asserted by our own test is **not** verification. The LeRobot exporter is verified when *LeRobot's own loader* reads it and a real training step runs.
- Prefer an honest `None` over a plausible default. `contact.<finger> = None` means *not measured*; `0.0` would mean *measured, nothing touching*. These are not the same claim.
- Never silently resolve a disagreement between two signals. Flag it and route to review.
- **The fail-closed consent/PII gate is always-on and hard-blocking.** Re-test it against the *auto-consent-bypass* bug class at every phase.

## Build order (Master Spec §8)

**Phase 0** freeze §3 schema → **Phase 1** canonical build from v1 outputs → **Phase 2** LeRobot v3 / RLDS exporters *(**ships value**; gate = load with LeRobot's own loader + one real training step)* → **Phase 3** L1 model swaps + L2 fusion → **Phase 4a** dexterous retargeting *(highest risk)* → **Phase 5** language + delivery hardening.

Phase 2 is deliberately before Phase 4: gripper/arm VLA data ships without waiting on the hardest component.

## Repo shape

Mid-migration from a flat numbered-script collection to the package layout of Master Spec §2.1.

```
src/actuate/          # the library — source of truth. schema/ io/ catalog/ config/ + layer pkgs.
scripts/NN_stage.py   # the 17 v1 stages. Still the execution path (run_pipeline.py).
                      # Numbered so DAG order is visible from the filesystem — keep that.
infra/                # AWS CDK (Python): StorageStack, DataStack
config.py             # v1's 633-line de-facto spec. Read before changing behavior.
service/api.py        # v1 FastAPI. Known-fragile: in-memory job state + regex-parses its own log.
```

Dependency direction is one-way and CI-enforced: `cli/` and `service/` import layers; layers import only `schema/`, `io/`, `config/`, `catalog/`, and each other in pipeline order. **Nothing in a layer imports `cli/` or `service/`.**

## Known-shaky — do not build on these without checking

- **Monocular depth has never run on real GPU hardware** in a dev environment, yet all metric 3D depends on it.
- **Task classification is unvalidated.** No session on hand is confirmed to depict any task in `TASK_SIGNATURES`; the classifier correctly returns `unknown` on the one real session.
- **`"dual"` IMU mode fuses one physical stream duplicated into two roles.** Not real fusion.
- **The corpus is n=1.** `processed/` has 7 directories but holds **one** 95-second recording (4 copies), 1 corrupt, 2 incomplete. See PIPELINE_STATUS.md.
- **Local consent records conflict.** The same capture is `pending` under `session_001` and `granted` under its 3 UUID copies. Consent must key on `capture_id` (AWS doc §3), not session.

## AWS

Infra targets **DatraAI's own AWS account** via the `datraai-admin` profile. The credentials otherwise present on a dev machine may belong to a *partner* account (`vendor-upload-only` @ <PARTNER_ACCOUNT>, holding `northstar-*`/`humanstryde-*` buckets) — **never provision Actuate infrastructure there.**

Stand up the **consent boundary first**, before real capture data lands (AWS doc §8).
