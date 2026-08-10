# Actuate

Actuate turns egocentric manipulation video into a provenance-carrying canonical episode,
quality certificate, and (when every gate passes) a LeRobot v3 or RLDS dataset.

The product is deliberately fail-closed: an absent measurement stays absent, uncalibrated
model scores are not shown as probabilities, and consent, PII, task, or retargeting failures
block the corresponding export rather than being replaced with plausible values.

Read [STATUS.md](STATUS.md) for real-data validation status and
[docs/RESEARCH_BASIS.md](docs/RESEARCH_BASIS.md) for the papers behind the trust and data
contracts. The reproducible NVIDIA acceptance record is in
[docs/GPU_VALIDATION_2026-08-10.md](docs/GPU_VALIDATION_2026-08-10.md).

> **Commercial-use warning:** the default WiLoR/MANO/UniDepth perception chain is not
> commercially cleared under its published terms, and WiLoR also brings an Ultralytics
> licensing decision. Do not sell or deliver that runtime or its generated artifacts to a
> commercial customer until separate rights are signed or the models are replaced. See
> [docs/COMMERCIAL_LICENSE_READINESS.md](docs/COMMERCIAL_LICENSE_READINESS.md).

## Supported today

- Head-mounted egocentric video: enabled and real-data verified.
- Stereo: registered but disabled until real stereo calibration and end-to-end validation pass.
- UMI, teleoperation, glove, and DexUMI: deferred; not offered as working inputs.
- WiLoR hands, UniDepth metric depth, Grounding DINO + SAM2 objects: live pipeline.
- LeRobot v3: supported behind consent, PII, task, and data-integrity gates.
- RLDS: supported only in its isolated `[rlds]` environment; TensorFlow never shares the GPU
  worker process with perception.
- Robot retargeting: IK and MuJoCo validation are implemented, but no trained root-frame model
  ships yet. Robot trajectories therefore skip honestly by default.

## Install

For the normal GPU processing path:

Python 3.10-3.13 is supported. Schema v5 is generated with an exact Pydantic version, so
Python 3.14 is intentionally rejected until that frozen contract is migrated explicitly.

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[perception,retarget,sim,sources]"
```

Core/schema/tests without GPU models:

```bash
python -m pip install -e ".[dev]"
```

LeRobot and RLDS must be installed separately. LeRobot's Torch/TorchVision floor conflicts
with WiLoR's tested Torch ceiling; RLDS's TensorFlow/protobuf stack conflicts with MediaPipe.

```bash
python -m venv .venv-lerobot
source .venv-lerobot/bin/activate
python -m pip install -e ".[export]"
```

RLDS uses a third environment:

```bash
python -m venv .venv-rlds
source .venv-rlds/bin/activate
python -m pip install -e ".[rlds]"
```

For the dashboard worker, set `ACTUATE_LEROBOT_PYTHON` and `ACTUATE_RLDS_PYTHON` to those
environments' Python executables. For the CLI export step, invoke `actuate` from
`.venv-lerobot`; canonical artifacts are portable between the environments.

## Customer flow

```bash
.venv/bin/actuate login --local
.venv/bin/actuate status

.venv/bin/actuate process my_video.mp4 \
  --rig head_mounted \
  --consent-granted \
  --redact-pii \
  --out my_output

.venv/bin/actuate report my_output/my_video_out

.venv-lerobot/bin/actuate export my_output/my_video_out \
  --format lerobot_v3 \
  --out my_output/lerobot_v3

.venv-lerobot/bin/actuate deliver my_output/lerobot_v3 \
  --customer example_lab \
  --episodes my_output/my_video_out/canonical.json \
  --local-root ./delivered
```

`--consent-granted` is an explicit attestation that every recorded subject granted this use;
local file ownership is not treated as consent. `--redact-pii` runs the privacy pass required
for export/delivery. Revocation blocks every later export.

The default processes the full source. `--max-frames N` is an explicit sampling mode and the
coverage is recorded. Add `--verbose` only when diagnosing model/dependency output; the normal
console shows Actuate's stage results and exact skip reasons.

If no task is supplied, Actuate uses `ANTHROPIC_API_KEY` for optional automatic task reading;
without the key it prompts. To avoid either behavior, pass `--task` explicitly. VLM language
annotation is opt-in and may incur API cost.

Customer S3 delivery is not generally available yet. `--local-root` is the verified delivery
path. Operator-only AWS commands are documented under `docs/architecture/` and should never
be run against ambient or partner-account credentials.

## Outputs

Every run writes a `README.md`, frozen canonical JSON Schema, `run_manifest.json` with code and
dependency lineage, and `checkpoint.json` with each stage's status and skip reason. The source
may be staged for reproducibility, and dataset writers may decode or re-encode video; Actuate
does not claim a no-copy/no-re-encoding path.

`pipeline.rrd` opens in the Rerun viewer. Dense depth lives under `artifacts/depth/`. Internal
perception caches live outside the customer output tree under `~/.cache/actuate` by default.

Identifiers:

- `capture_id`: SHA-256 identity of the original capture bytes.
- `episode_id`: capture prefix plus a segmentation suffix such as `_ep00`.
- `schema_version`: frozen canonical contract version (currently v5).

## Verification

```bash
pytest tests/unit
lint-imports
actuate schema freeze --check
```

Integration tests have explicit dependency markers and may require Docker, GPU models, AWS, or
the isolated RLDS environment. A feature is not described as real-data validated merely because
its unit tests pass.

## Architecture

The library is the source of truth; CLI and API are thin consumers. Dependency direction is
checked by import-linter.

```text
src/actuate/
  ingest/ perception/ fusion/ canonical/ certify/
  retarget/ language/ package/ feedback/
  config/ schema/ io/ catalog/
  cli/ service/
```

The canonical schema is frozen and provenance-enforcing. See
[docs/architecture/MASTER_IMPLEMENTATION_SPEC.md](docs/architecture/MASTER_IMPLEMENTATION_SPEC.md)
for the contract and [docs/PIPELINE_STATUS.md](docs/PIPELINE_STATUS.md) for the evidence ledger.
