# Quickstart: process and inspect a capture

This path produces a certified canonical episode, then exports it only if the dataset gates
are satisfied. Runtime depends on video length and GPU hardware; Actuate does not promise a
fixed five-minute conversion.

## 1. Install the isolated environments

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[perception,retarget,sim,sources]"

python -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -e ".[export]"
```

LeRobot's Torch/TorchVision requirements conflict with WiLoR's tested Torch ceiling. Keeping
the writer separate also prevents native model/export libraries from sharing one process.

## 2. Process the full video

```bash
.venv/bin/actuate login --local
.venv/bin/actuate process ./my_video.mp4 \
  --rig head_mounted \
  --task "pick up the cup" \
  --consent-granted \
  --redact-pii \
  --out ./actuate_runs
```

`--consent-granted` is an operator attestation for every recorded subject; local ownership is
not consent. The default evaluates the full source. Use `--max-frames N` only as an explicit
sample—the manifest records both the source count and resulting coverage. Stereo is detected
but currently rejected until calibrated real-data validation is complete.

## 3. Inspect before export

```bash
.venv/bin/actuate report ./actuate_runs/my_video_out
.venv/bin/actuate viz preview ./actuate_runs/my_video_out
```

The report uses `not measured` for absent signals and states why delivery or robot retargeting
is blocked. A missing trained embodiment model produces human-space data, not a fabricated
robot trajectory.

## 4. Export with the isolated writer

```bash
.venv-lerobot/bin/actuate export ./actuate_runs/my_video_out \
  --format lerobot_v3 \
  --out ./dataset
```

Export rechecks consent, PII status, task presence, successor integrity, grasp availability,
and any requested embodiment's physics-eligibility verdict.

## 5. Load with LeRobot

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "actuate/local",
    root="./dataset",
    video_backend="pyav",  # portable when system TorchCodec/FFmpeg libraries are absent
)
print(len(dataset), dataset[0]["observation.state"].shape)
```

## Python SDK

Use the SDK for processing inside the perception environment:

```python
import actuate

run = actuate.process(
    "./my_video.mp4",
    rig="head_mounted",
    task="pick up the cup",
    consent="granted",
    redact_pii=True,
)
print(run.status, run.quality, run.num_frames)
```

Run export from `.venv-lerobot` against the durable `canonical.json`, as shown above. The
dashboard performs the same boundary with `ACTUATE_LEROBOT_PYTHON`.

## Sources

| Prefix | Reads from | Example |
|---|---|---|
| *(none)* | local file/session | `./data/capture.mp4` |
| `hf://` | Hugging Face Hub | `hf://lerobot/pusht` |
| `s3://` | customer-configured S3 | `s3://bucket/session_001/` |
| `http(s)://` | direct URL | `https://example.org/capture.mp4` |
| `openx://` | Open X / TFDS slice | `openx://fractal20220817_data` |

Every output includes the frozen schema, a run README, code/dependency lineage, structured
frame coverage, explicit omissions, stage skip reasons, and a recall-bounded privacy report
when redaction was requested. See `STATUS.md` for real-data validation limits.
