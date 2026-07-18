# Quickstart: process your first dataset

From an install to a training-ready LeRobot v3 dataset in under 5 minutes and under 10 lines.

## 1. Install

```bash
pip install -e ".[all]"      # from a checkout; `pip install actuate` once published
```

## 2. Set up (one-time)

```bash
actuate login --local        # local mode: runs on your machine, no API key needed
```

## 3. Process a video

```bash
actuate process ./my_video.mp4 --task "pick up cup" --export lerobot_v3 --dataset-out ./dataset/
```

`--rig auto` detects the rig from the video (side-by-side → stereo, otherwise head-mounted).
Drop `--task` and Actuate auto-detects it from a keyframe (needs an API key) or prompts you.

## 4. Check quality

```bash
actuate report ./actuate_runs/my_video_out
```

## 5. View the pipeline

```bash
actuate viz ./actuate_runs/my_video_out
```

## 6. Load in Python

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("actuate/dev", root="./dataset/")
```

That's it — your data is training-ready.

---

## The Python SDK (10 lines)

```python
import actuate

actuate.login()                              # local, one-time
run = actuate.process(
    source="./my_video.mp4",                 # or hf:// s3:// https:// openx://
    rig="auto",
    task="pick up cup",
)
print(run.status, run.quality, run.num_frames)
run.export("lerobot_v3", path="./dataset/")
```

Or the one-liner:

```python
actuate.process_and_export("./my_video.mp4", export_format="lerobot_v3",
                           out="./dataset/", task="pick up cup")
```

## Sources

| Prefix | Reads from | Example |
|---|---|---|
| *(none)* | local file / session dir | `./data/capture.mp4` |
| `hf://` | HuggingFace Hub | `hf://lerobot/pusht` |
| `s3://` | S3 (configured AWS creds) | `s3://bucket/session_001/` |
| `http(s)://` | direct URL / Google Drive | `https://drive.google.com/file/d/…` |
| `openx://` | Open-X-Embodiment (tfds) | `openx://fractal20220817_data` |

## What you get

Every processed episode is a **certified canonical episode**: MANO hands, metric depth,
object tracks, L2 fusion, fine-grained action intervals, a language task + paraphrases +
subtasks, and an L4 quality certificate (1–5) with per-component scores. Export to
**LeRobot v3** or **RLDS/Open-X**, both dual-space (human + retargeted robot).

The consent boundary is **fail-closed at delivery**: local processing marks your own data
`consent=granted`, but `pii_status` stays `pending`, so nothing ships to a customer bucket
until PII review passes — `actuate report` shows this.

## Honest limits

Perception is GPU-heavy; on a 4 GB card cap with `--max-frames 45`. Retargeting needs a
trained arm model (Kaggle GPU) — without it the retarget stage skips and you get human-space
data. See `STATUS.md` for the full what-works-vs-what's-a-demo ledger.
