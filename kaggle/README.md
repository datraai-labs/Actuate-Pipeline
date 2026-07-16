# Running Actuate perception on Kaggle (free T4)

The 4 GB dev card thrashes — SAM2 propagation once took **86 minutes**. A Kaggle **T4 (16 GB,
free 30 h/week)** runs the whole pipeline in minutes and is the only place MoGe-2 and
Video-Depth-Anything fit. This runs perception there, writes the `.rrd` + depth benchmark, and
caches every stage so you **view locally with no GPU** (`actuate viz --cache`).

> ⚠️ **CONSENT — read first.** The capture is **consent-pending human data**. Keep the Kaggle
> dataset **Private**, and delete it when you're done. On Kaggle *you* are the consent boundary.
> The video is **not** face/OCR-redacted here — this is R&D output that never ships.

---

## The dataset (as it actually exists)

The raw dataset mounts read-only at `/kaggle/input/datasets/elon7069/session-001/` with Kaggle's
raw filenames:

```
video (1).mp4              motion (1).json           timestamps (1).json
camera_intrinsic (1).json  metadata (2).json
```

**You do not normalise these by hand.** `run_perception.py` detects a raw dataset (no
`session_meta.json`, spaced/paren filenames), copies the files into a writable working dir,
renames them (`video (1).mp4 → compressed.mp4`, `camera_intrinsic (1).json →
camera_intrinsics.json`, etc.), and **generates `session_meta.json` from the video with OpenCV**
(frame count, fps, resolution). If you instead point `--session` at an already-processed Actuate
session, it's used as-is.

---

## Setup (one-time)

**1. Code onto Kaggle.** The repo is private — upload `src/actuate/`, `pyproject.toml`, `kaggle/`
as a **Private** dataset, or clone with a token in Kaggle *Secrets* (`GH_TOKEN`):

```python
import os
tok = __import__("kaggle_secrets").UserSecretsClient().get_secret("GH_TOKEN")
os.system(f"git clone https://{tok}@github.com/datraai-labs/Actuate.git /kaggle/working/actuate")
%cd /kaggle/working/actuate
```

**2. GPU + Internet on** (Notebook settings → Accelerator **GPU T4 x2**, Internet **on**).

**3. Install the models (one cell):**

```python
%pip install -q -e .
%pip install -q rerun-sdk transformers timm opencv-python-headless
%pip install -q git+https://github.com/warmshao/WiLoR-mini                    # hands
%pip install -q --no-deps git+https://github.com/lpiccinelli-eth/UniDepth      # depth (baseline)
# depth benchmark challengers (optional — needed for --depth-ab rows):
%pip install -q git+https://github.com/DepthAnything/Video-Depth-Anything      # temporal
%pip install -q git+https://github.com/microsoft/MoGe.git                      # metric+focal anchor
```

Weights auto-download from Hugging Face: WiLoR, UniDepthV2, Grounding DINO (`grounding-dino-tiny`),
SAM2 (`sam2-hiera-tiny`), MoGe-2 (`Ruicheng/moge-2-vitl`). **Video-Depth-Anything's checkpoint is
manual** — download the *metric* variant and point `VDA_CKPT` at it:

```python
import os
os.environ["VDA_CKPT"] = "/kaggle/input/vda-metric/metric_video_depth_anything_vits.pth"
```

---

## Run it — exactly this, no extra steps

```python
%cd /kaggle/working/actuate
!python kaggle/run_perception.py \
    --session /kaggle/input/datasets/elon7069/session-001 \
    --max-frames 60 \
    --depth-ab \
    --out /kaggle/working/session_001.rrd
```

It normalises the dataset, runs the pipeline (per-stage `ran`/`cached` timings), builds the
schema-v3 canonical episode, writes the `.rrd`, and runs the depth benchmark.

> **SLAM note:** the camera trajectory needs IMU on the video's frame axis (`session.h5`). The
> raw `motion.json` isn't converted, so SLAM is **skipped gracefully** — depth/hands/objects/
> fusion and the benchmark all still run. (Converting `motion.json` → `session.h5` is a separate
> step if you want the trajectory.)

---

## Get the results back (from the Output tab — no zip)

Everything lands under `/kaggle/working`; download from the notebook's **Output** tab:

```
session_001.rrd                              open locally:  rerun session_001.rrd
depth_ab.txt                                 the depth benchmark table + gate verdict
processed/session_001/.actuate_cache/*.pkl   copy into your local session, then:
```
```powershell
Copy-Item -Recurse .actuate_cache processed\session_001\
actuate viz show processed\session_001 --stages depth,hands,objects,fusion --max-frames 60 --cache
```

Cache keys are computed identically on Kaggle and locally (`sha256(stage|n=<frames>|...)`), so a
cache from the T4 is reused verbatim on your laptop **as long as `--max-frames` matches**.

---

## The depth benchmark (`--depth-ab`)

Scores four depth models on the **same hand keypoints**, with the number that decides everything:
**wrist z-jitter (mm/frame)** — place the wrist at each model's depth (`solve_root_depth` +
hand-cloud fit + temporal smooth) and measure how much the placed wrist z moves frame-to-frame.
That is what becomes the training action, so that is the gate.

- **UniDepthV2** — baseline (single-image).
- **MoGe-2** — per-frame *metric + focal* anchor. Does a better anchor alone help? (our `fx≈660`
  is a guess; MoGe-2 gives an independent focal.)
- **VDA (anchored)** — Video-Depth-Anything (temporal consistency) scale-anchored to the metric
  anchor by one global affine — keeps VDA's low jitter *and* gets metric scale.
- **UniDepth + flow_filter** — cheap optical-flow temporal EMA (does filtering alone help?).

**The gate: < 4 mm/frame smoothed** — a 3× reduction from the *fair* 11.3 mm hand-cloud-fit
baseline (NOT the 20.7 mm bbox strawman). A model can win the static-consistency column yet lose
the wrist column — the wrist *moves*, and a moving articulated hand is the hard case. The report
says so explicitly: **"wins static consistency but NOT the wrist; monocular floor stands"** — a
real finding ("monocular video-depth can't place a moving hand to < 4 mm, stereo is the real
fix"), not a bug. Saved to `depth_ab.txt`.

> At `--max-frames 60` the wrist column is a firmer read than at 20; use 60 for the Phase 4
> go/no-go decision.
```
