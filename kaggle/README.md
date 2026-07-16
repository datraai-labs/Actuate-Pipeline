# Running Actuate perception on Kaggle (free T4)

The 4 GB dev card thrashes — SAM2 propagation once took **86 minutes**. A Kaggle **T4 (16 GB,
free 30 h/week)** runs the whole pipeline in minutes and is the only place FoundationPose and
Video-Depth-Anything fit. This runs perception there and hands you the `.actuate_cache/` files
so you **view the result locally with no GPU** (`actuate viz --cache`).

> ⚠️ **CONSENT — read first.** The capture (`session_001`) is **consent-pending human data**.
> Never commit it, and never make the Kaggle dataset **public**. Upload it as a **Private**
> Kaggle dataset. The consent boundary that protects it lives in S3/IAM, not on Kaggle — so on
> Kaggle *you* are the boundary. Delete the dataset when you're done.

---

## One-time setup

**1. Upload the session as a PRIVATE Kaggle dataset.**
Zip `processed/session_001/` (the `redacted_compressed.mp4`, `session.h5`, `session_meta.json`,
`hand_pose*.json`, `depth_data.json`) and create a **Private** Kaggle dataset from it, e.g.
`actuate-session-001`. It mounts at `/kaggle/input/actuate-session-001/session_001`.

**2. Get the Actuate code onto Kaggle.** The repo is private, so the clean way is to also upload
`src/actuate/`, `pyproject.toml`, and `kaggle/` as a **Private** dataset (`actuate-src`), or
clone with a GitHub token stored in Kaggle *Secrets* (Add-ons → Secrets → `GH_TOKEN`):

```python
# Notebook cell — code via token (private repo)
import os
tok = __import__("kaggle_secrets").UserSecretsClient().get_secret("GH_TOKEN")
os.system(f"git clone https://{tok}@github.com/datraai-labs/Actuate.git /kaggle/working/actuate")
%cd /kaggle/working/actuate
```

**3. Turn the GPU on** (Notebook settings → Accelerator → **GPU T4 x2**), and **Internet on**
(for weight downloads).

---

## Install the models (one cell)

```python
%pip install -q -e .                        # the actuate package (from the cloned/uploaded repo)
%pip install -q rerun-sdk transformers timm opencv-python-headless
# WiLoR (hands) + MANO loader
%pip install -q git+https://github.com/warmshao/WiLoR-mini
# UniDepthV2 — triton has no wheel on some images; --no-deps avoids it (not needed for V2 infer)
%pip install -q --no-deps git+https://github.com/lpiccinelli-eth/UniDepth
# Video-Depth-Anything (the temporal A/B challenger) — optional
%pip install -q git+https://github.com/DepthAnything/Video-Depth-Anything
# MoGe-2 (metric depth + focal anchor) — optional, for the benchmark's anchor row
%pip install -q git+https://github.com/microsoft/MoGe.git
```

Weights that auto-download from Hugging Face on first use: WiLoR, UniDepthV2, Grounding DINO
(`grounding-dino-tiny`), SAM2 (`sam2-hiera-tiny`). **Video-Depth-Anything's checkpoint is
manual** — download the *metric* variant and point `VDA_CKPT` at it:

```python
import os
# e.g. from the VDA release assets; metric variant for metric depth
os.environ["VDA_CKPT"] = "/kaggle/input/vda-metric/metric_video_depth_anything_vits.pth"
```

---

## Run it

```python
%cd /kaggle/working/actuate
!python kaggle/run_perception.py \
    --session /kaggle/input/actuate-session-001/session_001 \
    --max-frames 60 \
    --depth-ab                # add this to A/B UniDepth vs VDA vs flow-filter
```

It prints per-stage timings (`ran`/`cached`), builds the schema-v3 canonical episode, writes
`episode.rrd`, runs the depth A/B, and bundles everything into
**`/kaggle/working/actuate_outputs.zip`**.

---

## Get the results back

Download `actuate_outputs.zip` from the notebook's **Output** tab. Then locally:

```powershell
# unzip its .actuate_cache/ into your session dir
Expand-Archive actuate_outputs.zip -DestinationPath .
Copy-Item -Recurse .actuate_cache processed\session_001\

# now viewing is instant — no GPU, no model re-run (cache keys match the CLI's)
actuate viz show processed\session_001 --stages depth,hands,objects,fusion,slam --max-frames 60 --cache
# or just open the recording that came in the zip:
rerun episode.rrd
```

The cache keys are computed the same way on Kaggle and locally (`sha256(stage|n=<frames>|...)`),
so a cache produced on the T4 is reused verbatim on your laptop **as long as `--max-frames`
matches**. Change the frame count and that stage re-runs (correctly — different inputs).

---

## The depth benchmark (`--depth-ab`)

Scores four depth models on the **same hand keypoints**, with the number that actually matters:
**wrist z-jitter (mm/frame)** — place the wrist at each model's depth (`solve_root_depth` +
hand-cloud fit + temporal smooth) and measure how much the placed wrist z moves frame-to-frame.
That is what becomes the training action, so that is the gate.

- **UniDepthV2** — baseline (single-image).
- **MoGe-2** — a per-frame *metric + focal* anchor. Isolates whether a better anchor alone helps
  (right now our `fx≈660` is a guess; MoGe-2 gives an independent focal).
- **VDA (anchored)** — Video-Depth-Anything (temporal consistency) scale-anchored to the metric
  anchor by a single global affine, so it keeps VDA's low jitter *and* gets metric scale.
- **UniDepth + flow_filter** — cheap optical-flow temporal EMA (shows if filtering alone helps).

**The gate: < 4 mm/frame smoothed** — a 3× reduction from the *fair* 11.3 mm hand-cloud-fit
baseline (NOT the 20.7 mm bbox-pseudo-depth strawman). A model can win the static-consistency
column yet lose the wrist column — the wrist *moves*, and a moving articulated hand is the hard
case. If that happens, the report says so: **"wins static consistency but NOT the wrist;
monocular floor stands"** — which is a real finding ("monocular video-depth can't place a moving
hand to <4 mm, stereo is the real fix"), not a bug. Saved to `depth_ab.txt` in the zip.
```
