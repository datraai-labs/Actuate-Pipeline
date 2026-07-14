# DatraAI Pipeline

Turns egocentric video + IMU recordings of manual manipulation work (bolt
tightening, pick-and-place, packaging, etc.) into labeled, quality-scored
episodes for robot-learning datasets: hand pose, motion primitives, phase
segmentation, task classification, physical-plausibility validation,
natural-language instruction grounding, metric 3D, and a composite quality
certificate — ending in a packaged, deliverable dataset bundle.

**Status:** actively evolving. See [`docs/PIPELINE_STATUS.md`](docs/PIPELINE_STATUS.md)
before assuming any given stage is production-complete — it's the single
source of truth for what's real, what's a stub, and what's not started.

## Quickstart

```bash
pip install -r requirements.txt
```

Run the included example session end-to-end:

```bash
python run_pipeline.py --session raw/session_001
```

Outputs land in `processed/session_001/`. Resume a partially-completed run
(skips steps whose outputs already exist):

```bash
python run_pipeline.py --session raw/session_001 --resume
```

Run every session in a folder as a batch, and package for delivery:

```bash
python run_pipeline.py --batch raw/ --batch-id my_batch_001 --upload
```

## Repository layout

```
config.py               Central config — every threshold/constant, organized by section
run_pipeline.py          Orchestrator: step sequence, --resume, --skip-qa, batch mode
scripts/                 One file per pipeline stage, numbered by execution order
                         (e.g. 04c_object_track.py runs after 04_hand_pose.py,
                         before 04d_depth_estimate.py — the numbering IS the DAG)
utils/                   Shared logic used by multiple stages
                         (video I/O, HDF5 I/O, S3 upload, IMU strategy routing)
tests/                   pytest suite, mirrors stage names (test_<stage>.py),
                         synthetic data only — no GPU/real video required to run it
calibration/             Per-device camera intrinsics (see calibration/README.md)
docs/                    Architecture + pipeline status docs (read these first)
raw/                     Input: {session_id}/raw.mp4 + imu.json|imu.csv
processed/               Output: {session_id}/*.json + session.h5 + compressed.mp4
delivery/                Final packaged batches (what actually ships to customers)
```

## Configuration

Every tunable threshold and constant lives in `config.py`, grouped by
section with inline comments explaining *why* each value is what it is —
that file is the closest thing to a spec for tunable behavior. Two settings
are worth knowing up front since they change which code path a session runs
through: `IMU_SOURCE_MODE` (head-mounted / wrist-mounted / dual / none) and
`DEPTH_MODE` (stereo / monocular / none). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for what each actually does.

## Testing

```bash
python -m pytest tests/ -q
```

The full suite runs in a few seconds — no GPU, no real video files, no
network calls. If you're adding a new pipeline stage, add
`tests/test_<stage>.py` alongside it (synthetic-data fixtures, following the
existing tests' pattern of loading numbered scripts via `importlib` since
`01_ingest.py`-style filenames aren't valid Python module names).

## Docs

- [`docs/PIPELINE_STATUS.md`](docs/PIPELINE_STATUS.md) — stage-by-stage status: done / stub / not started, and every known limitation or "needs GPU validation before trusting this" caveat in one place.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — data flow, storage formats, config-driven branch points.
- [`calibration/README.md`](calibration/README.md) — camera intrinsics format and fallback behavior.

## Contributing a new stage

1. Add `scripts/NN_your_stage.py` (or `NNx_` for a sub-step between two
   numbered stages, matching the `04c`/`04d` convention) with a module
   docstring stating real inputs/outputs, and a `run(session_id) -> dict`
   function following the existing stages' shape.
2. Add any new thresholds/constants to `config.py` — no magic numbers in
   the script itself.
3. Register the step in `run_pipeline.py`'s `PIPELINE_STEPS` list, in
   execution order.
4. Add `tests/test_your_stage.py` with synthetic data.
5. Update `docs/PIPELINE_STATUS.md`.

## License / confidentiality

No `LICENSE` file is included yet — add one reflecting your actual
IP/confidentiality stance (proprietary, internal-only, etc.) before sharing
this repo outside the team.
