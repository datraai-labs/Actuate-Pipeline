# Architecture

How data flows through the pipeline, stage by stage. For *what's actually
implemented vs. stubbed*, see [`PIPELINE_STATUS.md`](./PIPELINE_STATUS.md) —
this document describes the design regardless of build status.

## Data flow

```
raw/{session_id}/
  raw.mp4, imu.json|imu.csv
       │
       ▼
 01_ingest ─────► compressed.mp4, pts.npy, imu_raw.npy, session_meta.json
       │
       ▼
 02_sync ───────► session.h5   (IMU interpolated onto video timestamps)
       │
       ▼
 03_qa ─────────► qa_report.json            (skippable, non-fatal)
       │
       ▼
 03b_privacy_redact ► redacted_compressed.mp4, privacy_report.json
       │               (alongside, not replacing, compressed.mp4 — see below;
       │                skippable/non-fatal, but 11_package refuses to ship
       │                a session with no redacted_compressed.mp4)
       ▼
 04_hand_pose ──► hand_pose.json            (MediaPipe landmarks per frame,
       │                                     reads the UNREDACTED compressed.mp4)
       │
       ▼
 04c_object_track ► object_tracks.json      [STUB — see PIPELINE_STATUS.md §3]
       │
       ▼
 04d_depth_estimate ► depth_data.json, hand_pose_3d.json,
       │               object_tracks.json (enriched with centroid_3d_m)
       ▼
 05_primitives ─► primitives.json           (12-primitive rule engine; each
       │                                     primitive now also carries a
       │                                     per-frame confidence float — §9)
       ▼
 06_phase_segment ► phases.json             (reach/grasp/manipulate/release/idle;
       │                                     each segment carries mean_confidence — §9)
       │
       ▼
 06b_episode_segment ► episodes.json        (splits phases.json on qualifying idle
       │                                     gaps — see "Episode segmentation" below)
       ▼
 07_task_classify ► task_label.json         (L1 task per episode, primitive signature match)
       │
       ▼
 08_validate ────► validation_report.json   (physical plausibility checks — session-wide,
       │                                     NOT restructured per-episode by §6)
       ▼
 09_language_ground ► language_grounding.json  (NL instruction per episode — template
       │                                     by default, or VLM-grounded from sampled
       │                                     video frames — see "VLM-hybrid language
       │                                     grounding" below — §7)
       ▼
 10_eis ─────────► quality_certificate.json (composite quality score + flags, per episode;
       │                                     sync/blur/causal/retargeting checks stay
       │                                     session-wide and shared across episodes;
       │                                     label_confidence is a real weighted
       │                                     average across task + segment confidences — §9)
       ▼
 11_package ─────► delivery/{batch_id}/     (bundled + optional S3 upload;
                                              action_labels.json nested by episode,
                                              each with a confidence_tree — §9)
```

`run_pipeline.py` drives this sequence per session (`PIPELINE_STEPS`), with
`--resume` (skip steps whose outputs already exist) and `--skip-qa` support.
Batch mode (`--batch`) runs every session under a folder through the same
sequence.

## Key config-driven branch points

Three settings in `config.py` change which code path a session runs through
without touching any script:

| Setting | Values | Effect |
|---|---|---|
| `IMU_SOURCE_MODE` | `head_mounted` / `wrist_mounted` / `dual` / `none` | Selects the primitive-detection strategy in `utils/imu_source_router.py`. A head-mounted IMU cannot see hand/finger motion — `VisionPrimaryStrategy` derives wrist rotation from hand-pose landmarks instead of gyro; `WristPrimaryStrategy` is the original gyro/accel-based approach; `FusionStrategy` confidence-weights both. |
| `DEPTH_MODE` | `stereo` / `monocular_estimated` / `none` | Selects the depth source in `scripts/04d_depth_estimate.py`. `stereo` reads a calibrated depth stream if present, falling back to `monocular_estimated` if not. `none` skips depth entirely and marks the session `retargeting_eligible: false`. |
| `LANGUAGE_GEN_MODE` | `template` / `vlm` / `hybrid` | Selects the instruction-generation path in `scripts/09_language_ground.py`. `template` (default) — free, structured-facts-only, no API calls. `vlm` — grounds generation in real sampled video frames via Claude, raises on API failure. `hybrid` — same as `vlm` but falls back to `template` per-episode on any API error. See "VLM-hybrid language grounding" below. |

## Privacy redaction target: why two video files

`03b_privacy_redact.py` writes `redacted_compressed.mp4` **alongside**
`compressed.mp4`, never overwriting it. This is deliberate: perception
stages (`04_hand_pose.py`, `04c_object_track.py`, `04d_depth_estimate.py`)
keep reading the original, unredacted `compressed.mp4` for maximum
tracking fidelity — a blurred face/badge region is a data-quality risk to
hand/object tracking we don't need to take internally, and redaction only
matters for what actually leaves the building. Only `scripts/11_package.py`
substitutes the redacted copy (renamed to `compressed.mp4` in the delivered
bundle) — and it does so with a fail-closed check: a session missing
`redacted_compressed.mp4` is excluded from the batch entirely rather than
silently shipping the unredacted original.

**Consent gate:** independent of redaction quality, `run_pipeline.py` reads
`session_meta.json`'s `consent_status` (written by `01_ingest.py`,
defaulting to `"pending"` — never assumed granted) before invoking
`11_package.py`. Anything other than `"granted"` blocks packaging for that
session, in both the per-session and batch-upload code paths
(`config.BLOCK_DELIVERY_WITHOUT_CONSENT`).

## Storage formats

**`session.h5`** (written by `utils/hdf5_writer.py`):

```
session.h5/
├── video/
│   ├── timestamps       [N_frames] float64, absolute epoch seconds
│   └── pts_relative     [N_frames] float64, seconds from stream start
├── imu/
│   ├── timestamps       [N_frames] float64, same axis as video
│   ├── accel            [N_frames, 3] float32: ax, ay, az (m/s²)
│   └── gyro             [N_frames, 3] float32: gx, gy, gz (rad/s)
└── metadata             JSON string: session_meta + sync_stats
```

**Per-frame JSON files** (`hand_pose.json`, `primitives.json`,
`object_tracks.json`, `depth_data.json`, `hand_pose_3d.json`) are all flat
JSON arrays, one entry per video frame, each entry carrying at minimum a
`frame_idx`. This lets any stage load only the fields it needs and keeps
frame alignment trivial to check (`len(a) == len(b)`).

**`episodes.json`** (written by `scripts/06b_episode_segment.py`):

```json
{
  "session_id": "...",
  "episode_gap_threshold_sec": 8.0,
  "min_episode_duration_sec": 2.0,
  "episodes": [
    {"episode_id": "{session_id}_ep00", "start_frame": 0, "end_frame": 2849,
     "start_sec": 0.0, "end_sec": 94.9667, "duration_sec": 94.9667}
  ]
}
```

Every per-episode stage (`07_task_classify.py`, `09_language_ground.py`,
`10_eis.py`, `11_package.py`) loads this via `utils/episode_utils.load_episodes()`
and filters its per-frame/per-segment inputs to each episode's
`[start_frame, end_frame]` range with `filter_frames()` (frame-indexed
data) or `filter_segments()` (`phases.json`-style ranges, overlap-based).
`load_episodes()` raises `FileNotFoundError` if `episodes.json` is
missing — it's a required input for every per-episode stage, not optional.

## Episode segmentation (v2 §6)

`06b_episode_segment.py` splits one session's `phases.json` segments into
one or more episodes wherever an `idle` segment's duration exceeds
`config.EPISODE_GAP_THRESHOLD_SEC` (default 8.0s) — the boundary-scanning
loop that does this treats "no qualifying gap found" and "one qualifying
gap found" the same way, so a session with zero long idle gaps (the most
common real shape) naturally falls out as exactly one episode spanning the
whole session, not zero episodes or a truncated range. Episode groups
shorter than `config.MIN_EPISODE_DURATION_SEC` (default 2.0s) are dropped
as noise fragments; if every candidate group is too short, the function
falls back to treating the entire session as one episode rather than
producing an empty `episodes.json`.

Downstream, `08_validate.py` was deliberately **not** restructured
per-episode — its checks (causal ordering, physical plausibility) run once
per session, and `10_eis.py` treats that output (along with sync drift,
blur, retargeting eligibility, and IMU-mount plausibility) as session-wide
facts shared across every episode's score. Only `hand_presence_rate` and
`label_confidence` are computed per-episode in `10_eis.py`. This split
mirrors which upstream checks actually operate on frame ranges vs. whole
recordings — extending §6 to `08_validate.py` would be additional scope,
not a gap in this section.

## Confidence propagation (v2 §9)

Every primitive detector in `utils/imu_source_router.py` has a companion
confidence function in `utils/confidence.py` — a float in [0,1] measuring
CERTAINTY in that boolean call either way, not "how likely True". A
measurement sitting exactly at its decision threshold is a coin flip
regardless of which side it landed on (confidence near 0); a measurement
far from the threshold — in either direction — is confident (near 1).

```
primitive-level (05_primitives.py)
  margin over/under threshold (_margin_confidence)         — every primitive
  × landmark/tracking confidence (_landmark_confidence,      — vision-derived
    _nearest_object — hand_pose.json / object_tracks.json)     primitives
  × IMU signal-to-noise (_snr_factor — peak vs. window std)  — gyro/accel-window
                                                                 primitives
        │  primitive_confidences: {...} per frame in primitives.json
        ▼
segment-level (06_phase_segment.py)
  mean_confidence = mean over a segment's own frames' confidence
  in whichever primitive(s) justify that phase label (PHASE_RELEVANT_PRIMITIVES)
        │  mean_confidence per segment in phases.json
        ▼
episode-level (10_eis.py)
  label_confidence = frame-count-weighted average of the task-classification
  score (weighted by the episode's total segment frame count) and every
  overlapping segment's own mean_confidence (weighted by that segment's
  frame count) — _compute_label_confidence
        │  components.label_confidence.score in quality_certificate.json
        ▼
delivery (11_package.py)
  confidence_tree per episode: task_level_confidence, segment_level_confidences
  (phase + frame range + mean_confidence per segment), frame_level_available: true
  (frame detail already lives in that episode's own L3_primitives entries)
```

Each strategy (`WristPrimaryStrategy` / `VisionPrimaryStrategy` /
`FusionStrategy`) implements its own `compute_confidences()`, mirroring how
it derives the boolean in `detect_primitives()` — e.g. `FusionStrategy`
weights vision-vs-IMU confidence the same way it weights the boolean vote
(`config.IMU_FUSION_WEIGHT_VISION` / `IMU_FUSION_WEIGHT_IMU`).
`compute_confidences()` is a **separate call**, not merged into
`detect_primitives()`'s return dict — some existing tests assert
`set(result.keys()) == set(ALL_PRIMITIVES)` on `detect_primitives()`, and
keeping the two calls independent leaves that contract untouched.

Two known simplifications (documented inline in `utils/confidence.py`):
vision-only contact-onset/release confidence (no `object_tracks.json`
wired in yet) falls back to a fixed moderate baseline discounted by
landmark confidence, since there's no continuous geometric quantity to
score without real object tracking (§3) — this improves automatically once
§3 lands. And `WristPrimaryStrategy`'s `detect_contact_release`/`detect_idle`
decision thresholds are literals inside those functions (not named
`config.py` constants) — the matching confidence functions mirror those
same literals directly rather than introducing new config knobs that could
drift out of sync with the tuned detection thresholds.

## VLM-hybrid language grounding (v2 §7, revised)

The original addendum §7 spec generated instructions purely from
structured facts (task label, grasp type, dominant hand, etc.). This
revision grounds generation in the actual video instead — structured
facts become supporting context, not the sole input, and the model's own
independent read of the episode becomes a cross-check on the upstream
task classifier rather than a second copy of its output.

```
scripts/06_phase_segment.py's phases.json
        │
        ▼
utils.vlm_language.sample_representative_frames(episode, phase_segments)
  — episode start, 1-2 interior frames from grasp/active_manipulation/
    release segments (phase TRANSITIONS, not fixed intervals), episode end
  — 3-5 frames total (config.VLM_MIN/MAX_SAMPLE_FRAMES)
        │
        ▼
utils.vlm_language.encode_frame_base64() per sampled frame
  — reads compressed.mp4 directly via cv2, JPEG-encodes to base64
        │
        ▼
utils.vlm_language.generate_instruction_vlm(client, structured_facts,
  frame_images_b64, recent_instructions)
  — Claude vision call (config.VLM_MODEL), structured output
    (output_config.format json_schema): instruction text + task_guess +
    task_guess_confidence + objects_mentioned
        │
        ├──► utils.vlm_language.check_task_disagreement(task_guess, ...)
        │      — VLM's independent task read vs. 07_task_classify.py's
        │        label for the same episode → task_classification_disagreement
        │
        └──► utils.vlm_language.should_spotcheck(episode_id, rate) — gate
               │  (deterministic per episode_id, not random)
               ▼ (for the sampled fraction only)
             utils.vlm_language.check_instruction_hallucination(client,
               instruction, frame_images_b64)
               — second, independent VLM call: does the generated text
                 describe only what's visible in these same frames?
               → hallucination_check.consistent / .unsupported_claims
```

`scripts/09_language_ground.py`'s `run()` selects the path via
`config.LANGUAGE_GEN_MODE`:

- **`template`** (default): `_ground_episode()` — the pre-§7 pure,
  structured-facts-only templating. No network, no cost.
- **`vlm`**: `_ground_episode_vlm()` for every episode. Any API failure
  (network, rate limit, auth, malformed response) propagates up and
  `run()` raises — no silent degradation.
- **`hybrid`**: tries `_ground_episode_vlm()`, and on any exception falls
  back to `_ground_episode()` for that episode only, marking
  `generation_method: "template_fallback"` and recording
  `vlm_fallback_reason` — logged, never silent, and doesn't abort the rest
  of the session's episodes.

**Cost/latency tracking is not an afterthought.** Every VLM call records
its own `cost_usd` (from `response.usage` via
`config.VLM_PRICING_USD_PER_MTOK`) and `latency_sec` in that episode's
`vlm_audit`; `run()` aggregates a session-level `vlm_cost_summary`
(`avg_cost_per_episode_usd`, `cost_per_hour_of_video_usd`) and prints it
immediately after each session — this exists specifically so a batch run
never gets deep into cost before anyone notices the per-episode number.

**Auditability**: `vlm_audit` stores `frames_sampled` (frame *indices*,
not the images themselves — keeps `language_grounding.json` light),
`prompt_system`, `raw_response`, and `model` for every VLM-generated
episode — the same auditability standard as the template path's
`template_fields`/`template_version`.

**Anti-repetition** — `recent_instructions` (the last few episodes'
generated instructions within the *same* `run()` call, i.e. same session)
is passed into each generation call with an explicit instruction to vary
phrasing/sentence structure. This is session-scoped, not batch-scoped —
`09_language_ground.py` still runs once per session, not once per batch,
so cross-session anti-repetition within one delivery batch isn't covered
yet.

## Primitive detection strategy (v2 §1)

```
config.IMU_SOURCE_MODE
        │
        ▼
get_primitive_strategy()  (utils/imu_source_router.py)
        │
   ┌────┼────────────┬──────────────┐
   ▼    ▼             ▼              ▼
"head_  "wrist_       "dual"        "none"
mounted" mounted"
   │      │             │              │
   ▼      ▼             ▼              ▼
Vision  Wrist        Fusion         Vision
Primary Primary      Strategy       Primary
Strategy Strategy   (runs both,     Strategy
                     confidence-
                     weighted vote,
                     flags
                     disagreement)
```

Every strategy implements the same `detect_primitives(frame_idx, pose_frame,
prev_pose_frame, imu_window, object_track_frame) -> dict[str, bool]`
interface, so `scripts/05_primitives.py`'s orchestration loop (windowing,
temporal smoothing via `apply_minimum_duration_filter`, output shape) never
changes based on which strategy is active.

`check_imu_mount_plausibility()` runs alongside this as an advisory
heuristic — see `PIPELINE_STATUS.md` §1 for what it can and can't catch.

## Glove + per-worker calibration (v2 §2/§5)

Both sections adjust the same two thresholds (`power_grasp_dist`,
`lateral_pinch_dist`) that `detect_power_grasp`/`detect_lateral_pinch` and
their matching `utils/confidence.py` functions compare landmark distances
against. Rather than threading new parameters through every strategy's
`detect_primitives(frame_idx, pose_frame, prev_pose_frame, imu_window,
object_track_frame)` signature (which every strategy and every existing
test depends on), the resolved thresholds ride in the already-generic,
already-per-frame `imu_window` dict:

```
scripts/05_primitives.py's run() — once per session, before the frame loop
        │
        ▼
read session_meta.json's glove_type / worker_id (both optional; missing
entirely -> glove_type="none", worker_id=None -> identical to pre-§2/§5
behavior)
        │
        ▼
utils.glove_profile.resolve_grasp_thresholds(glove_type)
  -> {power_grasp_dist, lateral_pinch_dist} scaled by
     config.GLOVE_THRESHOLD_MULTIPLIERS
        │
        ▼ (only if worker_id is set)
utils.worker_profile_store.load_profile(worker_id)
  -> if a valid, non-expired, consented profile exists, ITS
     power_grasp_dist/lateral_pinch_dist override the glove-adjusted
     values entirely (worker calibration > glove default > raw default)
        │
        ▼
imu_window["power_grasp_dist"], imu_window["lateral_pinch_dist"]
  — set once, read every frame by both WristPrimaryStrategy and
    VisionPrimaryStrategy's detect_primitives()/compute_confidences()
    (FusionStrategy needs no separate change — it delegates to the same
    two sub-strategies, which already read the same imu_window dict)
```

`detect_power_grasp(pose_frame, threshold=None)` and
`detect_lateral_pinch(pose_frame, threshold=None)` (and their
`confidence_*` counterparts) default `threshold=None` to the raw
`config.POWER_GRASP_DIST`/`LATERAL_PINCH_DIST` constants — every existing
call site that doesn't pass a threshold is unaffected, which is why this
required no changes to `detect_primitives()`'s external interface or the
tests asserting `set(result.keys()) == set(ALL_PRIMITIVES)` on it.

**Where each input comes from:**
- `glove_type` — an optional `raw/{session_id}/session_config.json`
  (`{"glove_type": "thin", "worker_id": "worker_042"}`), read by
  `01_ingest.py` and echoed into `session_meta.json`. Mirrors
  `consent.json`'s existing pattern exactly.
- Worker calibration profile — produced by `scripts/00_calibration.py`
  from a short calibration clip (open hand → full fist → lateral pinch),
  stored via `utils/worker_profile_store.py` in
  `calibration/workers/{worker_id}_profile.json`. Fail-closed on consent
  (mirrors §10's delivery gate) and retention-enforced — `load_profile`
  deletes an expired profile on read rather than silently serving stale
  biometric-like data past `config.WORKER_PROFILE_RETENTION_DAYS`.

## Object detection & tracking + object-identity task bonus (v2 §3)

`scripts/04c_object_track.py` replaced its dummy-bbox stub with real
Grounding DINO (open-vocab detection) + SAM2 (mask-based video tracking):

```
04c_object_track.py's run() — per OBJECT_DETECT_SAMPLE_INTERVAL_FRAMES chunk
        │
        ▼
_get_grounding_dino() [cached singleton] .detect on chunk's first frame
        │
        ▼
filter_detections_near_hand() — keep only boxes within
OBJECT_TRACK_RADIUS_NORM of hand_pose.json's dominant-hand landmark
centroid for that frame (not the whole scene)
        │
        ▼
match_or_create_track_ids() — IoU (OBJECT_TRACK_MATCH_IOU_MIN) against
active_tracks from the previous chunk; unmatched detections get a new
track_id
        │
        ▼ (one call per tracked object — see below)
_get_sam2() [cached singleton] .propagate through this chunk's frames,
seeded with that object's box
        │
        ▼
mask_to_bbox_and_centroid() per propagated frame
        │
        ▼
object_tracks.json — flat top-level list, one entry per frame:
[{"frame_idx": i, "stub": false,
  "tracked_objects": [{"track_id", "class_label", "confidence", "bbox",
                        "centroid_norm", "is_stub": false}, ...]}, ...]
```

**Known API constraint (found via live testing against real frames, not
assumed):** SAM2's video session errors when two objects are seeded at the
same `frame_idx` in one session
(`maskmem_features in conditioning outputs cannot be empty when not
is_initial_conditioning_frame`). Rather than guess at the correct
multi-object batching call shape, tracking runs **one SAM2 session per
tracked object** per chunk — simpler and verified-correct, at the cost of
some GPU efficiency versus true batched multi-object tracking.
`processor.post_process_masks()` was also bypassed after it threw a shape
mismatch on real mask output; mask resizing is done manually via
`F.interpolate` + threshold instead.

Consumers (`05_primitives.py`'s contact-onset/release detection,
`04d_depth_estimate.py`'s object-relative depth, `07_task_classify.py`'s
object bonus below) all expect this flat-list shape — confirmed by reading
their actual parsing code before finalizing the output format, not assumed
from the schema alone.

**Object-identity bonus into `07_task_classify.py`:** the primitive
vocabulary alone cannot discriminate `material_transfer`/`box_seal`/
`pick_and_place` (see `config.TASK_SIGNATURES`'s inline comment) — those
three share their entire primitive signature and can only be told apart by
which object is actually being handled. `_dominant_object_class()` finds
the most frequent `class_label` tracked during an episode's
active-primitive frames (any frame with a non-empty `active_primitives`
list — a proxy for "actively manipulating something," since phase
segmentation isn't available at this stage); if it's in a task's
`config.EXPECTED_OBJECT_CLASSES` entry, that task's `match_score` gets
`config.OBJECT_MATCH_BONUS` added **before** the tie/threshold logic runs,
so a real object match can actually break a tie or cross the confidence
threshold, not just annotate a decision already made.

**Status:** wired end-to-end and re-run against real `session_001` footage
(600 frames) — see `docs/PIPELINE_STATUS.md`'s §3 row for the actual
detections found (a "tool" and a "package", consistent with §7's
independent VLM read of this footage as paperwork/stapling) and why the
classifier still correctly returned `unknown`. The bonus mechanism itself
is only demonstrated against a synthetic fixture so far
(`tests/test_task_classify.py::test_bonus_breaks_a_primitive_only_tie`) —
no real session on hand yet exercises it against a genuine
`material_transfer`/`box_seal`/`pick_and_place` tie. See the top-of-file
business-blocker note in `docs/PIPELINE_STATUS.md`.

## Perception source toggle (v2 §8)

`utils/video_utils.py`'s `resolve_perception_source(session_id)` is the
single place that decides which video file a perception script reads,
replacing three separate hardcoded `proc_dir / "compressed.mp4"` lines:

```
config.PERCEPTION_SOURCE ("raw" | "compressed")
        │
        ▼
resolve_perception_source(session_id)
  "compressed" -> processed/{session_id}/compressed.mp4
  "raw"        -> raw/{session_id}/raw.mp4
  raises FileNotFoundError if the resolved path doesn't exist — never
  silently substitutes the other source
        │
        ▼
04_hand_pose.py / 04c_object_track.py / 04d_depth_estimate.py
  (each replaced its own hardcoded compressed.mp4 lookup with a call here)
```

`01_ingest.py` gates raw-video retention on `config.KEEP_RAW_AFTER_COMPRESSION`
(default `True`, matching pre-§8 behavior — raw.mp4 was never deleted
before this section). The gating logic lives in its own function
(`_maybe_delete_raw_video`) specifically so it's unit-testable without
invoking ffmpeg, and it refuses to delete even when
`KEEP_RAW_AFTER_COMPRESSION=False` if `PERCEPTION_SOURCE == "raw"` — this
section otherwise would let the toggle break itself for any session
configured to read the raw file.

## Dataset-level QC (v2 §11)

`scripts/11b_dataset_qc.py` runs after `scripts/11_package.py`, operating
across an entire delivered batch rather than one session/episode — the
first stage where "batch" is the natural unit:

```
delivery/{batch_id}/dataset_manifest.json (from 11_package.py)
        │
        ▼
per-session mid-frame perceptual hash (utils.dataset_qc.phash_frame)
        │
        ▼
dedup_sessions() — pairwise Hamming similarity vs config.DEDUP_SIMILARITY_THRESHOLD;
first occurrence in a near-duplicate cluster kept, later ones flagged
        │
        ▼
compute_task_imbalance_ratio() — max/min episode count across
represented tasks (manifest's existing task_distribution); None (not a
number) when fewer than 2 tasks are represented
        │
        ▼
compute_diversity_summary() — object-class counts from real
object_tracks.json class_label output (§3) + worker_id diversity from
session_meta.json (§2/§5)
        │
        ▼
stratified_split() — sorted episode_id + cumulative ratio boundaries per
config.SPLIT_STRATIFY_BY group (config.TRAIN_VAL_TEST_SPLIT); deterministic,
not RNG-based, so a rerun on the same manifest reproduces the same split
        │
        ▼
dataset_manifest.json extended in place (dedup_removed_count,
task_imbalance_ratio, diversity_summary, splits) +
dataset_qc_report.json (full per-session/per-episode detail)
```

All decision logic lives in `utils/dataset_qc.py`, kept free of file I/O
so it's directly unit-testable against synthetic multi-session fixtures
(`tests/test_dataset_qc.py`) — this repo has exactly one real session
today, so this stage's numbers against real data are structurally
undefined/trivial at batch scale; `dataset_qc_report.json`'s
`single_real_session_caveat` field says so explicitly whenever
`session_count < 2`, rather than presenting a single-session run as
dataset-scale validation. See the business-blocker note in
`docs/PIPELINE_STATUS.md`.

## Taxonomy (primitives, phases, tasks)

Defined entirely in `config.py` (`TASK_SIGNATURES`, `PHASES`,
`INSTRUCTION_TEMPLATES`) — that file is the source of truth for the
labeling vocabulary. Don't duplicate it here; read the comments there when
adding a new task or primitive.
