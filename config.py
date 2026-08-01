"""
DatraAI Pipeline — Central Configuration
All thresholds, paths, taxonomy, and templates in one place.
No magic numbers anywhere else.
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
RAW_DIR = Path("raw")
PROCESSED_DIR = Path("processed")
DELIVERY_DIR = Path("delivery")

# ═══════════════════════════════════════════════════════════
# VIDEO
# ═══════════════════════════════════════════════════════════
FFMPEG_CRF = 23
FFMPEG_PRESET = "veryfast"
FFMPEG_CODEC = "libx265"
TARGET_FPS = 30

# ═══════════════════════════════════════════════════════════
# IMU
# ═══════════════════════════════════════════════════════════
# Informational only (no code consumes it). Was 200 — stale: the real sensor
# measures 574.6 Hz over the one real session (54,590 samples / 95.0 s,
# audit 2026-08-01). Per-session truth lives in session_meta.imu_hz_measured.
IMU_HZ = 575

# Applied to max_drift_CLEAN_ms (nearest-sample distance over frames NOT
# spanning a raw dropout — 02_sync Step 3c). The audit showed the conflated
# metric hit 3.94 ms on the real session purely from one 7.97 ms dropout gap;
# the clean metric measures 1.05 ms there, so 2.0 ms is kept as the alignment
# gate and stream health is gated separately (SYNC_MAX_MISSING_FRACTION).
# PROVISIONAL: reasoned against one real session, not calibrated across rigs.
SYNC_DRIFT_THRESHOLD_MS = 2.0
ACCEL_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]

# A video frame whose enclosing raw-IMU gap exceeds this multiple of the median
# inter-sample interval gets its interpolated value flagged as fabricated-
# across-dropout (imu/interpolated_over_dropout in session.h5). 3.0 on the real
# 574.6 Hz stream (median dt 1.733 ms) flags gaps > 5.2 ms — the 15 genuine
# dropout gaps (5–8.5 ms) in session_001's stream — while leaving ordinary
# transport jitter (p99 = 2.86 ms) unflagged.
IMU_DROPOUT_GAP_FACTOR = 3.0

# QA gate on raw-IMU stream health: fail the sync check when more than this
# fraction of expected samples is missing (dropout accounting from 02_sync).
# The drift metric alone can't see stream health — a heavily-dropping sensor
# can still interpolate to a small nearest-sample distance. PROVISIONAL: the
# one real session measures 1.9% (1,053 of ~55,643); 5% is a reasoned ceiling,
# not a value calibrated across sensors.
SYNC_MAX_MISSING_FRACTION = 0.05

# ═══════════════════════════════════════════════════════════
# QA THRESHOLDS
#
# PROVENANCE (audit 2026-08-01): these four values are the v1 spec's initial
# reasoned guesses. None has been calibrated against real footage — no real
# session has ever been confirmed to fail each check for the right reason.
# QA_THRESHOLDS_PROVISIONAL below is surfaced in every qa_report.json so a
# qa_score is not read as calibrated evidence. Recalibrate against a labeled
# real capture set, then record the derivation here and flip the flag.
# ═══════════════════════════════════════════════════════════
QA_THRESHOLDS_PROVISIONAL = True
BLUR_THRESHOLD = 80.0            # Laplacian variance minimum (v1 guess, uncalibrated)
COVERAGE_THRESHOLD = 0.5         # optical flow mean minimum (v1 guess, uncalibrated)
FPS_STD_THRESHOLD_MS = 5.0       # frame interval stdev maximum (v1 guess, uncalibrated)

# Master Spec §L0 fps-bug gate: measured video cadence vs metadata's claimed
# fps_nominal must agree within this relative tolerance; beyond it the metadata
# is provably wrong and the session fails QA rather than being silently trusted.
FPS_MATCH_RTOL = 0.05
HAND_PRESENCE_RATE_MIN = 0.6     # hands visible in >= 60% of active frames

# ═══════════════════════════════════════════════════════════
# PRIMITIVE THRESHOLDS
# ═══════════════════════════════════════════════════════════
PRONATE_GYRO_Z_DEG_S = 15.0
SUPINATE_GYRO_Z_DEG_S = -15.0
FLEX_GYRO_X_DEG_S = 10.0
CONTACT_ACCEL_G = 1.2
REACH_WRIST_VEL = 0.3
POWER_GRASP_DIST = 0.15          # normalized fingertip distance
LATERAL_PINCH_DIST = 0.08        # thumb-index normalized distance
MIN_PRIMITIVE_FRAMES = 3         # minimum consecutive frames to confirm primitive

# contact_onset/contact_release mark an instantaneous state TRANSITION (the
# single frame where grasp/contact flips), not a sustained condition like
# "idle" or "power_grasp" — a real event is definitionally exactly 1 frame
# wide under both strategies' edge-detection logic (see
# VisionPrimaryStrategy._detect_contact_vision's grasp-transition fallback,
# and WristPrimaryStrategy's use of detect_contact_release's own history
# lookup). Applying MIN_PRIMITIVE_FRAMES=3 to these two would erase every
# real event outright — confirmed against real session_001 footage, where
# 23 genuine raw contact_onset detections (every one exactly 1 frame long)
# were being silently smoothed away to zero, which in turn made
# bolt_tightening's task-signature score fail its contact_onset>=3
# requirement it should have passed. These two primitives get their own,
# much shorter minimum-duration filter instead of the shared one.
CONTACT_EVENT_PRIMITIVES = ("contact_onset", "contact_release")
CONTACT_EVENT_MIN_FRAMES = 1     # effectively a no-op filter — any real 1-frame event survives

# ═══════════════════════════════════════════════════════════
# IMU SOURCE MODE (v2 addendum §1)
#
# A head-mounted IMU measures head/mount motion, not hand or finger motion —
# it cannot capture finger curl, grasp aperture, or wrist rotation. A
# wrist-mounted IMU (strap/glove) measures those directly, as in v1.
# ═══════════════════════════════════════════════════════════
IMU_SOURCE_MODE = "head_mounted"   # one of: "head_mounted", "wrist_mounted", "dual", "none"

HEAD_IMU_THRESHOLDS = {
    "IDLE_ACCEL_MAG_MAX": 0.15,
    "IDLE_GYRO_MAG_MAX": 5.0,          # deg/s, whole-head stillness
    "ACTIVE_MOTION_ACCEL_MIN": 0.25,
}

# Wrist-mounted / fine-motor IMU thresholds. Mirrors the flat constants above
# (kept as the single source of truth so existing call sites / tests that
# reference cfg.PRONATE_GYRO_Z_DEG_S etc. directly are unaffected) — this
# dict is the consumption point for utils/imu_source_router.py.
WRIST_IMU_THRESHOLDS = {
    "PRONATE_GYRO_Z_DEG_S": PRONATE_GYRO_Z_DEG_S,
    "SUPINATE_GYRO_Z_DEG_S": SUPINATE_GYRO_Z_DEG_S,
    "FLEX_GYRO_X_DEG_S": FLEX_GYRO_X_DEG_S,
    "CONTACT_ACCEL_G": CONTACT_ACCEL_G,
}

IMU_FUSION_WEIGHT_VISION = 0.7   # when both vision + wrist IMU available, vision dominates
IMU_FUSION_WEIGHT_IMU = 0.3
# How close a confidence-weighted fusion vote must be to the 50/50 decision
# boundary before a vision/IMU disagreement is considered "unresolved" and
# flagged for review (rather than confidently settled by the weight split).
IMU_FUSION_DISAGREEMENT_MARGIN = 0.3

# ═══════════════════════════════════════════════════════════
# IMU MOUNT PLAUSIBILITY CHECK (advisory only)
#
# The pipeline ingests a single physical IMU stream and TRUSTS
# IMU_SOURCE_MODE to correctly describe where it's mounted. If the hardware
# is actually misconfigured (e.g. a wrist strap feeding a session labeled
# "head_mounted"), nothing in ingest/sync can catch that directly — there is
# no second, independent stream to cross-check against. This is a
# best-effort heuristic on gross motion characteristics, not a
# ground-truth check: it can miss real mismatches and can false-positive on
# legitimate edge-case motion (e.g. a worker who rapidly turns their head).
# It only ever flags for human review — it never blocks the pipeline or
# changes primitive-detection behavior.
# ═══════════════════════════════════════════════════════════
IMU_MOUNT_PLAUSIBILITY_CHECK_ENABLED = True
IMU_MOUNT_WRIST_SNAP_LIKE_DEG_S = 150.0           # gyro magnitude implausible for head motion
IMU_MOUNT_WRIST_SNAP_FRACTION_THRESHOLD = 0.02    # >2% of frames this fast -> "head_mounted" looks like a wrist stream
IMU_MOUNT_HEAD_STILLNESS_PEAK_DEG_S = 20.0        # peak gyro implausibly low for an active wrist IMU
IMU_MOUNT_HEAD_STILLNESS_SPIKE_FRACTION_MAX = 0.001

# ═══════════════════════════════════════════════════════════
# TASK TAXONOMY
# ═══════════════════════════════════════════════════════════
TASKS = [
    "bolt_tightening",
    "material_transfer",
    "box_seal",
    "label_apply",
    "inspection_visual",
    "pick_and_place",
    "tool_change",
    "idle",
    "assembly_insert",
    "packaging_fold",
]

PHASES = ["reach", "grasp", "active_manipulation", "release", "reposition", "idle"]

TASK_SIGNATURES = {
    "bolt_tightening":    {"wrist_pronate": 50, "power_grasp": 30, "contact_onset": 3},
    "material_transfer":  {"transport": 80, "power_grasp": 20, "reach_onset": 15},
    "box_seal":           {"transport": 30, "finger_extend": 40, "contact_onset": 3},
    "label_apply":        {"lateral_pinch": 30, "transport": 20, "finger_extend": 15},
    "inspection_visual":  {"idle": 100, "reach_onset": 10, "finger_extend": 10},
    "pick_and_place":     {"reach_onset": 20, "power_grasp": 20, "transport": 30},
    "tool_change":        {"power_grasp": 20, "contact_release": 5, "contact_onset": 5},
    "assembly_insert":    {"wrist_flex": 30, "power_grasp": 20, "contact_onset": 5},
    "packaging_fold":     {"finger_curl": 50, "finger_extend": 30, "transport": 10},
}
# Every signature now requires >=3 distinct primitives (v2 addendum
# follow-up, 2026-07-06). The original 2-primitive signatures
# (material_transfer, box_seal, label_apply, inspection_visual,
# packaging_fold) let a SINGLE required primitive, fully satisfied alone
# with every other required primitive completely absent, average to
# exactly 0.5 — sitting right at the classifier's confidence threshold.
# Confirmed via synthetic discriminability testing (tests/test_task_classify.py):
# `power_grasp=20` alone (zero transport) confidently fired
# `material_transfer`; `lateral_pinch`/`idle`/`finger_curl` alone did the
# same for their respective tasks. A missing 1-of-3 component now averages
# to 0.667 — still not perfect (see note below) but a missing 2-of-3
# (i.e. only one generic primitive present) now averages to 0.333, safely
# under the 0.5 threshold. This does NOT achieve full primitive-level
# exclusivity per task — only 6 of the 12 primitives have a task-exclusive
# "anchor" (wrist_pronate->bolt_tightening, lateral_pinch->label_apply,
# idle->inspection_visual, wrist_flex->assembly_insert,
# finger_curl->packaging_fold, contact_release->tool_change); the
# remaining three tasks (material_transfer, box_seal, pick_and_place) share
# their entire primitive vocabulary with other tasks and rely on count
# magnitude + combination to discriminate, which is a real, accepted
# limitation of a 12-primitive vocabulary covering 9 tasks — not something
# a threshold/combination tweak alone can fully resolve. This has been
# verified with SYNTHETIC fixtures only (see TestSynthetic... classes in
# tests/test_task_classify.py) — it has NOT been validated against real
# footage, since no real session confirmed to match any of these tasks
# currently exists in this repo (see docs/PIPELINE_STATUS.md's
# "BUSINESS BLOCKER" note at the top of that file).
#
# §3 IS NOW WIRED (2026-07-06) — object-identity bonus, not more threshold
# tuning: material_transfer/box_seal/pick_and_place overlapped because
# they were distinguished only by WHICH primitives fire and HOW MANY,
# never by WHAT OBJECT the worker is holding — information the primitive
# vocabulary alone structurally cannot carry. scripts/04c_object_track.py
# now runs real Grounding DINO + SAM2 detection/tracking, and
# _classify_episode (scripts/07_task_classify.py) adds OBJECT_MATCH_BONUS
# to a task's match_score when the most frequently tracked object's
# class_label during that episode's active-manipulation frames is in that
# task's EXPECTED_OBJECT_CLASSES list. This directly targets the residual
# overlap documented above — it does not eliminate it (a session
# genuinely lacking a clear, consistently-tracked object still falls back
# to primitive-only scoring), but gives the classifier a second,
# independent signal the three overlapping tasks previously had no access
# to at all.
EXPECTED_OBJECT_CLASSES = {
    "bolt_tightening":   ["bolt"],
    "material_transfer": ["box", "package", "bin", "component"],
    "box_seal":          ["box"],
    "label_apply":       ["label", "box"],
    "inspection_visual": ["component", "tool"],
    "pick_and_place":    ["component", "bin"],
    "tool_change":       ["tool"],
    "assembly_insert":   ["component"],
    "packaging_fold":    ["box", "package"],
}
# Added to a task's averaged primitive-signature score (before the
# existing 0.0-1.0 clamp) when its expected object class matches the
# episode's dominant tracked object. Deliberately modest relative to a
# full primitive-signature match (1.0) — this is a tie-breaking nudge
# between plausible candidates, not a signal strong enough to override a
# session whose primitives don't fit the task at all.
OBJECT_MATCH_BONUS = 0.15

# A confident-looking top task-signature score can still be a genuine tie
# — >=2 tasks landing within this margin of the max score is treated as
# ambiguous (falls back to "unknown" / needs_human_review, same as a low
# single-task score) rather than silently picking one by dict-insertion
# order in TASK_SIGNATURES. Found via real data: session_001 scored 5
# tasks tied at 1.0, and the previous code picked whichever came first in
# this dict — a confident-looking but arbitrary (and wrong) label.
TASK_TIE_MARGIN = 0.05

# ═══════════════════════════════════════════════════════════
# LANGUAGE TEMPLATES
# ═══════════════════════════════════════════════════════════
INSTRUCTION_TEMPLATES = {
    "bolt_tightening": (
        "Pick up the bolt using {grasp_type} with the {dominant_hand} hand, "
        "position it over {target_location}, and tighten clockwise until seated. "
        "Task {outcome}. Duration: {duration:.1f}s."
    ),
    "material_transfer": (
        "Grasp {object_class} using {grasp_type} with the {dominant_hand} hand "
        "and transfer it to {target_location}. "
        "Task {outcome}. Duration: {duration:.1f}s."
    ),
    "pick_and_place": (
        "Pick up the object from the source location using {grasp_type} with the "
        "{dominant_hand} hand and place it at {target_location}. Duration: {duration:.1f}s."
    ),
    "assembly_insert": (
        "Insert the component into the assembly fixture using {grasp_type} with the "
        "{dominant_hand} hand. Task {outcome}. Duration: {duration:.1f}s."
    ),
    "box_seal": (
        "Seal the box by applying tape using {grasp_type} with the {dominant_hand} hand. "
        "Task {outcome}. Duration: {duration:.1f}s."
    ),
    "label_apply": (
        "Apply the label using {grasp_type} with the {dominant_hand} hand onto "
        "{target_location}. Task {outcome}. Duration: {duration:.1f}s."
    ),
    "inspection_visual": (
        "Visually inspect {object_class} at {target_location} using the {dominant_hand} hand. "
        "Task {outcome}. Duration: {duration:.1f}s."
    ),
    "tool_change": (
        "Replace the current tool with a new one using {grasp_type} with the {dominant_hand} hand. "
        "Task {outcome}. Duration: {duration:.1f}s."
    ),
    "packaging_fold": (
        "Fold packaging material using {grasp_type} with the {dominant_hand} hand "
        "and place at {target_location}. Task {outcome}. Duration: {duration:.1f}s."
    ),
    "idle": (
        "Idle period with no active manipulation. Duration: {duration:.1f}s."
    ),
    "default": (
        "Perform {task} task using {dominant_hand} hand with {grasp_type}. "
        "Duration: {duration:.1f}s."
    ),
}

# ═══════════════════════════════════════════════════════════
# EIS WEIGHTS
# ═══════════════════════════════════════════════════════════
SYNC_WEIGHT = 0.30
BLUR_WEIGHT = 0.20
HAND_PRESENCE_WEIGHT = 0.20
LABEL_CONFIDENCE_WEIGHT = 0.15
CAUSAL_CHECK_WEIGHT = 0.15

# ═══════════════════════════════════════════════════════════
# TASK-PHASE CONSISTENCY EXPECTATIONS
# ═══════════════════════════════════════════════════════════
TASK_PHASE_EXPECTATIONS = {
    "bolt_tightening":    {"active_manipulation": 0.30},
    "material_transfer":  {"reposition": 0.20},
    "box_seal":           {"active_manipulation": 0.20},
    "label_apply":        {"active_manipulation": 0.15},
    "inspection_visual":  {"idle": 0.30},
    "pick_and_place":     {"reach": 0.10, "grasp": 0.10},
    "tool_change":        {"release": 0.05},
    "assembly_insert":    {"active_manipulation": 0.25},
    "packaging_fold":     {"active_manipulation": 0.20},
    "idle":               {"idle": 0.50},
}

# ═══════════════════════════════════════════════════════════
# AWS (read from environment, never hardcode)
# ═══════════════════════════════════════════════════════════
S3_BUCKET = os.environ.get("DATRAAI_S3_BUCKET", "datraai-processed")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

# ═══════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════
PIPELINE_VERSION = "datraai-pipeline-v1.0"

# QA COMPOSITE WEIGHTS
QA_BLUR_WEIGHT = 0.30
QA_COVERAGE_WEIGHT = 0.20
QA_FPS_WEIGHT = 0.20
QA_SYNC_WEIGHT = 0.30

# PHASE SMOOTHING
PHASE_GAP_FILL_MAX_FRAMES = 5
PHASE_MIN_DURATION_FRAMES = 10

# VALIDATION
VALIDATION_SPIKE_RATE_MIN = 0.7
VALIDATION_DECEL_RATE_MIN = 0.6
VALIDATION_CAUSAL_INVERSION_MAX_RATE = 0.3
CAUSAL_MIN_OFFSET_MS = -50.0
CAUSAL_MAX_OFFSET_MS = 200.0

# ═══════════════════════════════════════════════════════════
# OBJECT DETECTION & TRACKING (v2 addendum §3)
#
# scripts/04c_object_track.py now runs real open-vocabulary detection
# (Grounding DINO, via HuggingFace transformers' zero-shot object
# detection pipeline) seeded from config.OBJECT_DETECT_PROMPT_LIST, with
# SAM2 (transformers' Sam2VideoModel) propagating each detection's mask
# across subsequent frames as a track rather than re-detecting from
# scratch every frame. Detections are filtered to within
# OBJECT_TRACK_RADIUS_NORM of the dominant hand — this tracks what the
# worker is holding/reaching for, not the whole scene.
# ═══════════════════════════════════════════════════════════
OBJECT_DETECT_PROMPT_LIST = ["bolt", "box", "label", "tool", "component", "package", "bin"]
OBJECT_TRACK_RADIUS_NORM = 0.3
HAND_OBJECT_CONTACT_DIST_NORM = 0.06   # fingertip-to-object-centroid distance for contact

# Grounding DINO (open-vocabulary detection) — re-detects on this cadence
# rather than every frame (expensive on top of SAM2 tracking every frame
# already); SAM2 propagates each detected mask forward between samples.
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_BOX_THRESHOLD = 0.35    # min detection confidence to accept a box proposal
GROUNDING_DINO_TEXT_THRESHOLD = 0.25   # min text-prompt-match confidence
OBJECT_DETECT_SAMPLE_INTERVAL_FRAMES = 30  # re-run Grounding DINO this often to catch new objects

# SAM2 (mask-based video tracking) — propagates each Grounding-DINO-seeded
# detection's mask forward frame-by-frame until the next detection sample.
SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"

# How many objects per SAM2 video session to seed in one propagation pass.
# Seeding all hand-near objects into a SINGLE session removes the per-object
# re-encode overhead. Keep at 1 to reproduce the old single-object-per-session
# behaviour; raise to match however many objects are expected near the hand
# (rarely more than 3-4 in factory footage). Kaggle T4 can safely use 4.
SAM2_OBJECTS_PER_SESSION = 4

# When a newly re-detected box (at the next sample interval) overlaps an
# existing track above this IoU, it's treated as the same object
# continuing (not a new track_id) — below it, a new track starts.
OBJECT_TRACK_MATCH_IOU_MIN = 0.3

# Deliberately no STUB_* knobs left — scripts/04c_object_track.py no
# longer has a stub code path (v2 addendum §3, 2026-07-06 build). Every
# tracked object's "is_stub" field is always False and the file's
# top-level "stub" field is always False — kept as a field (not removed)
# specifically so any code still checking it fails safe (treats it as
# "not a stub," which is now actually true) rather than KeyError-ing.

# ═══════════════════════════════════════════════════════════
# METRIC 3D — DEPTH (v2 addendum §4)
# ═══════════════════════════════════════════════════════════
DEPTH_MODE = "stereo"   # one of: "stereo", "monocular_estimated", "none"

# Per-device (not per-session) camera intrinsics. {device_id} is read from
# session_meta.json's optional "device_id" field, defaulting to "default"
# if absent. If the file doesn't exist, a documented pinhole approximation
# is derived from video resolution + CAMERA_DEFAULT_HFOV_DEG instead of
# failing — flagged in depth_data.json's "intrinsics_source" field either
# way so it's auditable which sessions used real calibration.
CAMERA_INTRINSICS_PATH = "calibration/{device_id}_intrinsics.json"
CAMERA_DEFAULT_HFOV_DEG = 82.0   # approximation only, used when no calibration file exists

DEPTH_CONFIDENCE_MULTIPLIER = {"stereo": 1.0, "monocular_estimated": 0.6, "none": 0.0}
DEPTH_STORAGE_MODE = "sparse"  # "sparse" = depth only at hand/object keypoints, "dense" = full map

# Dense depth maps are expensive to store; gate them behind an explicit
# opt-in rather than DEPTH_STORAGE_MODE alone, so a stray config flip
# doesn't silently start writing full depth maps for every session.
ENABLE_DENSE_DEPTH_MAPS = False

# Real, open-weights metric monocular depth model (HuggingFace
# transformers `depth-estimation` pipeline). Outputs METRIC depth directly
# (meters), sidestepping the scale-ambiguity problem that a relative-depth
# model would need ego-motion to resolve — see
# scripts/04d_depth_estimate.py's module docstring for the ego-motion hook
# this leaves in place regardless.
# NOTE: verify this is still the current best open-weights option before
# relying on it — model landscape moves fast; this was current as of this
# implementation pass, not independently re-verified against latest SOTA.
MONOCULAR_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# Number of frames fed to the monocular depth pipeline in a single batch
# call. Transformers pipelines accept a list/generator of PIL images and
# drive the GPU kernel once per batch rather than once per frame — this is
# the fix for the "you seem to be using the pipelines sequentially on GPU"
# warning produced by the previous one-frame-at-a-time loop.
#   4 GB VRAM (RTX 2050 / local):  4–8  frames  per batch
#   16 GB VRAM (Kaggle T4):        16–32 frames per batch
# Set to 1 to reproduce the old single-frame behaviour for debugging.
DEPTH_ESTIMATION_BATCH_SIZE = 8

# Sessions with no depth data are excluded from "VLA_finetuning" in
# recommended_use (quality_certificate.json) unless explicitly overridden —
# metric 3D is a hard prerequisite for any future retargeting engine.
ALLOW_2D_ONLY_VLA_FINETUNING = False

# ═══════════════════════════════════════════════════════════
# PRIVACY / PII HANDLING (v2 addendum §10)
# ═══════════════════════════════════════════════════════════
REDACT_BYSTANDER_FACES = True
# There is currently no worker face-enrollment/reference system in this
# pipeline (out of scope here) — see scripts/03b_privacy_redact.py's
# docstring. Without a reference photo, no detected face can be positively
# identified as the primary worker's, so every detected face is treated as
# an unidentified bystander and follows REDACT_BYSTANDER_FACES regardless
# of this setting. It's kept as a real, wired-through toggle for when a
# reference-photo/enrollment system exists (drop a
# processed/{session_id}/worker_reference_face.jpg to activate matching).
REDACT_PRIMARY_WORKER_FACE = False
REDACT_TEXT_AND_BADGES = True
BLOCK_DELIVERY_WITHOUT_CONSENT = True
WORKER_PROFILE_RETENTION_DAYS = 180   # ties to §5's biometric-like calibration data, once built

PRIVACY_FACE_DETECTION_MIN_CONFIDENCE = 0.5

# Coarse worker-face-match heuristic (HSV histogram correlation against an
# optional reference photo) — NOT real face recognition. Good enough as a
# forward-compatible hook; replace with a proper face-embedding model
# before relying on it to actually distinguish individuals.
PRIVACY_WORKER_FACE_MATCH_THRESHOLD = 0.75

# OCR runs on sampled frames only (expensive); detected text regions are
# carried forward and blurred on every frame until the next sample, so text
# isn't left exposed between samples.
# INVARIANT (enforced at runtime in scripts/03b_privacy_redact.py):
# PRIVACY_OCR_PROPAGATION_FRAMES must be >= PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES - 1,
# or a region confirmed at one sample would expire before the next sample
# has a chance to redetect it, leaving a real gap of unblurred frames.
# Set with 10 frames of margin above the minimum so ONE missed detection
# cycle (e.g. a momentary OCR false-negative) doesn't lose coverage either.
PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES = 30
PRIVACY_OCR_PROPAGATION_FRAMES = 40
# Three-tier confidence handling: below MIN -> discarded as noise, between
# MIN and BLUR -> blurred AND flagged for human review (fail-safe: when
# uncertain, blur first), at/above BLUR -> confidently blurred, no flag.
PRIVACY_OCR_MIN_CONFIDENCE = 0.3
PRIVACY_OCR_BLUR_CONFIDENCE = 0.7

PRIVACY_BLUR_KERNEL_SIZE = 51   # must be odd; Gaussian blur kernel for redacted regions

# ═══════════════════════════════════════════════════════════
# MULTI-EPISODE SEGMENTATION (v2 addendum §6)
#
# A session's recording often contains multiple task instances separated
# by idle gaps (finish one bolt, walk to next station, start another) —
# not one continuous task. scripts/06b_episode_segment.py splits phases.json
# into distinct episodes on idle segments exceeding EPISODE_GAP_THRESHOLD_SEC;
# 07_task_classify.py / 09_language_ground.py / 10_eis.py then run per
# episode, not per session.
# ═══════════════════════════════════════════════════════════
EPISODE_GAP_THRESHOLD_SEC = 8.0    # idle duration that signals a new episode boundary
MIN_EPISODE_DURATION_SEC = 2.0     # episode candidates shorter than this are dropped as noise fragments

# ═══════════════════════════════════════════════════════════
# CONFIDENCE PROPAGATION (v2 addendum §9)
#
# Every primitive detection in 05_primitives.py gets a continuous
# confidence float (0-1) alongside its existing boolean raw_flags entry,
# derived from how far the underlying measurement sits from its decision
# threshold (margin), discounted by upstream tracking/detection quality
# (hand-landmark confidence from hand_pose.json, object-track confidence
# from object_tracks.json) and, for IMU-derived primitives, a
# signal-to-noise factor. This is a measure of CERTAINTY in the boolean
# call either way — a value sitting exactly at the threshold scores near
# 0 confidence regardless of which side of the boolean it landed on; a
# value far from the threshold scores near 1 whether the flag came out
# True or False. See utils/confidence.py for the implementation.
# ═══════════════════════════════════════════════════════════

# IMU signal-to-noise discount: peak-to-window-std ratio at or above this
# is treated as "fully confident" (multiplier 1.0); below it, the
# multiplier scales down linearly to IMU_SNR_CONFIDENCE_FLOOR. This
# penalizes a nominally-large reading that's sitting inside a jittery
# (high-variance) window, which is less trustworthy than the same peak in
# a calm window.
IMU_SNR_CONFIDENCE_REFERENCE = 3.0
IMU_SNR_CONFIDENCE_FLOOR = 0.5

# ═══════════════════════════════════════════════════════════
# LANGUAGE GROUNDING — VLM-HYBRID (v2 addendum §7, revised)
#
# Revision of the original addendum's §7: instead of generating
# instructions purely from structured facts, generation is grounded in
# actual sampled video frames from the episode via a vision-capable Claude
# call, with structured facts as supporting context rather than the sole
# input. Defaults to "template" (the existing, free, deterministic path)
# — a VLM call costs real API money and needs network access, so turning
# this on is an explicit choice, not an implicit side effect of upgrading
# the pipeline.
# ═══════════════════════════════════════════════════════════
LANGUAGE_GEN_MODE = "template"   # one of: "template", "vlm", "hybrid" ("hybrid" = try vlm, fall back to template on any API error)

VLM_MODEL = "claude-opus-4-8"
VLM_MAX_TOKENS = 1024
VLM_MIN_SAMPLE_FRAMES = 3
VLM_MAX_SAMPLE_FRAMES = 5

# Rough $/MTok pricing for cost tracking, current as of this implementation
# pass (see docs/PIPELINE_STATUS.md §7) — re-verify against
# platform.claude.com/docs/en/pricing before trusting this for real budget
# decisions; pricing changes over time and this is not fetched live.
VLM_PRICING_USD_PER_MTOK = {"input": 5.00, "output": 25.00}

# Fraction of episodes whose VLM-generated instruction gets a second,
# independent "does this describe only what's visible" pass against the
# same sampled frames — spot-checking rather than double-calling on every
# episode, to bound the added cost while still catching systematic
# hallucination issues. Deterministic per episode_id (see
# utils.vlm_language.should_spotcheck), not truly random.
VLM_HALLUCINATION_SPOTCHECK_RATE = 0.2

# ═══════════════════════════════════════════════════════════
# GLOVE SUPPORT (v2 addendum §2)
#
# Gloves add material thickness around the fingers/palm — the SAME
# physical grasp closes to a larger apparent normalized landmark distance
# than a bare hand produces. Applying bare-hand POWER_GRASP_DIST/
# LATERAL_PINCH_DIST thresholds directly to gloved-hand footage would
# under-detect grasps (gloved fingers never register as "closed enough").
# glove_type is read per-session from an optional
# raw/{session_id}/session_config.json (mirrors 01_ingest.py's existing
# consent.json pattern), defaulting to "none" — a session is never
# assumed gloved.
# ═══════════════════════════════════════════════════════════
GLOVE_TYPE_DEFAULT = "none"   # one of: "none", "thin", "thick"
GLOVE_THRESHOLD_MULTIPLIERS = {
    "none": 1.0,
    "thin": 1.15,   # e.g. nitrile/latex work gloves
    "thick": 1.35,  # e.g. padded/insulated/leather work gloves
}

# ═══════════════════════════════════════════════════════════
# PER-WORKER CALIBRATION (v2 addendum §5)
#
# A short, explicit calibration clip (worker spreads their hand open,
# closes into a full fist, then does a lateral pinch) lets grasp/pinch
# thresholds be personalized to that worker's actual hand geometry
# instead of one global default tuned for an unknown "typical" hand. This
# is real, worker-linked biometric-like data (a hand's grasp/pinch
# aperture range is about as individual as gait, though far coarser than
# a fingerprint) — retention ties to WORKER_PROFILE_RETENTION_DAYS (§10,
# above) and storage is consent-gated in utils/worker_profile_store.py,
# not an afterthought bolted on after the fact.
# ═══════════════════════════════════════════════════════════
CALIBRATION_DIR = Path("calibration/workers")
CALIBRATION_MIN_OPEN_FRAMES = 10    # minimum open-hand frames required to trust the derived max-span value
CALIBRATION_MIN_CLOSED_FRAMES = 10  # minimum closed-fist frames required to trust the derived min-span value
CALIBRATION_MIN_PINCH_FRAMES = 10   # minimum pinch frames required to trust the derived thumb-index range

# ═══════════════════════════════════════════════════════════
# PERCEPTION SOURCE TOGGLE (v2 addendum §8)
#
# Every perception script (04_hand_pose.py, 04c_object_track.py,
# 04d_depth_estimate.py) reads compressed.mp4 unconditionally today. This
# lets a deployment choose raw.mp4 instead — e.g. to avoid H.265
# compression artifacts affecting fine-motor landmark detection — via
# utils.video_utils.resolve_perception_source(), without hardcoding a
# path in each script.
# ═══════════════════════════════════════════════════════════
PERCEPTION_SOURCE = "compressed"   # one of: "raw", "compressed"

# 01_ingest.py currently never deletes raw.mp4 after compression (kept
# indefinitely, matching pre-§8 behavior) — default True preserves that.
# Set False only once disk usage is a real constraint and PERCEPTION_SOURCE
# is confirmed to be "compressed" for the affected sessions; deleting
# raw.mp4 while PERCEPTION_SOURCE="raw" would break every perception
# script for that session.
KEEP_RAW_AFTER_COMPRESSION = True

# ═══════════════════════════════════════════════════════════
# DATASET-LEVEL QC (v2 addendum §11)
#
# scripts/11b_dataset_qc.py runs after scripts/11_package.py assembles a
# batch's dataset_manifest.json — it operates across ALL sessions in that
# batch, not within one session, so it's the first stage in the pipeline
# where "batch" is a meaningful unit at all.
#
# IMPORTANT: with only one real session in this repo (session_001,
# confirmed off-taxonomy — see the business-blocker note in
# docs/PIPELINE_STATUS.md), every number this stage produces against real
# data today is structurally meaningless at batch scale — there's no
# second real session to be a near-duplicate of, no real task distribution
# to be imbalanced, nothing to meaningfully split. This section's logic is
# verified against synthetic multi-session fixtures
# (tests/test_dataset_qc.py), same discipline as §2/§5/§7's task-signature
# tests — not presented as validated against a real multi-session dataset
# that doesn't exist yet.
# ═══════════════════════════════════════════════════════════

# Perceptual-hash Hamming similarity (0-1, 1.0 = identical) above which two
# sessions are flagged as near-duplicate episodes of the same recording
# (e.g. an accidental double-upload) rather than two distinct episodes.
# Deliberately hash-based (utils.dataset_qc._phash_frame), not an
# embedding model — no extra GPU dependency for a check this coarse.
DEDUP_SIMILARITY_THRESHOLD = 0.90

# Episode-level stratified train/val/test split. Must sum to 1.0 —
# validated in utils.dataset_qc.stratified_split, not silently
# renormalized, since a typo here (e.g. summing to 0.9) should fail loudly
# rather than quietly shrink one split.
TRAIN_VAL_TEST_SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

# Field name (within each episode's manifest entry) used to stratify the
# split — keeps each task's episodes distributed proportionally across
# train/val/test instead of e.g. all of one rare task landing in test by
# chance. "unknown" (session_001's actual label) is stratified like any
# other task value, not special-cased.
SPLIT_STRATIFY_BY = "L1_task"
