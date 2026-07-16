# Actuate — Honest Status

Per Master Spec §0: *"A component is 'done' only when tested against real data, and every
correctness test must be confirmed to fail against a broken version. 'Looks right' is not
a status."*

Graded **tested-against-real-data** / **unit-only** / **written-only**. Nothing is graded
on whether it compiles.

**Phase 3 COMPLETE (Parts A–F) + integration gate PASSED on the real capture.** 2026-07-15
`schema_version = 3` · **101 unit tests green** · import-linter clean · ruff clean

---

## Phase 4a Part D — ARM RETARGETING: sim machinery built, sim gates pass; real-capture DEFERRED

`actuate.retarget.arm` — wrist trajectory → robot joint trajectory. Built the depth-INDEPENDENT
sim machinery (the estimator is robot-side, sim-trained; the wrist trajectory is only an input at
inference), holding the real-capture gate until the depth benchmark passes. Module:
`retarget/arm/` (`robot.py`, `simdata.py`, `estimator.py`, `candidates.py`, `__init__.py`),
CLI `actuate retarget {train-arm, arm}`.

**Deviation, forced + honest: MuJoCo, not Pinocchio.** `pip install pin` has no Windows wheels
and fails to build; MuJoCo 3.10 installs on Windows AND Linux/Kaggle and gives FK, Jacobian IK,
and physics replay in one working dep. The `fk`/`jacobian`/`ik` interface is solver-agnostic so
Pinocchio can slot in later on Linux. Franka Panda registered in the embodiment registry
(`urdf_path="robot_descriptions:panda_mj_description"`), retarget-**ready** but not
sim-**validated** (those states kept distinct).

| Component | Status |
|---|---|
| MuJoCo kinematics + damped-least-squares IK (restarts) | **97% cold IK convergence**, 0.9 mm residual, 80 ms/solve |
| Vector-Neuron layers (SO(3)-equivariant) | **verified exactly equivariant** — max‖v(Rx,Rc)−R·v(x,c)‖ = 0.000000 |
| Flow-matching root-frame estimator (not regression → many candidates) | trains, loss decreases; samples distinct candidates |
| Sim data gen (OU joint paths → FK → valid pairs) | **GATE 1 PASS**: 100% joints in-limits, reachable, unit gravity |
| Candidate selection (IK-score: convergence/residual/manip/margin/smoothness) + cluster | end-to-end run: 100% IK on a smooth traj, joint_traj (T,7) |

**Sim gates:** GATE 1 (data validity) ✅. GATE 4 (candidate spread 0.24–0.31 m, no mode
collapse) ✅. Equivariance ✅ (the whole "SE(3)-equivariant" claim, to numerical zero).
**GATE 2 (>90% IK convergence on held-out sim)** is at **84% on an 18-second CPU smoke run** —
the mechanics are validated but >90% needs the full ~1.5–2 hr training the spec calls for,
which is a **Kaggle T4 job** (`actuate retarget train-arm`). **GATE 3 (real-capture validity)**
is **deferred** until the Phase 3.5 depth gate passes — `run` works on any wrist trajectory, but
it's only as good as the depth that produced it. Nothing about this arm-retargeting work changes
if depth pivots to stereo, so none of it is wasted.

**Not built (later parts):** ~~Part F reconcile + sim validation~~ — built in Phase 5 Part B.0
(below). Contact-stability / no-slip validation remains unbuilt (no contact model, no hardware).
Gravity in `run` is a camera-down placeholder (real IMU gravity not wired).

## Phase 5 Part B — THE QUALITY CERTIFICATE BECOMES REAL (schema v4, certify.score)

**B.0 unblocked the certificate first.** Part B's gate 1 needs `retarget_eligibility` (from sim
validation) and `strategy_alignment` (from L5 reconciliation) — neither existed (Phase 4a Part F
was never built). Built minimal, honest versions:

- `retarget/reconcile.py` — frame overlap, teleport detection (0.3 rad/frame — a discontinuity
  detector, NOT an actuator model), optional grasp-agreement vs fused interaction states.
  **FAILS on a shuffled trajectory** (the required broken variant).
- `retarget/sim_validate.py` — MuJoCo replay: joint limits, self-collision (added
  `RobotModel.self_collision_count`, same touching≠interpenetrating semantics as the hand),
  IK-convergence floor 90%. **CATCHES a deliberate limit violation**, including on the real
  Franka model. Kinematic only: "eligible" means executable, not "the grasp will hold".

**Schema v4** (one bump covering Parts B and D, additive-optional — v3 payloads validate
unchanged, verified): `EpisodeMeta.components: CertificateComponents` (5 measured inputs behind
`quality`, each `None` = NOT MEASURED, never zero), `CanonicalEpisode.retarget_eligibility`
(per-embodiment, absent ≠ False), `FieldStats.p02/p05/p25/p50/p75/p95/p98` (full percentile set
for TRI-LBM/EgoMimic re-derivation). Freeze shown red (2 failed) → `actuate schema freeze` →
green.

**`certify.score` on the real capture** (`actuate certify run`):

| Component | Value | Why |
|---|---|---|
| sync_integrity | 0.88 | 3.94 ms drift vs 33 ms frame period (L0 report) |
| calibration_completeness | **0.15** | intrinsics GUESSED (the fx 1.7× bug) caps at 0.3; no SLAM on the v1-legacy build halves it |
| perception_confidence | 0.62 | mean of per-frame L1/L2 confidence |
| contact_consistency | None | bare-hand rig, no sensor — not measured, NOT zero |
| ik_convergence_rate | None | L5 not run on this ego-contaminated legacy episode |
| **quality** | **3/5** | in the predicted 2–3 band, not 5 |
| speed | 3 (slow) | 95 s by TIMESTAMP span — the raw-frame-count shim it replaces would misgrade a subsample |
| mistakes | 9 flags | 6 low-confidence segments (seekable: `low_confidence@42s-43s`) + v1's 3 flags |

**Gate 4 proven:** an episode boosted to quality=5 with consent=pending still raises
`ConsentViolation`. Quality never opens the consent gate.

Composite renormalises over MEASURED components only (an unmeasured channel neither helps nor
hurts — tested). `certify` sits below `retarget` in the import contract, so L5 results arrive
duck-typed; the CLI wires the layers. 145 unit tests green, import-linter 2/2.

## Phase 5 Part D — DUAL-SPACE DELIVERY + TIERING (the product differentiator)

**All five gates pass, verified through LeRobot's OWN loader on the real capture:**

1. **Dual-space** — `export_lerobot_v3(..., embodiment="franka_panda")` ships the human-space
   `action` (8-dof wrist) AND `action.robot.franka_panda` (7 joints) in ONE dataset, selectable
   by feature tag. Loaded back via `LeRobotDataset`; both spaces present, no NaNs. (The robot
   trajectory in the gate is SYNTHETIC and labelled so — it proves the plumbing on real video;
   the retargeted values are validated by the L5 gates, not here.)
2. **Tier filter** — episodes kept by their OWN `episode.tier`; unassigned = stage1_volume
   (stage2 is a claim, never a default). `--tier stage1` on a stage2-only set refuses with
   "excluded every episode — that is the filter working". Both directions tested; the
   two-tier case is **synthetic fixtures (n=1 real corpus), unit-only and says so**.
3. **Norm round-trip red→green** — `verify_round_trip` judges "in range" by the DATA's own
   recomputed percentiles, never the stat under test: a check that trusts the shipped stat
   lets a corrupted p99 shrink the clip region and hide its own damage (could never fail —
   caught in design, rejected like the coherence-cos metric). Tampered p99 and degenerate
   p01==p99 both FAIL loudly; swapped bounds documented as NOT catchable (sign-flipped but
   perfectly invertible). Runs inside every export.
4. **Manifest** — scene and demonstrator diversity SEPARATE (EgoVerse), task/tier
   distributions, per-component certificate means (None stays None), modality inventory.
   Honest on n=1: `episodes_with_unknown_demonstrator: 1` ships in the artifact.
5. **Load+train gate still green** — full integration suite passes on the extended exporter
   (provenance file became dataset-level: hashes/consent as lists, notes keyed by episode).

Also: full percentile set (1,2,5,25,50,75,95,98,99) per space — human stats in the historical
`actuate_norm_stats.json`, robot per-embodiment in `actuate_norm_stats.<emb>.json`; the two are
verified to be genuinely different stats. Export-time co-training transforms (`masked_hand`,
`eef_overlay` — EgoMimic) refuse to run without intrinsics: a mask projected with guessed
intrinsics hides the wrong pixels silently. Robot-action alignment is by length and REFUSES
ambiguity (RobotAction carries no frame ids — schema limitation, recorded). 159 unit tests
green, import-linter 2/2.

## Phase 5 Part C — RLDS/Open-X SECONDARY EXPORTER (gated by tfds.load on the real capture)

Written THROUGH tfds's own ad-hoc builder (`store_as_tfds_dataset`), same doctrine as the
LeRobot exporter: the on-disk layout is inherited, never re-implemented from memory. Per v3's
"training knowledge only" flag on the RLDS layout, every structural fact was verified against
the installed tfds 4.9.10 by **writing and re-loading a probe dataset first** — which caught
two facts memory would have gotten wrong (iterable splits must yield `(key, example)` PAIRS;
nested steps are `tfds.features.Dataset`, not a list feature).

**Gate (real capture, 60 frames):** `tfds.load("actuate_gate", data_dir=...)` — the
INDEPENDENT reader, not our own assertion — iterates one episode: step keys exactly
`{observation{image(224,224,3) u8, state(8) f32}, action(8) f32, language_instruction,
reward, discount, is_first, is_last, is_terminal, action_robot_franka_panda(7)}`, boundary
flags correct, image a real video frame. `reward=0, discount=1` throughout — human demos
carry no reward signal (Open-X convention), stated in shipped provenance rather than faked.

Dual-space parity with Part D (`action_robot_<emb>` — underscores; tfds feature names are
identifiers). Same fail-closed refusals: no task, no video, mixed layouts, tier filter
excluding everything. Manifest + provenance ship next to the tfds data_dir. CLI:
`actuate package rlds --in <canonical> --video <redacted> --out <dir>`.

Install note: `pip` on PATH belonged to Python 3.11 while `python` is 3.10 — installs were
silently going to the wrong interpreter; `python -m pip` fixed it. tensorflow bumped protobuf
to 6.x (mediapipe pins <5 but still imports; v1-legacy only, `src/` never imports it).

## Phase 5 Part A — LANGUAGE RICH-CONTEXT (all four gates PASS on the real capture, real API)

Ported v1's `utils/vlm_language.py` (the injected-client seam kept — every code path
unit-tests against a fake client, 10 tests, no network) and upgraded to π0.7-grade:

| Gate | Result |
|---|---|
| 1 — paraphrases diverse | **PASS** — 5 paraphrases, different structures not word-swaps ("Head to the workbench, sort through the papers, and staple them" vs "Organize and bind the paperwork with staples once you're at the workbench") |
| 2 — subtasks align with phase boundaries | **PASS** — 9 subtasks exactly on the (flicker-healed) v1 phase segments; subgoal_frames at each boundary |
| 3 — judge catches hallucination, red→green | **PASS, real API on real frames** — planted "a red stapler and a coffee mug": object_consistency **0.10**, verdict inconsistent, all four fabrications named in unsupported_claims. The faithful caption passed at min 0.70, NOT flagged (a judge that flags everything is as useless as one that flags nothing — tested both directions) |
| 4 — schema fields populated | **PASS** — task_paraphrases(5) / subtasks(9, all with judge-derived confidence) / subgoal_frames(9) in schema-valid v4 canonical output |

**The judge design:** independent second call (never the generator scoring itself), four
dimensions (hand/object/action/global), grounded in perception facts (tracked objects,
interaction states) so it is more than a vibe check. Below-threshold captions are **flagged
for review and shipped with low confidence — never silently accepted, never silently
dropped** (a dropped segment hides the disagreement a reviewer needs). The real run flagged
1/9 (a 10-frame flicker segment, conf 0.60) — the QA protocol doing its job.

**Cost honesty:** estimate shown before every billed call (`--yes` to skip the prompt).
Real run: estimated $0.84, actual **$0.8756** — above the ~$0.25 quoted at planning, because
the capture has 9 real segments, not the 3-4 assumed. Merging v1's flickery same-phase
segments (15 → 9) was done for subtask quality and cut the cost 40% as a side effect.

**API-drift facts encoded** (per the claude-api reference, not memory): `temperature` is
REMOVED on the current model family (the "temperature=0 for judge determinism" plan would
have 400'd); structured-output schemas carry no numeric bounds (scores clamped client-side).
Key from env/.env.local (git-ignored), never hardcoded/logged; no key → SKIP with a warning,
the pipeline never fails on missing language. 169 unit tests green, import-linter 2/2.

## Phase 4a Part E — GeoRT FINGER RETARGETING (Allegro): GATE 3 NOT TESTABLE ON THIS CAPTURE

`retarget.finger.train/run` maps MANO fingertips → Allegro's 16 DoF. Contact-blind by
construction; Allegro has no pinky, so the human pinky is dropped (real information loss).

| Gate | Status | Evidence |
|---|---|---|
| 1 — trains without errors | **PASS** | fwd-model 4.3e-5 m², recon 1.4e-4 m², ~19 s CPU |
| 2 — synthetic MANO plausible | **PASS** | open 0.112 > pinch 0.105 > fist 0.094 m; in-limits; no interpenetration. Discriminates: identity calibration **inverts** the ordering |
| 3 — real capture | **NOT TESTABLE** | see below |

**Gate 3's premise is false for this capture, and that is the finding.** It asks that a real flat
hand retarget to an open robot hand. Measured against the demonstrator's *own* MANO canonical
poses, the hand in this footage is **never open and never a fist** — median openness **0.50** of
its own fist→open range, i.e. a half-curled writing posture, across 95 s. There is no flat hand to
test, and no fist to calibrate on. The method's stated prerequisite (~5 min of per-human canonical
finger motion) **was never captured**. Retargeting is therefore **validated on synthetic MANO +
sim only — not on real manipulation footage.**

**Two silent convention bugs were found and fixed on the way** (each ran fine and produced
in-limit joints while being wrong):

1. **The canonical fist wasn't a fist.** MANO's 45 = 15 joints × 3 axis-angle, and only axis 2 is
   flexion. Filling all 45 uniformly twists and splays the hand: tips 0.146 → 0.113 m (barely
   curled, thumb-only). Bending the flexion axis alone gives a real fist at 0.073 m. The
   calibrated magnitude **0.8 is the end of the monotonic range** — past it the fingers
   over-rotate and tips travel back *out* (1.6 → 0.092 m), so a bigger "more closed" number
   silently means a *less* closed hand.
2. **Wrist-relative was not enough.** WiLoR's `keypoints_3d` still carry the hand's
   `global_orient`, while calibration poses are generated at orientation zero — comparing a
   rotated hand to an unrotated reference. Per-finger alignment with the canonical open pose:
   cosine **0.68 → 0.95** once de-rotated.

Net effect on real frames: openness **0.086 → 0.093 m** and interpenetration **12/12 frames → 0**.
The old value sat *below* the fist reference (0.094) — a real flat-ish hand retargeting to *more
curled than a fist*, i.e. the map was inverted. It now lands between fist (0.088) and open (0.104),
the correct region, though still more curled than the human's true 0.50.

**Also verified: WiLoR emits MediaPipe keypoint order** (thumb 1-4, index 5-8, …), not MANO's
(index 1-3, …). Confirmed empirically — its thumb chain matches MANO's to 1 mm across all four
joints. `ALLEGRO_MANO_TIPS = (8, 12, 16, 4)` is correct.

**A metric I rejected rather than banked.** Correlating human openness against retargeted openness
over 40 real frames gives **+0.998** — but the *broken* identity-calibration variant scores
**+0.938**. It does not discriminate (openness is a scalar dominated by tip magnitude, which
survives a bad calibration), so it is not evidence of anything. Same failure class as the
retracted coherence-cos depth metric. Gate 2's ordering test is the one that discriminates.

**To actually close Gate 3:** capture ~5 min of the demonstrator opening, fisting, and freely
moving their fingers. No amount of modelling substitutes for it.

---

## Phase 3.5 Part A — depth benchmark harness + scale-anchoring + MoGe-2 (pre-Kaggle)

The temporal-depth decision is benchmark-first; the gate is **wrist z-jitter < 4 mm/frame**
smoothed — a 3× reduction from the *fair* 11.3 mm hand-cloud-fit baseline (NOT the 20.7 mm bbox
strawman). Built and unit-tested locally; the models run on Kaggle T4.

- **`depth.benchmark.run_benchmark`** — the gate harness. Scores any `{name: DepthResult}` on the
  same HandResult by placing the wrist (`solve_root_depth` + fit + smooth) and measuring the
  placed-z jitter (mm/frame), plus static-point consistency and metric scale. Prints a table +
  per-model gate verdict, and explicitly reports the "wins static consistency but NOT the wrist,
  monocular floor stands" case as a *finding*, not a pass.
- **`temporal.anchor_scale`** — keyframe scale-anchoring. Fits ONE global affine
  `d = a·d_video + b` from a temporal model (VDA) to a metric anchor (UniDepth/MoGe-2) over
  keyframes. Monotonic → preserves the video model's temporal consistency exactly, only fixes
  absolute scale + intrinsics. Unit-tested: takes the anchor's scale, keeps the video model's
  low jitter.
- **`depth.run(model="moge2")`** — MoGe-2 backend (per-frame metric depth + estimated focal).
  Not temporal; benchmarks whether a better metric/focal anchor alone lowers wrist jitter (our
  `fx≈660` is currently a guess), and serves as the anchor for VDA. Kaggle-validated (wrapper
  follows microsoft/MoGe's `MoGeModel.infer`).
- Kaggle scaffold's `--depth-ab` now runs the **full four-model table** (UniDepth / MoGe-2 /
  VDA-anchored / flow-filter) and writes `depth_ab.txt`.

**Status: awaiting the Kaggle run.** VDA + MoGe-2 need the T4; the harness, anchoring, and
flow-filter are validated on synthetic data (the metric reads 11.2 mm on synthetic 2% noise,
matching UniDepth's real ~13 mm). Decision gate on the real numbers is pending. 111 unit tests.

---

## Post-Phase-3 follow-ups (2026-07-16)

**Temporal / video-depth stage — the A/B challenger to UniDepth's noise floor.**
`perception.depth.run(model=...)` now switches: `auto`/`vitl`/`vits` → UniDepthV2 (default);
`video_depth_anything` → Video-Depth-Anything (temporal; **Kaggle**); `flow_filter` → an
optical-flow-warped temporal EMA over any base DepthResult (runs anywhere). All return the same
`DepthResult`, so downstream is agnostic. The **decision function** is
`perception.depth.consistency.compare` — the Part C physics test packaged: track static points,
measure frame-to-frame depth wobble, lower wins. Validated locally: the metric reads **11.2 mm
on synthetic 2% noise** (matches UniDepth's real ~13 mm), and the flow-filter cuts static-scene
wobble **66%**. VDA itself is validated on Kaggle (needs the model + T4), same reserve-the-seam
discipline as the FoundationPose stub. 5 unit tests.

**Objects → canonical `ObjectState` (was unwired).** `build_from_perception(objects=...)` now
carries the SAM2 **mask** per tracked object into `frame.objects[<track_id>]` with provenance
`vision_primary`. Pose stays `None` — 6-DoF needs FoundationPose (mesh + bigger GPU), and a
position-only SE3 would fabricate an orientation. Mask round-trips exactly. 1 unit test.

**Kaggle scaffold (`kaggle/`).** `run_perception.py` runs the full pipeline (SLAM + WiLoR +
UniDepth + GDINO/SAM2 + fusion) on a T4, writes `.actuate_cache/*.pkl` in the exact format
`actuate viz --cache` reads (same key scheme), builds the v3 episode + `.rrd`, optionally runs
the depth A/B, and zips it for download → view locally with **no GPU**. `README.md` has the
consent warning (upload the capture as a *private* dataset), the model-install cells, and the
download-then-`--cache` loop. Motivated by the 4 GB card's 86-min SAM2 thrash.

108 unit tests green.

---

## Phase 3 INTEGRATION GATE — full pipeline end-to-end on the real capture ✅

`hands (WiLoR) → objects (GDINO+SAM2) → slam → depth (UniDepth) → fuse (L2) →
build_from_perception (canonical v3, MANO) → export LeRobot v3 → LeRobot load + one train step`,
run on 32 real frames of session_001. The L1/L2→L3 wiring is `canonical.build_from_perception`.

| Gate | Result |
|---|---|
| LeRobot load+train with 45-MANO in `observation.state` | **state_dim = 53** (8 wrist+grasp + 45 MANO), action (16, 53); loss 71.7 finite; **grad_norm 1248 > 0** — a real gradient flowed through the MANO params |
| Actions metric (SLAM + measured depth, not bbox pseudo-depth) | wrist median depth **0.59 m** from UniDepth `solve_root_depth` (not WiLoR's bbox inference); ego-motion from SLAM |
| Rerun viz shows a coherent 3D scene, hand at metric depth not floating | Part F: hand placed at ~0.6 m in the depth cloud (unit-tested `0.4 < root_z < 0.8`) |
| All existing tests still pass (no regression) | **101 unit tests green** (was 493 in the v1+v2 suite; the Phase 3 unit subset is 101) |
| `actuate-delivery-dev`: still 0 objects, nothing ships | episode `consent=pending` → **`is_deliverable = False`**; export writes to a local/work dir, never delivery; no S3 write this session |

**Honest scope notes:**
- The canonical episode carries MANO (45) + metric wrist pose + `interaction_state` + SLAM
  `camera_pose`. It does **not** carry `contact`/`finger_joints` — a bare-hand rig measures
  none, and the schema forbids a vision-inferred contact from posing as a measurement.
- "Metrically correct" is bounded by Part C: the wrist depth is *measured* (UniDepth), not
  bbox-inferred, and ego-motion rotation is compensated — but monocular depth is still noisy
  (STATUS Part C), so the action is metric-in-kind, not metric-to-the-millimetre. It is the
  honest best from this rig, and a real improvement over v1's white-noise trajectory.
- The delivery-bucket check is structural (non-deliverable by consent), not an S3 query — AWS
  calls need explicit approval and the SSM tunnel, neither used here. Nothing was written to S3.
- Fixed during integration: `smooth_root_depth` zero-padded at the array boundaries, reading the
  first/last frames' wrist depth ~4/7 too shallow. Now edge-normalised; unit-tested.

---

## 🔴 LAUNCH BLOCKER — MANO / WiLoR are NON-COMMERCIAL

**WiLoR is CC-BY-NC-4.0. MANO is a Max Planck model licensed for non-commercial use.**
Actuate sells datasets. HaMeR — the spec's designated fallback — is also MANO-based, so it
inherits the identical constraint and solves nothing.

This does **not** stop at "which model we run internally". Master Spec §3 makes **MANO the
intermediate**, and §L5's Stage-I *delivered* target is *"retargeted joints on a canonical
reference hand"* — **derived from MANO**. So the licence follows the parameters into the
**delivered dataset** and into all of Phase 4.

Cleared for **INTERNAL RESEARCH ONLY**. **A commercial licence from MPI is required before
any MANO-derived data reaches a customer.** Not a code problem; do not let it surface late.

---

## Phase 3 Part A — SLAM ego-motion: BUILT, real-data gate DEFERRED

`perception.slam` — swappable backend, VIO built (gyro rotation + vision), `orb_slam3`
registered and honestly unimplemented (no pip dist, no C++ toolchain; it also solves a
harder problem than reprojection needs — relative pose over ≤0.5 s, never a global map).
Wired into `canonical build` → reprojection. 2850 frames in ~40 s on CPU.

**Its gate could not pass, and the reason is the finding of this phase:**

```
implied hand speed  : median 1.08 m/s   p99 32.4 m/s (= 115 km/h)
velocity direction  : reverses 49% of frames, coherence cos +0.046
corr(action, head rotation) : +0.022        <- essentially ZERO
```

**The v1 3D wrist trajectory is white noise.** Ego-motion is real but buried under a noise
floor many times larger — you cannot measure a 13 mm correction inside a 274 mm error.

Attribution is unambiguous:

| Source | Evidence | Verdict |
|---|---|---|
| MediaPipe **2D** | cos **+0.905**, 19% reversals, 9.4 px | **Smooth. Not the problem.** |
| **Depth lift** (Depth-Anything-V2, conf 0.6) | wrist depth jumps **28 mm median / 274 mm p99 per frame** | **This is the noise.** |

**Two errors of mine, recorded so they aren't repeated:**
1. I first reported "36% of the action is head motion". That was a *magnitude bound*, not a
   demonstration. The correlation test says otherwise.
2. My gyro-vs-vision cross-check (r=0.92) compared rotation **magnitudes** — and `|R|` is
   frame-invariant, so it was **structurally blind** to the bug that was present: the IMU
   and camera axes are misaligned (agreement 0.618; extrinsic −6°/−39°/−51°). **A gate that
   cannot fail on the bug in front of it is not a gate.**

**Ego-motion is not the blocker. Depth is.** Part A's real-data gate resumes after C.

## Phase 3 Part B — WiLoR → MANO: RUNS ON REAL DATA (depth still broken)

| Check | Result |
|---|---|
| **VRAM on RTX 2050 (4.29 GB)** | **fits**: 1.50 GB weights (fp16), **2.44 GB peak, 1.85 GB headroom**. fp32 would not fit. |
| Throughput | 220 ms/frame → **full 2850-frame capture in ~10 min** |
| MANO params on real frames | populated, **no NaN**, betas max 2.68 (non-exploding) |
| Biomechanical plausibility | median 29°/joint, max 109°, **zero joints beyond 180°** |
| 2D keypoint jitter | **2.8 px/frame** vs MediaPipe's 9.4 px — **3.4× better** |
| Articulation (root-relative) | 3.7 mm/frame — plausible |
| **Root DEPTH** | ❌ **still broken: 20.7 mm/frame** vs **0.1 mm laterally** |

**Why depth is still broken, and why it is not a bug:** WiLoR solves a *weak-perspective*
camera against the hand's bbox, so `depth ≈ focal/(scale·bbox)`. The detector re-runs every
frame with no tracking, the box breathes ~6.8%, and depth breathes with it (corr −0.31). It
**infers depth from apparent size; it does not measure it.** Its `scaled_focal_length` of
37500 px is not a lens — it is HaMeR's *virtual* focal (5000 × img/256). Rescaling by a real
focal recovers a plausible **1.03 m** hand depth, so **the placement of the hand depends
entirely on a focal length we do not have.**

**Same root cause as v1, different model.** Depth-Anything: 28 mm/frame. WiLoR
weak-perspective: 20.7 mm/frame. **Parts B and C are not independent** — B gives an
excellent 2D hand and articulation; **C must supply the metric depth and the true
intrinsics** to place it.

### ✅ Schema v2 → v3 DONE — MANO widened to the full 45 axis-angle

v2 stored MANO pose as **15-PCA**. Projecting WiLoR's native **45** axis-angle onto the top-15
PCA subspace (correct least-squares projection — the components are **not** orthonormal, so an
earlier transpose-inverse over-reported the loss as 31°) costs **median 10.3° / p90 17.3° /
p99 22.5°** per joint on **real** poses — material for retargeting where contact placement is
the whole point (§L5), and carrying the full 45 is free and exactly lossless. So (approved,
then implemented):

- `MANOParams.theta_pca` (15) → **`MANOParams.theta` (45)**; `SCHEMA_VERSION` **2 → 3**; frozen
  `canonical_v3.schema.json` regenerated; `test_schema_version_is_three` asserts it.
- `io/store.py` dense MANO array widened 15 → 45.
- **LeRobot exporter now emits the full 45 in `observation.state` and `action`** (cols 8–52),
  via an **adaptive DOF layout**: a MANO-bearing episode ships **53-dim** (wrist + 45); a
  MediaPipe-only episode ships **8-dim** (wrist only), because fabricating 45 NaN/neutral
  columns would be a false claim and would make the exporter drop every frame. The layout
  reflects what the pipeline measured; names travel with the dataset. `episode_dof_names()`.
- MANO articulation is root-relative, so the action's MANO is the next frame's MANO **without**
  ego-motion reprojection (only the wrist is reprojected).

**Verified:** 69 unit tests + 9 LeRobot structural gate tests on the real capture pass;
adaptive 8/53 layout confirmed synthetically. **Not yet verified:** the 53-dim path through
LeRobot's own writer+loader on a *real WiLoR-built* episode — WiLoR→`build_episode` wiring does
not exist yet (a Part B/C→canonical integration, still pending). **Migration:** no persisted v2
canonical outputs exist (disk empty; S3 empty per Part B) — the migrate step is a genuine no-op,
so no speculative migrator was written (nothing to verify it against).

---

## Phase 3 Part D — OBJECTS: detection + segmentation + tracking RUN ON REAL DATA; 6-DoF stubbed

`perception.objects.run(store, prompts)` — Grounding DINO (open-vocab boxes) + SAM2 (temporal
mask propagation) + IoU tracking + metric position via Part C depth. Module:
`perception/objects/` (`objects.py`, `rle.py`, `foundationpose.py`). All three GPU models are
TINY and run sequentially, never co-resident:

| model | VRAM | role |
|---|---|---|
| grounding-dino-tiny | ~0.7 GB | text-prompted detection (once per chunk) |
| sam2-hiera-tiny | ~0.2 GB | mask propagation across the chunk |
| UniDepthV2 (Part C) | 3.64 GB | metric position back-projection (own pass) |

**All three verification gates PASS on the real capture (45 frames):**

| Gate | Result |
|---|---|
| ≥1 object detected & tracked across frames | **6 tracks** (documents + hands), each persisting all 45 frames |
| SAM2 masks temporally consistent (adjacent IoU > 0.7) | longest track median **IoU 0.937**, 95% of pairs > 0.7 — locked on, not flickering |
| Positions back-projected via Part C depth are plausible | 270 detections, median depth **0.73 m** (matches Part C's 0.76 m scene depth) |

Fixed during the gate: two GDINO boxes over one object were both associated to the same
track_id and both propagated (track 2 showed 90 masks over 45 frames). Association now lets each
existing track be claimed by at most one detection per chunk (highest score first); re-verified
every track is 1 mask/frame.

**6-DoF pose (FoundationPose) is an INTERFACE STUB, for two honest reasons:**
1. It needs a **CAD mesh** — the capture shows paperwork/stapler, for which we have none, so
   there is no geometry to estimate orientation against. Only 3D *position* is recoverable
   (mask centroid + metric depth), which is what the module emits.
2. It needs **nvdiffrast + >8 GB VRAM** for its render-and-compare loop; the RTX 2050 (4.29 GB)
   OOMs. Flagged for T4/A100. The interface (`FoundationPoseEstimator.estimate`) is reserved
   and raises `FoundationPoseUnavailable` naming the gap; a synthetic smoke test exercises the
   contract shape.

Provenance stamps: `objects.bbox`/`objects.mask` = vision_primary; `objects.position` =
vision_fallback (inherits depth's weakness); `objects.pose_6dof` = approximated/not-produced.

**Verified:** 10 unit tests (RLE bit-exact incl. a broken-row-major-decoder demo, IoU,
FoundationPose stub) + the 3-gate run on real data. **Caveat (same as v1):** detection
*accuracy* is not validated against human-annotated ground truth — only that the models load,
run, and produce plausible, temporally-consistent output. "document" and "hand" are what GDINO
returned for the desk scene; whether every mask is semantically correct is unverified.
**Not wired:** objects → canonical `ObjectState` (schema carries pose+mask only; bbox/confidence
would be another field addition) — a build integration, not done.

---

## Phase 3 Part F — RERUN VISUALIZATION: `actuate viz` produces a real recording on real data

`viz.log_episode(...)` + `actuate viz <session>`. Logs every present modality to Rerun on ONE
scrubable `frame` timeline. Module: `viz/` (`conventions.py` = viewer-free topology/colours,
`rerun_log.py` = logging), CLI: `cli/viz.py`. rerun imported lazily so the library never
requires a viewer to import.

**Modalities logged** (verified on the real capture, 30 frames, via the CLI):

```
logged modalities: video 30 | depth 30 | camera 30 | hand 61 | objects 30
                   state 30 | contact 60 | action 59
```

- **video** ego RGB (`rr.Image`); **depth** as `rr.DepthImage` + a back-projected 3D point
  cloud; **camera** trajectory from SLAM (`rr.Transform3D` + `rr.Pinhole`, 30 finite poses);
  **hand** 3D keypoints + skeleton (`rr.Points3D`/`rr.LineStrips3D`), L/R coloured, fingertips
  tinted by contact confidence; **objects** `rr.Boxes2D` + `rr.SegmentationImage`; **state**
  colour-coded (green/blue/red) + `rr.TextLog`; **contact** per-finger `rr.Scalars`; **action**
  wrist deltas as `rr.Arrows3D`. FoundationPose 6-DoF would be `rr.Boxes3D` — absent (stubbed).

**Verification gate met:**

| Condition | Result |
|---|---|
| `actuate viz` produces the real capture with all modalities | valid 364 MB `.rrd` (RRF2), 8/8 modalities logged with correct per-frame counts |
| Timeline is scrubable across synchronized modalities | everything logged on one `frame` timeline (`rr.set_time`) |
| 3D view: camera trajectory + hand + depth cloud in one frame | SLAM path, hand, and depth cloud all in the `world` RDF frame |
| Interaction states visible as colour-coded annotations | STATE_COLORS green=STATIC / blue=GRASPED / red=MOVING, per frame |

**Hand is placed at metric depth, not floating:** the hand root is solved from the depth map at
the hand keypoints (Part C's `solve_root_depth` + smoothing) and back-projected, so it sits in
the depth cloud at ~0.6 m. Unit-tested numerically (`0.4 < root_z < 0.8`).

CLI: `actuate viz show <session> [--live] [--stages depth,hands,objects,fusion,slam] [--out X.rrd]`.
Default writes a self-contained `.rrd`; `--live` streams to a running viewer as stages finish.

**Verified:** 8 unit tests (skeleton-tree topology, colour conventions, NaN-safe contact colour,
`log_episode` against a real in-file recording, metric-depth placement) + the real-capture CLI
run. Headless here, so the GUI "drag the scrubber" step is manual; the `.rrd` is valid and
complete, which is the verifiable proxy. **Note:** the wrist-action arrow uses the frame-to-frame
root displacement for viewing — it is NOT the ego-motion-compensated canonical action, and depth
noise (Part C) makes it rough; it shows motion direction, not a trained target.

---

## Phase 3 Part E — L2 FUSION: trust-weighted arbiter + Schmitt-gated states RUN ON REAL DATA

`fusion.run(hands, rig=..., objects=...) -> FusionReport`. Pure pipeline logic (typed state
machine + NumPy), no model. Module: `fusion/` (`arbiter.py`, `states.py`, `fusion.py`).

**The arbiter** resolves same-channel conflicts by a fixed trust ordering (one place,
`TRUST_RANK`): measured_robotspace > measured_human > gripper_aperture > vision_primary >
vision_fallback > approximated. Confidence is a tie-breaker *within* a tier, never across —
a vision reading at 1.0 never beats a glove at 0.3.

**All four verification conditions met:**

| Gate | Result |
|---|---|
| States temporally coherent (no single-frame flickers after Schmitt) | **0 interior flickers** on 60 real frames (Schmitt hysteresis + min-dwell) |
| Provenance = vision_fallback for grasp on the head-mounted rig | grasp **vision_fallback**, contact **vision_fallback** — no hardware sensor exists, so this is correct |
| Synthetic-glove test: hardware overrides vision, broken-priority MUST fail | glove (measured_human) beats vision; the **inverted-rank variant flips the outcome and fails the assertion** — demonstrated red |
| Per-finger contact populated (non-zero, non-NaN) | 600 readings, all finite, 100% non-zero, range 0.001–0.212, **capped ≤ 0.40** (vision can't feel contact) |

**Honest calibration finding:** on session_001 the arbiter emits only STATIC/MOVING — **no
GRASPED state** — because the hand never power-grasps: the mean-finger-curl signal maxes at
**0.15** (open/flat hand over paperwork). Thresholds (0.35/0.22) sit above that baseline, so
firing a GRASPED state here would be fabricating a label. The state machine is *not* stuck — a
unit test drives a synthetic closed fist and confirms the GRASPED branch fires. Two documented
limits: the curl metric is a **power-grasp proxy** that under-detects pinch grasps, and vision
contact is a weak proxy — both reasons grasp/contact are stamped vision_fallback here.

**Verified:** 11 unit tests (trust ordering, tier-beats-confidence, Schmitt hysteresis,
min-dwell, grasp geometry, the broken-priority demo, and `fusion.run` end-to-end on synthetic
hands incl. a glove-override) + the 4-condition run on real data. **Not wired:** fusion →
canonical `interaction_state`/`contact` fields on the frame (a build integration, not Part E).

---

## Phase 3 Part C — DEPTH: intrinsics FIXED, wrist trajectory STILL NOT RECONSTRUCTABLE

UniDepthV2 metric depth + estimated intrinsics + per-pixel confidence.
Module: `perception/depth/unidepth.py`. Ran on the real capture, 300 frames.

| Check | Result |
|---|---|
| **VRAM on RTX 2050 (4.29 GB)** | ViT-L (vitl14) **fits**: 1.46 GB weights (fp16), **3.64 GB peak, 0.65 GB headroom**. fp32 does not fit. ViT-S is the fallback. |
| Throughput | **0.62 s/frame** (EdgeGuidedLocalSSI runs un-CUDA-optimised on Windows; the compile step needs Linux) |
| Metric depth plausibility | scene median **0.76 m**, full range 0.31–2.22 m — plausible for a head-cam over a bench |
| Estimated intrinsics | **fx≈660** (per-frame median, std 19), cx/cy at image centre → **~111° HFOV** |
| Confidence | present, non-uniform (**not** normalised to [0,1]; ranges ~0.5–98) |

### ✅ The intrinsics were the hidden bug — and real intrinsics fix the SLAM rotation

Our assumed 82° HFOV gave **fx=1104**; UniDepthV2 measures **fx≈660** — **~1.7× too large.**
Re-running the essential-matrix rotation against the gyro at both focals:

| | n valid frames | median inliers | median \|R_vis\|/\|R_gyro\| |
|---|---|---|---|
| GUESSED fx=1104 | 7 | 34 | **1.24** (24% too large) |
| UNIDEPTH fx=660 | 15 | 53 | **1.00** (exact) |

A wrong focal *scales* the recovered rotation; the real focal makes vision and gyro agree to
1.00. This confirms the Part A hypothesis: the guessed `K` corrupted `recoverPose`.
(Caveat: only 15/300 frames yield a valid essential matrix — a close-range bench scene with a
moving hand has little parallax-rich static structure. Sparse, not wrong.)

### ❌ The Part C premise FAILED: real depth does NOT fix the wrist z-jitter

Placing the wrist by sampling UniDepth at the WiLoR wrist keypoint (confidence-weighted
median over a 7×7 patch), back-projected with the real `K`:

```
                          x       y      z (mm/frame)
WiLoR bbox pseudo-depth   0.1     0.1    10.7
UniDepth measured depth   5.6     9.0    13.4     <- z got 25% WORSE
```

Real metric depth did not fix the trajectory — it is still white noise. Two hard measurements:

- **Static-point test (physics, not another estimate):** 281 background points tracked across
  all frames — their depth *cannot* change, yet UniDepth's reading of them wobbles **2.14%
  frame-to-frame, 10% full swing (0.76→0.85 m)**. That is ~13 mm of pure model noise at the
  hand's 0.58 m, and it sits **above** the real per-frame hand motion (1.7–10 mm).
- **Horizon sweep (k = 1…64 frames):** coherence stays at chance at *every* horizon,
  including the 16-frame / 0.53 s action chunk (raw cos −0.31, scale-normalised −0.01), and
  per-second speed decays toward zero — the signature of a random walk with no net drift.
  Aggregating over the action chunk does **not** recover coherent motion.

Dividing out the global scale error (measured from the static points) barely helps
(z 13.4→10.0 mm, coherence still chance): the residual is **local** depth error at a small,
moving, articulated object, not a global scale term background-anchoring could cancel.

### Chosen fix — fit the hand cloud + smooth (`solve_root_depth` / `smooth_root_depth`)

Instead of one wrist pixel, solve **one root depth per frame from all 21 keypoints**: each
keypoint i predicts the wrist depth as `D(kp_i) − dz_i` (WiLoR's trusted root-relative depth
offset), robust-averaged confidence-weighted. Then temporally low-pass the root depth. Result
on the real capture:

```
                                   z-jit    cos(k=1)  rev   |  cos(k=16, action chunk)
wrist-pixel sampling (rejected)    13.4 mm   +0.03    49%   |     -0.31   (chance)
21-keypoint fit, per-frame         16.1 mm   -0.17    53%   |       —
21-keypoint fit + temporal smooth  11.3 mm   +0.75    20%   |     -0.16   (still chance)
```

The fit+smooth path is a **real, measured improvement** — the first version that is coherent
at short scale (cos +0.75 at k=1) with a plausible 0.14 m/s speed, and lower z-jitter. **But
it does not fully reconstruct the action-chunk trajectory:** coherence decays with horizon
(0.75 → 0.50 → 0.20 → −0.16 at k=16), and the decaying shape is partly the smoothing kernel
manufacturing short-range autocorrelation. At the 0.53 s action-chunk horizon the trajectory
is still at chance.

**Conclusion.** Real intrinsics + real metric depth were *necessary* (they fix the SLAM
rotation and the hand's absolute placement) but are **not sufficient** to reconstruct the
wrist *trajectory* on a monocular bare-hand rig. The binding floor is UniDepthV2's ~13 mm
single-image depth noise at the hand (measured against static points), which exceeds the real
per-frame hand motion; the hand-cloud fit + smoothing takes it as far as monocular allows and
no further. WiLoR's hand remains excellent laterally and in articulation (0.1 mm x/y). The
honest position for delivery: **metric wrist *translation* is not trustworthy from this rig**
— stamp it VISION_FALLBACK and lean on the measured-robotspace rigs (DexUMI) for that channel;
articulation and 2D are trustworthy. Solver is implemented and folded into the module; the
open item is whether a stereo/temporal-consistent depth model (FoundationStereo, or a
multi-frame monocular method) could push below the noise floor — deferred, not attempted.

Also corrected here: an earlier note claimed fx=551 (a single-frame reading) and confidence in
[0,1] — both wrong; the 90-frame median fx is 660 and confidence is unbounded. Module docstrings fixed.

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
