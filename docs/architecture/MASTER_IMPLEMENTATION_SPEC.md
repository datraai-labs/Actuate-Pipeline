# Actuate — Master Implementation Specification

**Buildable Reference: Architecture, Tech Stack, Interfaces, and Sequenced Build Plan**

> Consolidates v1 (L0–L8 pipeline), v2 (VLA + dexterous), v3 (10-paper merge), v3.1 (EgoScale + EgoVerse).
> **THIS IS THE DOCUMENT WE BUILD FROM.** It supersedes ARCHITECTURE_V2.md and IMPLEMENTATION_SPEC.md as the plan of record; those remain useful history.
> Spec only — nothing herein is built; every benchmark-before-lock and GPU-validation point is flagged.
> Source PDF: [pdf/Actuate_Master_Implementation_Spec.pdf](pdf/Actuate_Master_Implementation_Spec.pdf)

---

## 0. How to read this document

For each layer it gives: **status** (built / extend / net-new), **tech stack** (specific libraries and models, with "benchmark-before-lock" marked), **public interface**, **inputs/outputs** (typed against the canonical schema in §3), and **verification gate** (what must pass, and what needs real GPU/real-data validation before it counts as done).

Build order is §8. **Start at Phase 0 — the canonical schema in §3 is the freeze point everything else compiles against.**

### Two non-negotiables carried from the whole project

1. **CLI/API-first.** Everything is a Python library function first. The Typer CLI and the FastAPI service are thin consumers of that library — never dependencies of it. **If a capability only works through the CLI or only through the service, it is wrong.**
2. **Verify against real data.** A component is "done" only when tested against real data, and **every correctness test must be confirmed to fail against a broken version, not merely pass against the current one.** "Looks right" is not a status.

---

## 1. Build status ledger

| Component | v1 status | Target state | Delta type |
|---|---|---|---|
| Ingestion, sync, QA | Built | Add 6th rig (DexUMI exoskeleton), aligned-capture mode, SLAM/IMU ego-motion, phone tier | Extend |
| Hand pose (MediaPipe) | Built | Swap to **WiLoR→MANO** (HaMeR fallback) | Extend (model swap) |
| Object detect/track (Grounding DINO + SAM2) | Built | Keep; add FoundationPose 6-DoF object pose | Extend |
| Depth (Depth-Anything-V2) | Built | Swap to rig-specific: **UniDepthV2** (mono) / **FoundationStereo** (stereo); benchmark vs MoGe-2+FLOW3R+GeoCalib | Extend (model swap) |
| Primitives / phases / episode segmentation | Built | Keep; phase boundaries double as subgoal-image anchors | Keep |
| VLM language grounding | Built | Add multi-paraphrase, structured hand+object+action captions, LLM-as-judge gate | Extend |
| EIS certification | Built | Map to π0.7 episode metadata; add strategy-alignment item | Extend |
| PII redaction + fail-closed consent gate | Built | **Untouched — hard-blocking** | Keep |
| Dataset packaging | Built (format-generic) | LeRobot v3 + RLDS exporters, rich norm stats, Stage-I/Stage-II tiers | Net-new export |
| FastAPI service + web dashboard | Built | Surface new fields/formats | Extend (consumer) |
| **Sensor-fusion arbiter (L2)** | — | Trust-weighted arbiter + per-finger contact confidence | **Net-new** |
| **Canonical schema** (MANO + retargeted joints, both action spaces, metadata) | Partial | Frozen, versioned | **Net-new / rework** |
| **Retargeting engine (L5)** | — | Arm (VN+flow), finger (GeoRT), reconciliation, sim validation | **Net-new** |

---

## 2. Package architecture (CLI/API-first)

### 2.1 Repository layout

```
actuate/            # importable library — the source of truth
  __init__.py
  config/           # Pydantic-settings, YAML profiles, rig registry
  schema/           # canonical data contracts (§3) — msgspec/pydantic
  io/               # MCAP, PyAV, Zarr/HDF5, Parquet readers/writers
  ingest/           # L0
  perception/       # L1: hand/ object/ depth/ slam/ pose submodules
  fusion/           # L2
  canonical/        # L3
  certify/          # L4: quality/ pii/ consent/ llm_judge/ strategy_align
  retarget/         # L5: arm/ finger/ reconcile/ sim_validate/ embodiments
  language/         # L6
  package/          # L7: lerobot/ rlds/ normalize/ tiers
  feedback/         # L8
  cli/              # Typer app (thin)
  service/          # FastAPI app (thin consumer)
tests/
  unit/             # fast, mock-free where possible
  integration/      # real-data fixtures, small real clips
  fixtures/         # tiny real capture samples per rig
pyproject.toml      # uv/pip; extras per layer to keep deps optional
```

**Dependency direction:** `cli/` and `service/` import from the layer packages; layer packages import only from `schema/`, `io/`, `config/`, and each other in pipeline order. **Nothing in a layer imports from `cli/` or `service/`. Enforce with an import-linter contract in CI.**

**Optional heavy deps:** perception/retarget models (torch, CUDA libs, sim engines) live behind extras (`actuate[perception]`, `actuate[retarget]`, `actuate[sim]`) so the core library and exporters install without a GPU stack.

### 2.2 CLI surface (Typer)

Each layer is one command group; every command is a thin wrapper over a library call.

```bash
actuate ingest run        --rig <rig> --in <path> --out <store>
actuate perceive hands    --in <store> --model wilor
actuate perceive depth    --in <store> --model auto        # auto = rig-specific
actuate perceive objects  --in <store> --prompts prompts.yaml
actuate perceive slam     --in <store>                     # moving-camera rigs
actuate fuse run          --in <store>
actuate canonical build   --in <store> --out <canonical>   # freezes to schema vN
actuate certify run       --in <canonical>                 # quality + consent + judge + strategy
actuate retarget arm      --in <canonical> --embodiment <name>
actuate retarget finger   --in <canonical> --embodiment <name>
actuate retarget validate --in <canonical> --embodiment <name>   # sim replay
actuate language annotate --in <canonical>
actuate package lerobot   --in <canonical> --out <dataset> --tier stage1|stage2
actuate package rlds      --in <canonical> --out <dataset>
actuate run all           --config profile.yaml            # full pipeline
```

**Design rules:** every command reads/writes the canonical store (§3), **is idempotent, records provenance**, and takes `--dry-run` and `--limit N` for cheap real-data spot-checks. Long/GPU stages print an explicit "requires GPU" banner and a resumable checkpoint path.

### 2.3 Config

Pydantic-settings with layered YAML profiles (`base.yaml` → `rig/<rig>.yaml` → `embodiment/<name>.yaml`).

- A **rig registry** maps each capture rig to its native channels, ego-motion method, depth model, and trust ordering.
- An **embodiment registry** maps each robot to its kinematic model (URDF), hand type/DoF, control modes, and camera config.

**These two registries are the extension points for new rigs and new customer robots.**

---

## 3. Canonical schema — THE FREEZE POINT, build this first

The contract every downstream layer compiles against. Freeze and version it (`schema_version`) **before** building L4–L7. Use `msgspec.Struct` (fast, typed, Parquet/Zarr-friendly) or Pydantic v2.

### Per-frame record

| Field | Type | Source | Required by |
|---|---|---|---|
| `t` | float64 (s, canonical clock) | L0 | all |
| `rig`, `episode_id`, `frame_idx` | str/str/int | L0 | all |
| `images.<cam>` | ref to chunked MP4 (consistent cam names) | L0 | VLA output |
| `camera_pose` | SE(3), world frame | L1 SLAM / MPS | egocentric reframing |
| `hand.<L/R>.mano` | MANO params (β fixed, θ 15-PCA) | L1 WiLoR | EgoVLA-style pretrain (**intermediate**) |
| `hand.<L/R>.keypoints_3d` | 21×3, camera frame | L1 | fingertip retarget input |
| `hand.<L/R>.wrist_pose` | SE(3), camera frame | L1 | wrist/EEF action |
| `finger_joints_human` | per-finger joint angles | L2 (glove) / L1 (vision) | dexterous |
| `finger_joints_robotspace` | joint angles in robot hand space | L2 (DexUMI exoskeleton) | dexterous (**measured GT**) |
| `object.<id>.pose` | SE(3) | L1 FoundationPose | interaction, contact |
| `object.<id>.mask` | RLE/ref | L1 SAM2 | grounding |
| `depth.<cam>` | ref to depth map + uncertainty | L1 | metric 3D |
| `contact.<finger>` | confidence ∈ [0,1] + source enum | L2 | **dexterous prerequisite** |
| `interaction_state` | enum {STATIC, GRASPED_L/R/BOTH, MOVING} | L2 | fusion |
| `confidence.<field>` + `provenance.<field>` | float / source-tag | all | certification |

### Per-episode / per-segment record

| Field | Type | Source | Required by |
|---|---|---|---|
| `task` | str (imperative) + `task_paraphrases[]` | L6 | VLA |
| `subtasks[]` | list of (span, instruction) | L6 | π0.7 subtask/subgoal anchor |
| `subgoal_frames[]` | frame refs at phase boundaries | L6 | π0.7 subgoal images |
| `action.human` | relative-SE(3) wrist + **retargeted joints on canonical reference hand** | L5 | Stage-I pretrain target |
| `action.robot.<embodiment>` | joint traj **and** EE traj, per control mode | L5 | Stage-II / finetune |
| `control_mode` | enum {joint, ee} tag | L5 | π0.x contract |
| `finger_action_repr` | {absolute, relative} **both available** | L3 | DexUMI robustness |
| `episode_meta.quality` | 1–5 (from L4 EIS) | L4 | π0.7 metadata |
| `episode_meta.speed` | length in steps (binned) | L4 | π0.7 metadata |
| `episode_meta.mistakes[]` | per-segment flags | L4 | π0.7 metadata |
| `strategy_alignment.<embodiment>` | ok / flagged + reason | L4/L5 | EgoVerse Robot-B guard |
| `norm_stats` | per-dim 1/99 percentiles (+ mean/std) | L7 | training |
| `diversity.scene_id`, `diversity.demonstrator_id` | ids | L0 | L7 diversity reporting |
| `tier` | {stage1_volume, stage2_anchor} | L7 | delivery tiering |
| `effective_hours` | float (post-filter) | L7 | scaling-law value |
| `consent`, `pii_status` | enum (**fail-closed**) | L4 | **hard gate** |
| `schema_version` | str | — | migration |

### Key decisions baked in (from v3/v3.1)

**MANO is the *intermediate*.** The delivered pretraining action target is **relative-SE(3) wrist + retargeted joints on a canonical high-DoF reference hand** (pick one, ~20+ DoF). **Both control modes and both finger-action representations are carried.** Episode metadata mirrors π0.7. Strategy-alignment and consent are certification-gated.

> **Verification gate:** round-trip a **real** processed episode through `canonical build` → Parquet/Zarr → reload, asserting **bit-exact field recovery**; and **confirm the schema migration test FAILS when a required field is dropped.**

---

## 4. Per-layer implementation

### L0 — Ingestion & Sync · `actuate/ingest`
- **Status:** built; extend.
- **Stack:** MCAP (container), PyAV (decode), Polars + NumPy (tabular/sync), Zarr/HDF5 (frame store), Pydantic (rig manifests). Ego-motion: **ORB-SLAM3** (stereo+IMU rigs) or **Aria MPS**. Phone tier: commodity iPhone, cloud VI-tracking.
- **New:** **DexUMI-exoskeleton rig** (robot-space encoder joints + tactile); **aligned-capture mode** (human rig shares camera intrinsics/extrinsics with a target robot — required for Stage-II anchor); multi-paraphrase slot reserved.
- **Interface:** `ingest.run(rig, src, store) -> IngestReport`; per-rig adapter classes implementing `RigAdapter`.
- **Gate:** on a real multi-rig sample, assert every stream lands on one canonical clock within tolerance; **regression-test the known fps bug** (fixture where source fps ≠ metadata fps must be caught, not silently trusted).

### L1 — Perception & Metric Reconstruction · `actuate/perception`
- **Status:** built (MediaPipe hands, Depth-Anything-V2); extend with model swaps.
- **Stack:**
  - Hand → MANO: **WiLoR** (primary), HaMeR (fallback). *Benchmark-before-lock vs MediaPipe on real clips.* **GPU.**
  - Object detect/segment/track: Grounding DINO + SAM2 (built, keep). **GPU.**
  - Object 6-DoF pose: **FoundationPose** (new). **GPU.**
  - Depth (rig-specific): **UniDepthV2** (mono) / **FoundationStereo** (stereo, mm-level) / MoGe-2 + FLOW3R + GeoCalib. *Benchmark-before-lock — **do not assume the impl-spec default; EgoInfinity does not use UniDepthV2.*** **GPU.**
  - Ego-motion: ORB-SLAM3 / Aria MPS.
- **Gate:** each model validated on real frames against held-out GT or cross-sensor agreement. **All L1 model swaps need real GPU validation before they count as done.**

### L2 — Sensor-Fused Interaction Refinement · `actuate/fusion`
- **Status: net-new.**
- **Stack:** typed state machine (enums + transition table), SciPy/NumPy. **Trust-weighted arbiter with ordering: `measured_robotspace` (DexUMI) > `measured_human` (glove) > `gripper aperture` > `vision-primary` > `vision-fallback`.**
- **Output:** `interaction_state`, `contact.<finger>` confidence — **the hard prerequisite for the dexterous branch.**
- **Gate:** on real glove + vision episodes, assert measured channels override vision when present; **unit test the arbiter with a broken-priority variant that MUST fail.**

### L3 — Canonical Representation · `actuate/canonical`
- **Status:** partial; **rework to the frozen §3 schema.**
- **Stack:** msgspec/Pydantic v2 structs; Parquet (episode/metadata) + Zarr (dense per-frame); camera-centered **stable-frame reprojection** (NumPy).
- **Interface:** `canonical.build(store, out, schema_version) -> CanonicalDataset`.
- **Gate:** §3 round-trip test; stable-frame reprojection validated against a moving-camera clip.

### L4 — Certification · `actuate/certify`
- **Status:** built (EIS, PII, consent); extend.
- **Stack:** existing EIS scorer → emit π0.7 `quality`/`speed`/`mistakes`; **PII/consent fail-closed gate (untouched)**; **LLM-as-judge** caption-consistency gate; **strategy-alignment** check fed from L5 (flag if retargeting forces an undemonstrated strategy — EgoVerse Robot-B guard).
- **Gate:** **the consent gate must be re-tested against the auto-consent workaround class of bug** — a fixture that attempts to bypass consent must be blocked; strategy-alignment flag must trigger on a known mismatched episode.

### L5 — Cross-Embodiment Retargeting · `actuate/retarget`
- **Status: net-new.**
- **Stack:**
  - **Arm/wrist:** SE(3)-equivariant **Vector-Neuron** root-frame estimator, **flow-matching** objective, trained in **MuJoCo** (~1.5–2 hr/robot on a single 3060). IK: **cuRobo** (GPU) or **Pinocchio/PyRoki** (CPU). Candidate clustering (k-means over SE(3)) + scoring.
  - **Finger (kinematic):** **GeoRT** (public code) — per-finger fingertip→joint MLP, unsupervised, ~3–5 min/hand. Needs per-human ~5-min C-space calibration. Contact-blind → feeds reconciliation.
  - **Canonical reference-hand retarget:** MANO→canonical high-DoF hand (EgoScale-style) as the delivered Stage-I output.
  - **DexUMI bypass:** exoskeleton rigs **skip finger retargeting** (already robot-space).
  - **Contact-consistency reconciliation:** arm-IK wrist + finger posture must agree at contact.
  - **Sim validation:** MuJoCo (default) / Isaac Lab (scale) — collision, joint-limit, contact-stability, **no-slip**. *DexMachina's contact-guided idea informs **scoring** — not wholesale DexMachina RL (flagged research track, not on this path).*
- **Gate:** replay in sim, assert no-slip/collision/joint-limit pass on a real episode; **the reconciliation test must FAIL on a deliberately contact-inconsistent trajectory.** Training needs real GPU validation.

### L6 — Language & Task Annotation · `actuate/language`
- **Status:** built; extend.
- **Stack:** grounding VLM for structured hand+object+action captions; **multi-paraphrase** (1 human + N LLM, per TRI LBM); phase boundaries → subtask spans + **subgoal-frame anchors** (π0.7); closed action ontology beneath open task names.
- **Gate:** LLM-as-judge consistency on a labeled real subset; **a hallucinated-caption fixture must be flagged, not passed.**

### L7 — Packaging & Delivery · `actuate/package`
- **Status:** built (format-generic); **net-new exporters.**
- **Stack:** **LeRobot v3** exporter (chunked Parquet + per-camera chunked MP4 + tying metadata) primary; **RLDS/TFDS** secondary (*verify field layout against the real library, not from memory*); normalization: default **1st/99th-percentile → [−1,1]**, ship raw percentiles (+ mean/std) so customers on 2/98-per-timestep or z-score can re-derive; **Stage-I volume tier** and **Stage-II aligned-anchor tier**; dual-space delivery (human + retargeted robot); scene- and demonstrator-diversity reported **separately**; `effective_hours` recorded.
- **Gate:** **load the exported LeRobot v3 dataset with LeRobot's own loader and run one real training step**, including the normalization round-trip. **A schema-valid-but-untrainable export must be caught by this gate — do not assert schema-correctness in place of a real load+train.**

### L8 — Feedback · `actuate/feedback`
- **Status:** built; keep with one reframe: honestly-graded failure/low-quality episodes route to delivery as **π0.7 metadata-labeled robustness data** (not just quarantine). Consent/PII failures stay hard-blocked.

### Service & Dashboard · `actuate/service`
Built; extend as a **consumer**. FastAPI surfaces retarget-eligibility, per-finger contact, episode metadata, export formats, tier toggle. **Neither the service nor the dashboard is a dependency of the library.**

---

## 5. Cross-cutting concerns

- **Provenance & confidence on every derived field** — carried through L3, consumed by L4.
- **Idempotence & resumability:** every stage checkpoints and can resume; `--limit N` runs a cheap real-data slice.
- **Testing discipline:** `tests/unit` (avoid vacuous mocks), `tests/integration` (tiny **real** clips per rig). CI import-linter enforces the CLI/API-first dependency direction. **Every "proves correctness" test carries a companion broken-variant test that must fail.**
- **GPU boundary:** perception, retargeting training, sim validation are GPU stages behind extras; **the core library, schema, certification (non-VLM), and exporters run CPU-only so value ships without a GPU stack.**

---

## 6. Compute & GPU-validation map

| Stage | Compute | Needs real-GPU validation before "done" |
|---|---|---|
| L1 hands (WiLoR), objects, depth, pose | GPU inference | **Yes** — validate each swap on real frames |
| L1 SLAM (ORB-SLAM3) | CPU/GPU | **Yes** — on real moving-camera clips |
| L2 fusion, L3 canonical, L4 (non-VLM), L7 exporters | **CPU** | No (unit + real-data integration) |
| L4 LLM-as-judge, L6 VLM captions | GPU/API | **Yes** — consistency on labeled real subset |
| L5 arm estimator training (VN+flow, MuJoCo) | GPU (single 3060 sufficient) | **Yes** |
| L5 finger (GeoRT) | GPU (light) | **Yes** |
| L5 sim validation | GPU | **Yes** |
| L7 LeRobot round-trip train step | GPU (small) | **Yes — the load+train gate** |

Free-tier (Kaggle/Lightning) covers L5 arm/finger training and most L1 validation.

---

## 7. Third-party choices to lock via benchmark (NOT from memory)

1. **Depth model** — UniDepthV2 vs FoundationStereo vs MoGe-2+FLOW3R+GeoCalib, per rig, on real Actuate captures.
2. **Hand model** — WiLoR vs HaMeR vs incumbent MediaPipe, real clips.
3. **RLDS field layout & OpenVLA discrete-token option** — confirm against the real libraries.
4. **Canonical reference hand** — pick the ~20+ DoF hand used as the Stage-I retarget target.
5. **IK backend** — cuRobo (GPU) vs Pinocchio/PyRoki (CPU).
6. **EgoScale scaling-law constants** — refit on Actuate's own certified data; **do not reuse 0.024 / 0.003.**

---

## 8. Sequenced build plan

**Phase 0 — Schema & input analysis** (days, CPU). Frozen §3 canonical schema (`schema_version=1`), rig + embodiment registries. *Gate: §3 round-trip passes; drop-a-required-field migration test fails as expected.*

**Phase 1 — Canonical build against the freeze** (CPU). `canonical build` producing the frozen schema from already-processed v1 outputs; stable-frame reprojection. *Gate: real episode round-trips.*

**Phase 2 — LeRobot v3 / RLDS exporter with π0.7 rich context. SHIPS VALUE.** Exporters + normalization + episode metadata + multi-paraphrase + both control modes + Stage-I tier. Training-ready gripper/arm data independent of dexterous work. *Gate: **load with LeRobot's own loader + run one real training step + normalization round-trip.** First shippable milestone.*

**Phase 3 — L1 model swaps + L2 fusion** (GPU). *Gate: each model validated on real frames; arbiter broken-priority test fails.*

**Phase 4a — Kinematic dexterous track** (GPU). **Highest near-term risk.** Arm (VN+flow) + GeoRT finger + reconciliation + sim no-slip. **DexUMI-exoskeleton data flows end-to-end first (bypasses retargeting, de-risks).** *Gate: sim replay passes; reconciliation fails on a contact-inconsistent trajectory.*

**Phase 4b — Functional/RL research track** (flagged, parallel/after). DexMachina-style contact-guided refinement. **Not gated in front of launch.**

**Phase 5 — Language rich-context + aligned-anchor + delivery hardening.**

**Certification & consent (Phase 0 onward, ALWAYS-ON):** the fail-closed consent/PII gate is wired from the first canonical build and **re-tested against the auto-consent-bypass bug class at every phase.**

---

## 9. Honest status

**Nothing in this document is built or tested; it is the spec you build from.** The v1 pipeline (through packaging) exists and is the foundation; everything marked net-new or rework in §1 is unbuilt. Every model swap, retargeting component, sim validation, and the LeRobot train-step gate **needs real GPU/real-data validation before it counts as done.** The benchmark-before-lock items in §7 **must not be resolved from memory.**
