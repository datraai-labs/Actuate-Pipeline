# Actuate — Architecture Design Document (v1)

**A Production-Grade, Multi-Rig, Sensor-Fused Data Engine for Zero-Touch Robot Foundation Model Training**

> Version 1.0 · Internal Architecture Reference · DatraAI
> **STATUS: superseded as the forward plan by [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md), but NOT obsolete.** v1 defines the L0–L8 layer structure, the sensor-truth-priority principle, the rig taxonomy, and the certification model — all of which v2 keeps. Read v1 for *what the pipeline is*; read v2 for *what we build next*.
> Source PDF: [pdf/Actuate_Architecture_v1.pdf](pdf/Actuate_Architecture_v1.pdf)

**Scope:** egocentric head-mounted capture, UMI handheld gripper rigs, calibrated stereo rigs, instrumented-glove capture, and teleoperated robot data — explicitly **not** internet/web-scale video mining. **Target bar:** a customer (e.g. a frontier humanoid or manipulation lab) should be able to take delivered data and begin policy training with zero additional processing.

---

## 1. Executive Summary & Design Mandate

Most robotics data pipelines stop at the point of producing *labeled video*. That is necessary but not sufficient. The mandate here is stricter: a customer should be able to take a delivered dataset and begin training **without writing a single additional line of preprocessing code**.

The central design principle is **sensor-truth priority**. Unlike systems built to mine uncontrolled internet video — where no ground-truth sensor stream exists and every quantity must be estimated visually — this pipeline is built for *controlled capture*. **Wherever a real sensor signal exists, it is treated as ground truth and vision is used to fill gaps, not the reverse.** This one architectural choice is what allows the pipeline to exceed the physical reliability of a pure-vision engine, at the cost of requiring calibrated hardware rather than arbitrary found footage.

### What "training-ready" means here

- Every episode carries **metric-scale 3D geometry** for hands, grippers, and manipulated objects — not 2D pseudo-actions.
- Every episode carries a machine-checkable **quality certificate**: per-modality confidence, calibration completeness, kinematic feasibility, and an explicit retargeting-eligibility flag per target embodiment.
- Every episode is **retargeted**, not just recorded — delivered as executable joint trajectories for the customer's actual robot morphology, validated in simulation before delivery.
- **Delivery format is a customer-facing concern, not a pipeline concern**: the same canonical representation exports to LeRobot v3, RLDS, or a bespoke schema without touching upstream stages.
- **Nothing ships without consent and PII redaction resolved** — a hard, fail-closed gate, not a checklist item.

---

## 2. Why the EgoInfinity Model Doesn't Directly Transfer

EgoInfinity (Wang et al., 2026) is the strongest published reference for this problem class. Its core techniques — cross-module metric calibration, interaction-state-driven trajectory refinement, and an SE(3)-equivariant root-frame estimator trained entirely in simulation — are directly relevant and adapted throughout. But its governing constraint is different, and that difference cascades through nearly every layer.

### The core constraint difference

| Dimension | EgoInfinity (web video) | This pipeline (controlled capture) |
|---|---|---|
| Data source | Uncontrolled internet footage, arbitrary viewpoint | Purpose-captured: head-mount, UMI, stereo, glove, teleop robot |
| Ground-truth sensors | **None** — everything estimated visually | Real IMU, gripper aperture encoders, glove joint sensors, robot joint encoders where applicable |
| Camera assumption | Approximately static camera; excludes body-mounted footage | Explicitly **includes** moving, body-mounted, and handheld capture as a first-class case |
| Grasp-state detection | Vision heuristics only (2D mask overlap, 3D fingertip proximity) | Real gripper aperture / glove flex-sensor signal as **primary**, vision as fallback |
| Object identity | Auto-discovered from noisy video captions | Operator-tagged at capture time, cross-checked against vision |
| Scale | Bounded by internet corpus size, diversity-optimized | Bounded by capture program throughput, **precision-optimized** |
| Delivery target | Research artifact + interactive browser | Commercial delivery in customer's exact training schema, with certification |

### What we keep from EgoInfinity

- The **modular, component-replaceable** engine philosophy — every perception module (hand mesh, depth, segmentation) is swappable as stronger foundation models emerge.
- **Cross-module metric calibration** as a first-class step, not an afterthought.
- The **interaction-state-driven refinement** pattern (static / grasped / moving) as the backbone of trajectory stabilization, extended in §6 with real sensor overrides.
- The **SE(3)-equivariant, flow-matching root-frame estimator** for cross-embodiment retargeting (§9), trained in simulation exactly as EgoInfinity does, but conditioned on rig type.
- The discipline of reporting **kinematic feasibility, contact consistency, and real-robot validation** as the actual bar for "does this work," not just perceptual accuracy.

### What we change or add

- **Sensor fusion arbitration** (§6) — a trust hierarchy where real sensor ground truth overrides vision-only estimates whenever present.
- **Rig-aware ingestion** (§4) — five structurally different raw formats normalized into one schema before any perception model runs.
- **Moving-camera support as default**, not an excluded case.
- **A formal quality certificate per episode** (§8) — a customer-facing, machine-readable trust artifact.
- **Customer-format delivery as a pluggable adapter layer** (§11).

> EgoInfinity answers "how do we extract robot-usable signal from video we don't control." This pipeline answers "how do we extract the highest-fidelity signal possible from capture we **do** control, and prove it's trustworthy enough to train on without human review." **The techniques transfer; the trust model does not.**

---

## 3. Capture Rig Taxonomy

Five structurally distinct capture configurations converge through a rig-aware ingestion adapter into one canonical stream schema.

| Rig type | Native channels | Primary use |
|---|---|---|
| **Head-mounted egocentric** | RGB (mono or stereo), head IMU, optional audio | Bare-hand manipulation at scale; broad task/environment diversity |
| **UMI handheld gripper** | Wrist-mounted RGB (fisheye, 1+ cameras), gripper aperture encoder, wrist IMU | Gripper-realistic manipulation without full robot hardware |
| **Stereo depth rig** | Calibrated stereo RGB pair, real metric depth stream | Highest-confidence 3D reconstruction; calibration reference set |
| **Instrumented glove** | RGB, per-finger flex/joint-angle sensors, glove IMU, optional tactile array | **Ground-truth finger kinematics independent of vision** |
| **Teleoperated robot** | Multi-camera RGB(+D), native robot joint encoders, gripper state, end-effector force | Directly executable action labels; no retargeting required |

### Why this taxonomy matters architecturally

Every layer from L1 onward must declare, **per rig type, which of its inputs are *estimated* versus *measured***. A head-mounted session has no grasp ground truth at all. A UMI session has a real aperture encoder. This distinction is carried as a first-class field — **`ground_truth_channels`** — on every session from ingestion onward, and every downstream confidence score is computed differently depending on it.

---

## 4. Layer 0 — Multi-Rig Ingestion & Canonicalization

### 4.1 Container format

Raw multi-channel capture is ingested into a per-session **MCAP-style container** — one session per recording, one topic per channel, each message carrying a hardware timestamp. Storing raw provenance in this form — rather than only the processed derivative — is what allows **re-processing an entire historical corpus when a perception module improves, without re-capturing anything.**

### 4.2 Canonical stream schema

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | Globally unique, immutable once assigned |
| `rig_type` | enum | `head_mounted` / `umi_gripper` / `stereo` / `glove` / `teleop_robot` |
| `ground_truth_channels` | list[enum] | Which of `{grasp, depth, finger_pose, joint_state}` are **hardware-measured, not estimated** |
| `camera_streams[]` | list | One entry per physical camera: intrinsics, extrinsics-if-known, resolution, native fps |
| `imu_streams[]` | list | Mount location (head / wrist / glove), sample rate, units |
| `encoder_streams[]` | list | Gripper aperture, joint angle, or robot joint encoder, as applicable |
| `timestamp_basis` | string | Shared clock domain all streams are aligned to |
| `calibration_ref` | string | Pointer to intrinsics/extrinsics file, or explicit `"none — approximated"` |

### 4.3 Synchronization

All streams are resampled onto a single, shared timestamp axis at ingestion, using hardware timestamps where the rig provides them (preferred) and interpolation against the highest-rate stream otherwise. **Millisecond-level synchronization precision is enforced as a hard requirement** carried into the Layer 4 quality gate — sessions whose native hardware cannot support this are flagged, not silently accepted at lower precision.

### 4.4 Calibration ingestion

Camera intrinsics/extrinsics are ingested **per physical device, not per session** — a rig's calibration is a property of the hardware, re-used across every session until recalibration. Sessions lacking a calibration reference fall back to an approximated pinhole model, and **this fact is carried forward explicitly as a lowered-confidence flag** rather than silently treated as equivalent to real calibration.

---

## 5. Layer 1 — Perception & Metric Reconstruction

Every component follows EgoInfinity's precedent of using strong, **swappable, off-the-shelf foundation models** rather than training bespoke perception networks — the pipeline's value is in orchestration, calibration, and fusion.

### 5.1 Metric camera and depth calibration
- **Stereo rigs:** real calibrated depth stream consumed directly — the highest-confidence source.
- **Monocular rigs** (head-mounted, UMI, glove): a metric monocular depth/geometry model estimates per-frame depth and focal length, calibrated against co-located stereo reference data where available.
- **Gravity direction** estimated per session, used later for retargeting root-frame estimation and camera-motion-aware egocentric reframing.

### 5.2 Hand pose and mesh estimation
A metric hand-mesh estimator recovers per-frame hand pose, shape, and 3D keypoints from RGB. **For instrumented-glove sessions, this vision estimate is *never* treated as the primary signal** — it exists only as a cross-check against the glove's own joint-angle sensors (Layer 2). For all other rig types it is the primary hand-kinematics source, with biomechanical joint-limit clamping.

### 5.3 Object discovery, segmentation, and tracking
Target objects identified from **operator-provided task metadata (preferred**, since capture is controlled) or an open-vocabulary detector. Segmented and tracked by a promptable video segmentation model, lifted into per-frame 3D point clouds using calibrated depth. Where sufficient unoccluded views exist, a canonical object mesh is reconstructed once per object and reused.

### 5.4 Two-pass processing for throughput
A lightweight first pass scans for hand/gripper presence and motion activity to filter out dead time; the full reconstruction stack runs **only on active segments**. Keeps compute cost proportional to actual manipulation content, not raw recording duration.

---

## 6. Layer 2 — Sensor-Fused Interaction Refinement

The layer that most substantively departs from a pure-vision engine, and **the core of this pipeline's reliability advantage.**

### 6.1 Trust-weighted fusion arbiter

For every frame, each candidate signal — vision (mask overlap, fingertip proximity), gripper aperture encoder, glove joint angle, contact/force sensing — reports both a value and a native confidence. The arbiter applies a **fixed trust ordering**: a hardware-measured grasp signal is *authoritative*, and vision is used only to resolve ambiguity vision can uniquely see (e.g. *which* object is grasped). On rig types with no hardware channel (head-mounted bare-hand), the arbiter degrades gracefully to a vision-only heuristic, **with the resulting lower trust tier explicitly recorded rather than silently equated with a sensor-confirmed grasp.**

### 6.2 Interaction states
Six-state internal representation (static-global, static, grasped-left, grasped-right, grasped-both, moving), collapsed to three coarse states downstream (static / grasped / moving) for pose-source selection, with majority-vote-then-proximity-tiebreak for ambiguous bimanual frames.

### 6.3 Pose-source selection by state

| State | Translation source | Rotation source |
|---|---|---|
| Static | Robust point-cloud centroid (outlier-filtered) | Canonical mesh orientation |
| Grasped (vision-only rig) | Hand-relative rigid bind, aggregated over the grasp segment | Same rigid bind |
| Grasped (sensor-confirmed rig) | Gripper/glove kinematic chain forward-computed from encoder + wrist pose | Same forward-kinematic chain |
| Moving | Smoothed mask-bbox back-projection | Per-frame PCA, sign-corrected |

### 6.4 Sanity filtering
A **scale-sanity check** (overrides implausible monocular object scale with mask-implied scale) and a **spurious-detection check** (flags objects whose 3D position is inconsistent with any hand activity). **Flagged objects are dimmed in the delivered representation, not silently dropped** — the certification layer surfaces this rather than hiding it.

---

## 7. Layer 3 — Canonical 4D Multimodal Representation

**The single most important interface in the whole architecture**: everything upstream is rig-specific, everything downstream (retargeting, annotation, packaging) is rig-agnostic and reads only this schema.

### 7.1 Per-frame episode state

| Field | Description |
|---|---|
| `hand_mesh` / `gripper_state` | Full hand mesh + keypoints, or gripper aperture + wrist pose, depending on rig |
| `hand_pose` (SE(3)) | Global metric pose of the acting end-effector |
| `object_point_cloud` | Per-frame back-projected, outlier-filtered point cloud |
| `object_mesh` | Canonical reconstructed mesh, one per object per clip |
| `object_pose` (SE(3)) | 6-DoF object pose in the shared metric camera frame |
| `interaction_state` | static / grasped / moving, plus dominant-hand / side resolution |
| `ground_truth_provenance` | **Per-field flag: measured vs. estimated, and which sensor if measured** |
| `per_field_confidence` | Numeric confidence **per field**, not just a single episode-level score |

### 7.2 Coordinate conventions
Right-handed camera-world frame, gravity-aligned using the Layer 1 gravity estimate.

### 7.3 Egocentric-consistent reframing
For fixed/exocentric capture, an optional **rigid, geometry-preserving reframing** into a synthesized egocentric view — re-rendering from a virtual camera anchored to hand activity, **rather than a generative 2D video translation that risks pixel hallucination.** Never fabricates visual content.

---

## 8. Layer 4 — Quality Certification & Fail-Closed Gating

Turns "we processed your data" into "here is exactly how much to trust every part of it."

### 8.1 Certificate composition

| Component | What it measures |
|---|---|
| Sync integrity | Max/mean timestamp drift across all streams against the shared clock |
| Calibration completeness | Real vs. approximated intrinsics/extrinsics; gravity estimate confidence |
| Perception confidence | Aggregated per-field confidence from L1/L2, weighted by ground-truth provenance |
| Contact consistency | Cross-check between hardware grasp signal (if present) and vision-estimated contact |
| Kinematic feasibility | Fraction of frames for which downstream IK (§9) converges within tolerance |
| **Retargeting eligibility** | **Per-target-embodiment boolean**, driven by depth/calibration confidence thresholds |
| Consent status | Explicit granted / pending / denied — packaging is **blocked** until granted |

### 8.2 Fail-closed delivery gate
Two independent gates must both clear: an overall quality score above a configurable floor, **and** an explicit consent grant. **Neither gate can be bypassed by a default value** — a missing or pending consent status blocks packaging outright, and a low quality score routes the episode to human review rather than silently downgrading its recommended use.

### 8.3 Recommended-use tiers
- **Fine-tuning / direct policy training** — highest tier: real ground-truth channels present, high perception confidence, retargeting-eligible.
- **Pretraining only** — usable for representation learning but not held to the same contact/kinematic precision bar.
- **Quarantined / needs review** — certificate below floor, or an unresolved sensor disagreement (e.g. vision says grasped, encoder says open) — **never silently resolved one way.**

---

## 9. Layer 5 — Cross-Embodiment Retargeting Engine

Converts an agent-agnostic 4D representation into motion a specific customer's robot can actually execute. Follows EgoInfinity's **functional-retargeting** philosophy: rather than requiring exact human body-pose recovery (often unavailable), the engine estimates a feasible robot-specific root transformation and preserves task-relevant end-effector motion within the target robot's own kinematic constraints.

```
Canonical 4D → Root-Frame Estimator → Candidate Clustering → IK Solve & Scoring
             → Sim Feasibility Validation → Embodiment Joint Trajectory
                              ↓
                  Per-Embodiment Adapter Library
```

### 9.1 SE(3)-equivariant root-frame estimator
A simulation-trained neural estimator predicts a shared kinematic root frame, conditioned on recovered bilateral end-effector trajectories and the L1 gravity estimate. Built from **Vector-Neuron layers** for exact rotation-equivariance, trained with a **flow-matching objective rather than deterministic regression** — because the same end-effector motion is consistent with multiple plausible torso/root poses under partial-body observation, and a single deterministic answer would **silently discard that ambiguity** rather than let downstream scoring resolve it.

### 9.2 Per-embodiment adapter library
Each target robot morphology is a separately trained root-frame estimator plus a kinematic model for IK — an **embodiment adapter**. New embodiments are onboarded by procedurally generating paired hand-trajectory / ground-truth-root-pose training data in simulation, then registering the adapter. This is the mechanism by which a new customer's robot becomes a supported delivery target **without re-architecting the pipeline**.

### 9.3 Candidate selection and IK scoring
Multiple root-frame hypotheses sampled and clustered; each scored by running the full-trajectory IK solve and evaluating convergence rate, residual tracking error, manipulability, joint-limit margin, and trajectory smoothness. Highest-scoring candidate selected, blended with per-frame estimates to preserve genuine root motion, and smoothed.

### 9.4 Simulation feasibility validation
Every retargeted trajectory is replayed in a physics simulator against the target embodiment's actual kinematic and collision model. **This is not optional polish — it is the concrete, checkable claim underlying the retargeting-eligibility field.** A trajectory that fails simulation replay does not ship as retargeting-eligible for that embodiment, even if it ships successfully for a different one.

### 9.5 Finger-level retargeting for dexterous end-effectors
Arm joints follow the wrist-level IK target; finger joints retargeted **separately and directly** from recovered hand keypoints (or real glove joint angles) via a geometry-based, robot-specific finger mapping. **Arm-level and finger-level retargeting are deliberately decoupled**, since they are governed by different kinematic constraints.

> **v2 supersedes §9.5.** The geometry-only finger mapping is replaced by the dexterous cascade — see [ARCHITECTURE_V2.md §7](ARCHITECTURE_V2.md#7-layer-5-expansion--dexterous-retargeting-cascade).

---

## 10. Layer 6 — Language & Task Annotation

### 10.1 Episode-level summary
Task family, task name, natural-language task description, scene description, success/failure with reason, and a **canonical object list**. Object references throughout all annotation layers are constrained to this list — **free-form object naming is not permitted**, to keep annotations auditable against the video.

### 10.2 Subtask segmentation
Coherent subtasks with start/end timestamps, generated by a VLM **grounded in actual sampled frames** — not filled into a fixed template. Held to a concreteness bar: *"the right gripper lifts the red cube, transports it above the bin, and releases it"* — **not** *"handles object."*

### 10.3 Fine-grained action intervals and fixed ontology
Beneath each subtask, action intervals use a **closed, fixed action vocabulary**: `reach, grasp, hold, lift, transport, lower, place, align, stabilize, release, open, close, insert, remove, push, pull, rotate, wipe, pour, idle`.

This closed-vocabulary constraint is **deliberately different** from the open-vocabulary task-naming in §10.1: **task *identity* should be open and descriptive, but atomic *action primitives* should be closed and standardized**, since that is what makes cross-episode and cross-dataset comparison possible.

### 10.4 Actor labeling for bimanual and multi-actor scenes
Every action-interval row attributed to a specific actor (left/right gripper, left/right hand, left/right fingers, controller, head, torso, body), constrained to actors for which a real tracking stream exists for that rig. **Concurrent bimanual actions are separate, potentially overlapping rows — never collapsed into one.**

### 10.5 Hallucination and consistency checking
Every LLM-generated field is checked against the structured facts and sampled frames it was conditioned on. **A fluent but ungrounded description is exactly as untrustworthy as a low-confidence pose estimate**, and treated as seriously.

---

## 11. Layer 7 — Dataset Packaging & Delivery

Packaging is deliberately the **last** layer to touch a customer-specific concern.

### 11.1 Structured delivery formats
- **LeRobot v3** — chunked structured data, synchronized video assets, episode metadata.
- **RLDS** — for Open X-Embodiment convention stacks.
- **Raw provenance (MCAP)** — original per-episode raw capture + channel manifest + raw→episode ID mapping, **delivered as required provenance, not an optional extra.**
- **Customer-specific schema adapters** — a thin export adapter per customer format, reading **only** from the L3 canonical representation + the L4 certificate, so a new customer format never requires touching perception or retargeting code.

### 11.2 Manifests
Episode manifest, stream manifest, and dataset-level manifest generated for every delivery.

### 11.3 Dataset-level quality operations
- **Deduplication** — cross-episode near-duplicate detection via embedding similarity, applied **before** splits so near-duplicates cannot leak across train/val/test.
- **Diversity and balance reporting** — object-class diversity, task-distribution imbalance ratio, environment/operator diversity.
- **Stratified splits** — train/val/test at the episode level, stratified by task, dedup results consulted first.

### 11.4 Consent and provenance as delivery-blocking, not advisory
The packaging stage **re-verifies consent at the moment of delivery**, not only at initial processing — a session whose consent was revoked between processing and delivery **must not ship**, even if processed and certified earlier.

---

## 12. Layer 8 — Feedback Loop & Continuous Improvement

### 12.1 Human-in-the-loop review routing
Episodes quarantined by L4, flagged by the L6 hallucination check, or exhibiting a sensor-vs-vision disagreement in L2 are **routed to a review queue rather than silently resolved**. Reviewer decisions (accept, correct, discard) captured as **structured feedback, not just a binary approval**.

### 12.2 Component versioning and reprocessing lineage
**Every** perception, refinement, and retargeting component carries an explicit version identifier, recorded **per output artifact** — not one blanket pipeline version. When a component is upgraded, the set of historical sessions whose outputs depended on the old version is **queryable directly from this lineage**, making selective reprocessing tractable rather than requiring a full corpus re-run.

### 12.3 Calibration drift monitoring
Per-device calibration periodically re-validated against fresh reference captures; a drifted device is flagged, and sessions captured after the drift point are marked with reduced calibration-completeness confidence until recalibration is confirmed.

---

## 13. Infrastructure & Scaling

- **13.1 Compute topology** — Perception (L1) and retargeting inference (L5) are GPU-bound and stateless per episode → horizontally scalable worker pools behind a job queue. Ingestion and certificate/packaging are CPU-bound and run on cheaper compute.
- **13.2 Job orchestration** — Queue-based workers; jobs tracked through a **persistent status record (not only in-memory)** and safely resumable — a worker restart mid-run must not lose track of which stages completed.
- **13.3 Storage** — Raw provenance (MCAP) in cold, high-durability object storage, kept indefinitely for reprocessing lineage. Canonical representation + intermediates in warm storage. Delivered packages in versioned, customer-scoped storage with signed, time-limited access.
- **13.4 Cost proportionality** — Two-pass processing + the layered architecture ensure compute cost scales with **actual manipulation content** and requested delivery formats, not raw capture volume or the number of export adapters.

---

## 14. Comparative Positioning

| Dimension | EgoInfinity | Open X-Embodiment / DROID | This pipeline |
|---|---|---|---|
| Data source | Internet video | Teleoperated robots | Multi-rig controlled capture (5 types) |
| Ground-truth sensing | None | Full (native robot) | Partial-to-full, per rig type, sensor-fused |
| Scale driver | Corpus size | Collection cost/hardware | Capture program throughput |
| Retargeting | Functional, cross-embodiment | N/A (native to one robot) | Functional, cross-embodiment, **sim-validated** |
| Certification | Sanity checks only | Not standardized | **Formal per-episode certificate, fail-closed** |
| Delivery format | Research browser + raw output | RLDS-family | Pluggable — LeRobot v3 / RLDS / customer schema |
| Consent/PII handling | Not addressed | Not addressed | **Fail-closed gate, delivery-blocking** |

> EgoInfinity proves the underlying techniques are sound; Open-X/DROID prove the market values this data category; **this architecture is the synthesis aimed at production delivery.**

---

## 15. Appendix — Schemas, Thresholds & Coordinate Conventions

### A.1 Consolidated tunable parameters

| Parameter | Purpose |
|---|---|
| `sync_drift_threshold_ms` | Maximum allowed cross-stream timestamp drift before sync-integrity flag fires |
| `grasp_trust_priority` | Ordered list: **hardware-confirmed > vision-2D-overlap > vision-3D-fallback** |
| `scale_sanity_factor` | Maximum allowed ratio between monocular and mask-implied object scale before override |
| `spurious_distance_m` / `spurious_2d_motion_px` | Thresholds for flagging a tracked object as a likely background false match |
| `retarget_ik_convergence_floor` | Minimum per-frame IK success rate for an embodiment to be marked retargeting-eligible |
| `cert_quarantine_floor` | Minimum composite certificate score for an episode to be delivery-eligible at all |
| `dedup_similarity_threshold` | Embedding similarity above which two episodes are flagged as near-duplicates |

### A.2 Coordinate convention
Right-handed camera-world frame (**+x right, +y down, +z into scene**), gravity-aligned per session.

### A.3 Ground-truth provenance enum

Every field in the Layer 3 canonical representation carries one of:

```
measured_hardware | estimated_vision_primary | estimated_vision_fallback | approximated
```

**This single field is what every downstream confidence computation — from the Layer 4 certificate through customer-facing delivery documentation — is ultimately keyed on.**

---

> *End of document. This is a living architecture reference — component versions, thresholds, and embodiment adapters are expected to evolve as described in §12, without requiring changes to this document's layer boundaries.*
