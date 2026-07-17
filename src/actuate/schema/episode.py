"""Per-episode canonical record — Master Spec §3, exactly.

Includes the two fields the whole delivery path is gated on — `consent` and `pii_status`
— and the π0.7 episode metadata (`quality`, `speed`, `mistakes[]`) that turns an honestly
graded bad episode into useful robustness data rather than landfill (§L8).

**Key decisions baked in, per §3:**
  - MANO is the *intermediate*. `action.human` is relative-SE(3) wrist + retargeted joints
    on a canonical high-DoF reference hand — NOT MANO, and NOT raw keypoints.
  - Both control modes (`joint`, `ee`) and both finger-action representations
    (`absolute`, `relative`) are carried. Never conflated, always tagged.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from actuate.config import (
    Actor,
    ActionVerb,
    Channel,
    ConsentStatus,
    ControlMode,
    FingerActionRepr,
    PiiStatus,
    RigType,
    Side,
    Tier,
    get_rig,
)
from actuate.schema.frame import SE3, CanonicalFrame
from actuate.schema.version import SCHEMA_VERSION

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Language (L6)
# --------------------------------------------------------------------------------------


class Subtask(_Base):
    """(span, instruction) — π0.7 subtask / subgoal anchor."""

    instruction: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    confidence: Unit | None = None


class SubgoalFrame(_Base):
    """Frame ref at a phase boundary → π0.7 subgoal image."""

    frame_idx: int = Field(ge=0)
    label: str | None = None


class ActionInterval(_Base):
    """One atomic action (Master Spec v1 §10.3-10.4). NEW in v5.

    Fine-grained and per-actor, distinct from the coarse L2 interaction_state
    (STATIC/GRASPED/MOVING) and from phase/subtask segmentation. `action_label` is drawn
    from the CLOSED 20-verb vocabulary; `actor` names which end-effector performs it.
    Joinable to per-frame pose/sensor data by [start_frame, end_frame]. Concurrent
    bimanual actions are separate rows with different actors and overlapping spans.
    """

    action_label: ActionVerb
    actor: Actor
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    confidence: Unit
    #: optional index into `subtasks` this interval falls within
    subtask_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _span_ordered(self) -> "ActionInterval":
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"action interval end_frame {self.end_frame} < start_frame "
                f"{self.start_frame}")
        return self


# --------------------------------------------------------------------------------------
# Actions (L5) — the delivered target
# --------------------------------------------------------------------------------------


class ReferenceHandAction(_Base):
    """Retargeted joints on the canonical high-DoF reference hand.

    Master Spec §7 item 4 makes picking that hand a benchmark-before-lock decision, and
    it is NOT yet picked (see config.embodiments.CANONICAL_REFERENCE_HAND is None). So
    `hand_name` is required here: an exporter must never have to guess which hand's joint
    ordering it is looking at, and a dataset delivered without that label is unusable.
    """

    hand_name: str
    joints: tuple[float, ...] = Field(min_length=1)
    finger_action_repr: FingerActionRepr


class HumanAction(_Base):
    """`action.human` — the Stage-I pretrain target (Master Spec §3).

    relative-SE(3) wrist + retargeted joints on the canonical reference hand.
    `reference_hand` is None until L5 exists; the wrist trajectory alone is still a
    legitimate Stage-I target for gripper/arm work (which is exactly why Phase 2 can ship
    before Phase 4).
    """

    wrist_delta: dict[Side, SE3]
    reference_hand: dict[Side, ReferenceHandAction] | None = None


class RobotAction(_Base):
    """`action.robot.<embodiment>` — joint traj AND EE traj, per control mode.

    Only L5 retargeting produces this. It does not exist yet, so this model is
    exercised by tests and by nothing else — stated plainly rather than implied.
    """

    embodiment: str
    control_mode: ControlMode
    joint_traj: tuple[tuple[float, ...], ...] | None = None
    ee_traj: tuple[SE3, ...] | None = None
    finger_action_repr: FingerActionRepr | None = None

    @model_validator(mode="after")
    def _has_a_trajectory(self) -> RobotAction:
        if self.joint_traj is None and self.ee_traj is None:
            raise ValueError(
                f"action.robot.{self.embodiment}: neither joint_traj nor ee_traj is present. "
                "An action that names a robot but contains no motion is not an action."
            )
        return self


# --------------------------------------------------------------------------------------
# Certification (L4)
# --------------------------------------------------------------------------------------


class CertificateComponents(_Base):
    """The measured inputs behind `quality` (Master Spec §4 L4). NEW in v4.

    Each is a [0,1] score, and `None` means NOT MEASURED — never zero. The distinction is
    the same one the contact fields carry: a bare-hand rig has no hardware grasp to
    cross-check, so its `contact_consistency` is None (nothing measured), not 0.0 (measured,
    catastrophic). Publishing the components, not just the composite, is what lets a customer
    dispute a score.
    """

    #: 1 - normalised cross-stream timestamp drift (L0 sync report).
    sync_integrity: Unit | None = None
    #: Real vs approximated intrinsics + SLAM confidence (L0+L1). Approximated intrinsics
    #: were a REAL bug (guessed fx=1104 vs measured ~660), so this scoring low is honest.
    calibration_completeness: Unit | None = None
    #: Weighted aggregate of per-field confidence from L1/L2.
    perception_confidence: Unit | None = None
    #: Hardware grasp vs vision contact cross-check (L2). None on rigs with no grasp sensor.
    contact_consistency: Unit | None = None
    #: Fraction of frames where IK converged during retargeting (L5).
    ik_convergence_rate: Unit | None = None


class EpisodeMeta(_Base):
    """π0.7 episode metadata (Master Spec §3, §L4).

    `quality` is 1-5, a weighted composite of `components` mapped to π0.7's scale. Low
    quality is NOT a reason to discard — §L8 routes honestly-graded failures to delivery as
    metadata-labeled robustness data. Consent/PII failures are the only hard block.
    """

    quality: int | None = Field(default=None, ge=1, le=5)
    speed: int | None = Field(default=None, ge=0, description="length in steps, binned")
    mistakes: tuple[str, ...] = ()
    #: v4: the measured components behind `quality` (all-None until certify.score runs).
    components: CertificateComponents = Field(default_factory=CertificateComponents)


class StrategyAlignment(_Base):
    """EgoVerse Robot-B guard: flag when retargeting forces an undemonstrated strategy."""

    ok: bool
    reason: str | None = None

    @model_validator(mode="after")
    def _flag_has_a_reason(self) -> StrategyAlignment:
        if not self.ok and not self.reason:
            raise ValueError(
                "strategy_alignment flagged without a reason. A flag no human can act on "
                "is noise — say what mismatched."
            )
        return self


class FieldStats(_Base):
    """Per-dim 1/99 percentiles + mean/std (Master Spec §L7).

    Both are shipped, not one: the 1/99 → [-1,1] default is π0.5/EgoVerse, but customers
    on 2/98-per-timestep (TRI LBM) or z-score (EgoMimic) must be able to re-derive without
    recomputing over the whole dataset. TRI LBM's finding is that normalization dominates,
    so getting this wrong is not cosmetic.
    """

    p01: tuple[float, ...]
    p99: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    # v4: the full raw-percentile set (1,2,5,25,50,75,95,98,99) so customers on ANY
    # normalization convention re-derive without touching the raw dataset. Optional —
    # v3 data carries only p01/p99, and explicit typed fields (not a loose dict) so a
    # missing percentile is visible in the type, not discovered at training time.
    p02: tuple[float, ...] | None = None
    p05: tuple[float, ...] | None = None
    p25: tuple[float, ...] | None = None
    p50: tuple[float, ...] | None = None
    p75: tuple[float, ...] | None = None
    p95: tuple[float, ...] | None = None
    p98: tuple[float, ...] | None = None


class NormStats(_Base):
    state: FieldStats | None = None
    action: FieldStats | None = None


class Diversity(_Base):
    """Scene and demonstrator reported SEPARATELY (EgoVerse) — collapsing them into one
    'diversity' number hides which axis is actually thin."""

    scene_id: str | None = None
    demonstrator_id: str | None = None


# --------------------------------------------------------------------------------------
# The episode
# --------------------------------------------------------------------------------------


class CanonicalEpisode(_Base):
    schema_version: int = SCHEMA_VERSION

    episode_id: str
    #: The physical recording this episode came from. CONSENT IS KEYED ON THIS, not on
    #: episode_id (AWS Architecture §3) — one capture yields many episodes, and revoking
    #: consent must revoke all of them at once. The local v1 data violates this: the same
    #: recording exists under four session ids with conflicting consent records.
    #:
    #: As of schema_version 2 this is CONTENT-ADDRESSED: it is the SHA-256 of the raw
    #: capture bytes (actuate.ingest.content_address). Identity is derived from the data,
    #: not assigned to it.
    capture_id: str

    #: The hash of the raw bytes this episode was derived from. Equal to `capture_id` by
    #: construction, and carried explicitly anyway: it makes the provenance chain
    #: *verifiable* rather than merely conventional. Anything holding this episode can
    #: re-hash the raw capture and prove the two belong together — which is precisely what
    #: nothing could do in Increment 1, when a 1.49 MB video sat beside metadata describing
    #: a 181 MB one.
    #:
    #: Optional only so that pre-content-addressing episodes remain loadable. New ingestion
    #: must populate it.
    source_content_hash: str | None = Field(default=None, min_length=64, max_length=64)

    rig: RigType

    frames: tuple[CanonicalFrame, ...] = ()

    # --- language (L6) ---
    #: str, imperative. May be None: task classification is genuinely unvalidated on real
    #: data, and v1's language grounding emits fluent templates like "Perform unknown task
    #: using right hand" that contain no task. Inventing one to satisfy a required field
    #: would launder a failed classification into a training label. The EXPORTER
    #: fail-closes on None (§L7) — the requirement is enforced at the delivery gate, where
    #: it is visible, not papered over upstream.
    task: str | None = None
    task_paraphrases: tuple[str, ...] = ()
    subtasks: tuple[Subtask, ...] = ()
    subgoal_frames: tuple[SubgoalFrame, ...] = ()
    #: v5: fine-grained atomic actions (closed 20-verb vocab, per-actor). Empty until
    #: language.label_actions runs. Overlapping spans across actors are expected.
    action_intervals: tuple[ActionInterval, ...] = ()

    # --- actions (L5) ---
    action_human: HumanAction | None = None
    action_robot: dict[str, RobotAction] = Field(default_factory=dict)
    control_mode: ControlMode | None = None
    finger_action_repr: FingerActionRepr | None = None

    # --- certification (L4) ---
    episode_meta: EpisodeMeta = Field(default_factory=EpisodeMeta)
    strategy_alignment: dict[str, StrategyAlignment] = Field(default_factory=dict)
    #: v4: per-embodiment verdict from L5 sim validation (MuJoCo replay: joint limits,
    #: self-collision, IK convergence). Keyed by embodiment name. Absent = never validated,
    #: which is NOT the same claim as False (validated and failed).
    retarget_eligibility: dict[str, bool] = Field(default_factory=dict)

    #: THE HARD GATE. Fail-closed: only GRANTED + PASSED may be packaged for delivery.
    #: There is no default that opens the gate.
    consent: ConsentStatus = ConsentStatus.PENDING
    pii_status: PiiStatus = PiiStatus.PENDING

    # --- delivery (L7) ---
    norm_stats: NormStats | None = None
    diversity: Diversity = Field(default_factory=Diversity)
    tier: Tier | None = None
    effective_hours: float | None = Field(default=None, ge=0.0)

    #: Free-form record of values that were INFERRED rather than captured. The legacy
    #: adapter uses it to say so out loud. An empty dict means everything here came from
    #: the capture record itself.
    derivation_notes: dict[str, str] = Field(default_factory=dict)

    # ---------------- validators: the honesty constraints ----------------

    @model_validator(mode="after")
    def _rig_must_measure_what_it_reports(self) -> CanonicalEpisode:
        """A rig may not report a channel it has no sensor for.

        Without this, a vision-only pipeline could emit confident per-finger contact
        vectors and robot-space finger joints that no sensor ever produced — exactly what
        sensor-truth priority exists to prevent, and exactly what the dexterous branch
        (§L2, §L5) assumes cannot happen.
        """
        spec = get_rig(self.rig)
        for frame in self.frames:
            if frame.contact is not None and not spec.measures(Channel.CONTACT):
                raise ValueError(
                    f"episode {self.episode_id}: rig {self.rig.value} does not measure contact, "
                    f"but frame {frame.frame_idx} reports contact.<finger>. Use None — meaning "
                    "'not measured' — never a fabricated reading."
                )
            if frame.finger_joints_robotspace is not None and not spec.measures(
                Channel.FINGER_POSE_ROBOTSPACE
            ):
                raise ValueError(
                    f"episode {self.episode_id}: rig {self.rig.value} has no robot-space "
                    f"encoders, but frame {frame.frame_idx} reports finger_joints_robotspace. "
                    "Only a DexUMI exoskeleton produces this; it is measured ground truth "
                    "for the retarget target and cannot be inferred."
                )
        return self

    @model_validator(mode="after")
    def _measured_provenance_requires_a_sensor(self) -> CanonicalEpisode:
        """A rig may not claim a HARDWARE MEASUREMENT it has no hardware for.

        A head-mounted bare-hand rig stamping `measured_human` on a grasp signal would
        promote a vision heuristic to sensor ground truth, and the L2 arbiter — which
        resolves conflicts strictly by trust rank — would then let it override a real
        sensor on a fused rig.

        Only the MEASURED tiers are constrained. `vision_primary` is available to every
        rig: it means "a metric vision model was the best source available", which is an
        honest and often correct statement, not a claim about hardware.
        """
        from actuate.config import MEASURED_PROVENANCES, trust_rank

        spec = get_rig(self.rig)
        ceiling = spec.measured_provenance_ceiling

        for frame in self.frames:
            for field_name, prov in frame.provenance.items():
                if prov not in MEASURED_PROVENANCES:
                    continue
                if ceiling is None:
                    raise ValueError(
                        f"episode {self.episode_id}: frame {frame.frame_idx} claims "
                        f"provenance={prov.value} for {field_name!r}, but rig "
                        f"{self.rig.value} has no measuring hardware at all. Every value "
                        "it produces is inferred."
                    )
                if trust_rank(prov) < trust_rank(ceiling):
                    raise ValueError(
                        f"episode {self.episode_id}: frame {frame.frame_idx} claims "
                        f"provenance={prov.value} for {field_name!r}, but rig "
                        f"{self.rig.value} can measure no better than {ceiling.value}. "
                        "A rig cannot report a sensor it does not have."
                    )
        return self

    @model_validator(mode="after")
    def _content_hash_agrees_with_capture_id(self) -> CanonicalEpisode:
        """If both are present they must be the same value.

        The capture id IS the content hash (schema v2). Letting them diverge would
        reintroduce exactly the ambiguity content-addressing exists to remove — an episode
        that claims to come from one capture while its bytes say another.
        """
        if (
            self.source_content_hash is not None
            and self.source_content_hash != self.capture_id
        ):
            raise ValueError(
                f"episode {self.episode_id}: source_content_hash "
                f"{self.source_content_hash[:12]}... disagrees with capture_id "
                f"{self.capture_id[:12]}.... The capture id is the content hash; they "
                "cannot differ."
            )
        return self

    @model_validator(mode="after")
    def _frames_belong_to_this_episode(self) -> CanonicalEpisode:
        wrong = [f.frame_idx for f in self.frames if f.episode_id != self.episode_id][:3]
        if wrong:
            raise ValueError(
                f"episode {self.episode_id}: frame(s) {wrong} carry a different episode_id. "
                "The per-frame records do not belong to this episode."
            )
        wrong_rig = [f.frame_idx for f in self.frames if f.rig is not self.rig][:3]
        if wrong_rig:
            raise ValueError(
                f"episode {self.episode_id}: frame(s) {wrong_rig} declare a different rig."
            )
        return self

    @model_validator(mode="after")
    def _robot_action_names_a_real_embodiment(self) -> CanonicalEpisode:
        for key, action in self.action_robot.items():
            if action.embodiment != key:
                raise ValueError(
                    f"action_robot[{key!r}] carries embodiment={action.embodiment!r} — "
                    "the key and the action disagree about which robot this is."
                )
        return self

    # ---------------- the delivery gate ----------------

    @property
    def is_deliverable(self) -> bool:
        """Fail-closed. Both gates must clear; neither has a default that opens.

        This is the property the packaging path and the S3 write guard read. It is
        deliberately a positive assertion of two specific values, not `!= denied` —
        a new ConsentStatus member added later must not silently become deliverable.
        """
        return (
            self.consent is ConsentStatus.GRANTED and self.pii_status is PiiStatus.PASSED
        )

    def delivery_block_reason(self) -> str | None:
        """Why this episode may not ship. None means it may."""
        if self.consent is not ConsentStatus.GRANTED:
            return f"consent={self.consent.value} (required: granted)"
        if self.pii_status is not PiiStatus.PASSED:
            return f"pii_status={self.pii_status.value} (required: passed)"
        return None
