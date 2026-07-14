"""The canonical schema (Master Spec §3) — mostly about what it REFUSES to hold.

A schema that accepts anything is not a contract. Each rejection below is a specific way
this pipeline could otherwise ship a false claim, and every one of them corresponds to a
sentence in the spec about sensor-truth priority.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from actuate.config import (
    Channel,
    ConsentStatus,
    ControlMode,
    Finger,
    FingerActionRepr,
    InteractionState,
    PiiStatus,
    Provenance,
    RigType,
    Side,
    get_rig,
    trust_rank,
)
from actuate.schema import (
    SE3,
    CanonicalEpisode,
    CanonicalFrame,
    ContactReading,
    HandState,
    RobotAction,
    StrategyAlignment,
)

IDENT = (1.0, 0.0, 0.0, 0.0)


def _frame(rig: RigType = RigType.HEAD_MOUNTED, **over) -> CanonicalFrame:
    base = dict(t=0.0, rig=rig, episode_id="e", frame_idx=0)
    base.update(over)
    return CanonicalFrame(**base)


def _episode(rig: RigType = RigType.HEAD_MOUNTED, **over) -> CanonicalEpisode:
    base = dict(episode_id="e", capture_id="c", rig=rig)
    base.update(over)
    return CanonicalEpisode(**base)


def _contact(prov: Provenance) -> dict:
    return {Side.RIGHT: {f: ContactReading(confidence=0.5, source=prov) for f in Finger}}


# --- provenance is mandatory ------------------------------------------------------------


def test_populated_field_without_provenance_is_rejected():
    """A value with no stated origin is invisible to every downstream trust computation."""
    with pytest.raises(ValidationError, match="carry no provenance"):
        _frame(interaction_state=InteractionState.STATIC)


def test_empty_frame_needs_no_provenance():
    _frame()  # nothing populated, nothing to justify


# --- a rig may not report what it cannot sense -------------------------------------------


def test_bare_hand_rig_cannot_report_finger_contact():
    """Zeros here would assert 'we measured, nothing is touching' — a claim no vision-only
    rig can make. The dexterous branch must be able to tell absent from open."""
    f = _frame(contact=_contact(Provenance.VISION_FALLBACK),
               provenance={"contact": Provenance.VISION_FALLBACK})
    with pytest.raises(ValidationError, match="does not measure contact"):
        _episode(frames=(f,))


def test_glove_rig_may_report_measured_finger_contact():
    f = _frame(rig=RigType.GLOVE, contact=_contact(Provenance.MEASURED_HUMAN),
               provenance={"contact": Provenance.MEASURED_HUMAN})
    _episode(rig=RigType.GLOVE, frames=(f,))


def test_only_dexumi_may_report_robotspace_finger_joints():
    """finger_joints_robotspace is MEASURED ground truth for the retarget target itself.
    It cannot be inferred — which is why DexUMI rigs bypass finger retargeting."""
    f = _frame(rig=RigType.GLOVE, finger_joints_robotspace={Side.RIGHT: (0.1, 0.2)},
               provenance={"finger_joints_robotspace": Provenance.MEASURED_HUMAN})
    with pytest.raises(ValidationError, match="no robot-space encoders"):
        _episode(rig=RigType.GLOVE, frames=(f,))

    ok = _frame(rig=RigType.DEXUMI_EXOSKELETON,
                finger_joints_robotspace={Side.RIGHT: (0.1, 0.2)},
                provenance={"finger_joints_robotspace": Provenance.MEASURED_ROBOTSPACE})
    _episode(rig=RigType.DEXUMI_EXOSKELETON, frames=(ok,))


def test_a_rig_with_no_sensors_cannot_claim_a_measured_provenance():
    """A head-mounted rig stamping `measured_robotspace` would promote a vision heuristic
    to sensor ground truth — and the L2 arbiter, which resolves conflicts strictly by
    trust rank, would then let it override a real sensor on a fused rig."""
    f = _frame(interaction_state=InteractionState.STATIC,
               provenance={"interaction_state": Provenance.MEASURED_ROBOTSPACE})
    with pytest.raises(ValidationError, match="has no measuring hardware at all"):
        _episode(frames=(f,))


def test_a_rig_cannot_claim_a_measured_provenance_above_its_ceiling():
    """A UMI gripper has an aperture encoder — but not a glove. It may claim
    `gripper_aperture`, never `measured_human`."""
    f = _frame(rig=RigType.UMI_GRIPPER, interaction_state=InteractionState.STATIC,
               provenance={"interaction_state": Provenance.GRIPPER_APERTURE})
    _episode(rig=RigType.UMI_GRIPPER, frames=(f,))  # fine

    bad = _frame(rig=RigType.UMI_GRIPPER, interaction_state=InteractionState.STATIC,
                 provenance={"interaction_state": Provenance.MEASURED_HUMAN})
    with pytest.raises(ValidationError, match="can measure no better than"):
        _episode(rig=RigType.UMI_GRIPPER, frames=(bad,))


def test_any_rig_may_claim_vision_primary():
    """Regression: the ceiling constrains only the MEASURED tiers.

    A bare-hand head-mounted rig running a metric hand model (WiLoR) genuinely produces
    `vision_primary` hand pose — that is an honest statement that a metric vision model was
    the best available source, not a hardware claim. An earlier version of this validator
    forced such output down to `vision_fallback` (a heuristic), which is its own kind of
    lie. Caught by the real-data round-trip test on session_001.
    """
    f = _frame(hands={Side.RIGHT: HandState(keypoints_3d=((0.0, 0.0, 1.0),) * 21)},
               provenance={"hands": Provenance.VISION_PRIMARY})
    _episode(frames=(f,))


def test_partial_contact_vector_is_rejected():
    """A contact dict with 1 of 5 fingers reads as 'the other four aren't touching'."""
    with pytest.raises(ValidationError, match="Report all five fingers or none"):
        _frame(
            rig=RigType.GLOVE,
            contact={Side.RIGHT: {Finger.THUMB: ContactReading(
                confidence=0.5, source=Provenance.MEASURED_HUMAN)}},
            provenance={"contact": Provenance.MEASURED_HUMAN},
        )


# --- the trust ordering (Master Spec §L2) ------------------------------------------------


def test_trust_ordering_matches_the_spec():
    order = [
        Provenance.MEASURED_ROBOTSPACE,
        Provenance.MEASURED_HUMAN,
        Provenance.GRIPPER_APERTURE,
        Provenance.VISION_PRIMARY,
        Provenance.VISION_FALLBACK,
    ]
    assert [trust_rank(p) for p in order] == sorted(trust_rank(p) for p in order)


def test_the_six_rigs_and_what_they_measure():
    assert len(RigType) == 6
    assert get_rig(RigType.HEAD_MOUNTED).measured == frozenset(), (
        "a bare-hand head-mounted rig measures NOTHING — every value is inferred"
    )
    assert get_rig(RigType.DEXUMI_EXOSKELETON).measures(Channel.FINGER_POSE_ROBOTSPACE)
    assert get_rig(RigType.UMI_GRIPPER).measures(Channel.GRASP)


# --- geometry -----------------------------------------------------------------------------


def test_non_unit_quaternion_is_rejected():
    with pytest.raises(ValidationError, match="unit-norm"):
        SE3(position_m=(0, 0, 0), quaternion_wxyz=(1, 1, 1, 1))


def test_keypoints_must_be_21():
    with pytest.raises(ValidationError, match="must be 21x3"):
        HandState(keypoints_3d=((0.0, 0.0, 0.0),) * 20)


# --- episode integrity ---------------------------------------------------------------------


def test_frames_must_belong_to_their_episode():
    other = CanonicalFrame(t=0, rig=RigType.HEAD_MOUNTED, episode_id="OTHER", frame_idx=0)
    with pytest.raises(ValidationError, match="do not belong to this episode"):
        _episode(frames=(other,))


def test_robot_action_without_a_trajectory_is_rejected():
    with pytest.raises(ValidationError, match="contains no motion"):
        RobotAction(embodiment="franka_dual", control_mode=ControlMode.JOINT)


def test_strategy_alignment_flag_must_say_what_mismatched():
    with pytest.raises(ValidationError, match="A flag no human can act on"):
        StrategyAlignment(ok=False)
    StrategyAlignment(ok=False, reason="retarget forces a top grasp; demo used a side grasp")


# --- the delivery gate ---------------------------------------------------------------------


def test_delivery_gate_requires_both_and_defaults_closed():
    assert not _episode().is_deliverable
    assert not _episode(consent=ConsentStatus.GRANTED).is_deliverable
    assert not _episode(pii_status=PiiStatus.PASSED).is_deliverable
    assert _episode(
        consent=ConsentStatus.GRANTED, pii_status=PiiStatus.PASSED
    ).is_deliverable


# --- decisions the spec bakes in -------------------------------------------------------------


def test_both_control_modes_and_both_finger_reprs_exist():
    """Master Spec §3: both are CARRIED, never conflated (DexUMI robustness finding)."""
    assert {m.value for m in ControlMode} == {"joint", "ee"}
    assert {r.value for r in FingerActionRepr} == {"absolute", "relative"}


def test_canonical_reference_hand_is_deliberately_unpicked():
    """Master Spec §7 item 4 makes choosing the ~20+ DoF reference hand a
    benchmark-before-lock decision. It is NOT picked, and the code says so rather than
    hardcoding a guess that the Stage-I exporter would bake into every dataset."""
    from actuate.config import CANONICAL_REFERENCE_HAND

    assert CANONICAL_REFERENCE_HAND is None


def test_no_embodiment_is_retarget_ready():
    """L5 does not exist. No robot has a URDF registered, so nothing is retarget-eligible
    — and the registry says so rather than implying otherwise."""
    from actuate.config import retarget_ready_embodiments

    assert retarget_ready_embodiments() == []
