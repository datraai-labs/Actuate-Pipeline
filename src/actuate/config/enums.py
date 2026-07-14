"""Closed vocabularies shared by the config registries and the canonical schema.

These live in `config/` rather than `schema/` because the rig registry needs them and
the import-linter contract puts `config` at the bottom of the layer stack — `schema`
imports from here, never the reverse.

Every enum is a contract. Adding a member is a schema change and needs a SCHEMA_VERSION
bump; downstream exporters and the certification layer branch on these values.
"""

from __future__ import annotations

from enum import Enum


class RigType(str, Enum):
    """The six capture configurations (Master Spec §1, §L0)."""

    HEAD_MOUNTED = "head_mounted"
    UMI_GRIPPER = "umi_gripper"
    STEREO = "stereo"
    GLOVE = "glove"
    TELEOP_ROBOT = "teleop_robot"
    DEXUMI_EXOSKELETON = "dexumi_exoskeleton"  # 6th rig, new in Master Spec


class Channel(str, Enum):
    """A quantity that a rig may either measure with hardware or estimate from vision."""

    GRASP = "grasp"
    DEPTH = "depth"
    FINGER_POSE = "finger_pose"
    #: Distinct from FINGER_POSE: joint angles already expressed in the ROBOT hand's
    #: space (DexUMI exoskeleton). This is measured ground truth for the retarget
    #: target itself, which is why such rigs skip finger retargeting altogether.
    FINGER_POSE_ROBOTSPACE = "finger_pose_robotspace"
    CONTACT = "contact"
    JOINT_STATE = "joint_state"


class Provenance(str, Enum):
    """How a value was obtained. The trust ordering of Master Spec §L2.

    The L2 arbiter resolves competing signals strictly by this order — a measured
    channel always overrides vision when present. Every downstream confidence
    computation and the L4 certificate key on this field.

    Ordering, most trusted first:
        measured_robotspace  (DexUMI exoskeleton encoders — already in robot hand space)
        measured_human       (instrumented glove flex/joint sensors)
        gripper_aperture     (UMI aperture encoder)
        vision_primary       (a metric vision model is the best source available)
        vision_fallback      (heuristic; nothing better exists for this rig)
        approximated         (no measurement at all — e.g. an assumed-FOV pinhole model)
    """

    MEASURED_ROBOTSPACE = "measured_robotspace"
    MEASURED_HUMAN = "measured_human"
    GRIPPER_APERTURE = "gripper_aperture"
    VISION_PRIMARY = "vision_primary"
    VISION_FALLBACK = "vision_fallback"
    APPROXIMATED = "approximated"


#: Trust ordering as data, most-trusted first. Exposed so the arbiter reads a list
#: rather than scattering if/elses that can drift out of agreement with each other.
PROVENANCE_TRUST_ORDER: tuple[Provenance, ...] = (
    Provenance.MEASURED_ROBOTSPACE,
    Provenance.MEASURED_HUMAN,
    Provenance.GRIPPER_APERTURE,
    Provenance.VISION_PRIMARY,
    Provenance.VISION_FALLBACK,
    Provenance.APPROXIMATED,
)

#: Provenances that represent a real hardware measurement rather than an inference.
MEASURED_PROVENANCES: frozenset[Provenance] = frozenset(
    {
        Provenance.MEASURED_ROBOTSPACE,
        Provenance.MEASURED_HUMAN,
        Provenance.GRIPPER_APERTURE,
    }
)


def trust_rank(p: Provenance) -> int:
    """Lower is more trusted. The arbiter picks the minimum."""
    return PROVENANCE_TRUST_ORDER.index(p)


class EgoMotionMethod(str, Enum):
    NONE = "none"
    ORB_SLAM3 = "orb_slam3"
    ARIA_MPS = "aria_mps"


class InteractionState(str, Enum):
    """Master Spec §3 per-frame `interaction_state`."""

    STATIC = "STATIC"
    GRASPED_L = "GRASPED_L"
    GRASPED_R = "GRASPED_R"
    GRASPED_BOTH = "GRASPED_BOTH"
    MOVING = "MOVING"

    @property
    def is_grasped(self) -> bool:
        return self in (
            InteractionState.GRASPED_L,
            InteractionState.GRASPED_R,
            InteractionState.GRASPED_BOTH,
        )


class ControlMode(str, Enum):
    """Master Spec §3: both modes are carried, tagged, never conflated."""

    JOINT = "joint"
    EE = "ee"


class FingerActionRepr(str, Enum):
    """Master Spec §3: both representations available (DexUMI robustness finding)."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class Tier(str, Enum):
    """Delivery tiering (Master Spec §L7)."""

    STAGE1_VOLUME = "stage1_volume"
    STAGE2_ANCHOR = "stage2_anchor"


class ConsentStatus(str, Enum):
    """Fail-closed. Only GRANTED permits delivery packaging.

    There is deliberately no default. A missing consent record is not PENDING and is
    not GRANTED — it is an error, and the guard blocks. See actuate.io.consent.
    """

    GRANTED = "granted"
    PENDING = "pending"
    DENIED = "denied"
    REVOKED = "revoked"


class PiiStatus(str, Enum):
    """Fail-closed. Only PASSED permits delivery packaging (Master Spec §L4)."""

    PASSED = "passed"
    PENDING = "pending"
    FAILED = "failed"


class Side(str, Enum):
    LEFT = "L"
    RIGHT = "R"


class Finger(str, Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"
