"""Rig registry — Master Spec §2.3.

One of the two extension points of the whole system (the other is the embodiment
registry). A rig entry declares, for one capture configuration, **which channels are
hardware-measured and which are estimated from vision**.

This is not documentation. It is the fact the schema validates against: a rig that does
not measure finger pose may not emit `finger_joints_human`, and a rig with no tactile
sensor may not emit `contact.<finger>`. Without a registry those constraints would be
convention, and convention is what lets a vision-only pipeline quietly ship confident
per-finger contact vectors no sensor ever produced.

Adding a rig means adding an entry here — not touching any layer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from actuate.config.enums import Channel, EgoMotionMethod, Provenance, RigType


class RigSpec(BaseModel):
    """What one capture rig natively produces, and how much to trust it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rig: RigType
    description: str

    #: Channels this rig HARDWARE-MEASURES. Anything not listed is estimated.
    #: An empty set is a legitimate, common value — a bare-hand head-mounted rig
    #: measures none of them — and must not be read as "unknown".
    measured: frozenset[Channel] = Field(default_factory=frozenset)

    #: Camera names this rig emits. These become `observation.images.<name>` in
    #: LeRobot/RLDS, so they are enforced consistent from ingestion onward (Master
    #: Spec §3): a camera called "wrist" in one session and "wrist_cam" in the next
    #: silently splits a training dataset in two.
    cameras: tuple[str, ...] = ()

    ego_motion: EgoMotionMethod = EgoMotionMethod.NONE

    #: Default depth model. `auto` in the CLI resolves through here.
    #: Master Spec §7 item 1: these are BENCHMARK-BEFORE-LOCK, not settled.
    depth_model: str = "unidepth_v2"

    #: The best HARDWARE-MEASURED provenance this rig can claim.
    #:
    #: `None` means the rig measures nothing, so it may claim no measured provenance at
    #: all — every value it produces is an inference.
    #:
    #: Note this ceiling constrains only the MEASURED tiers (measured_robotspace,
    #: measured_human, gripper_aperture). The vision tiers are available to every rig:
    #: `vision_primary` is not a weaker claim a rig has to earn, it is an honest statement
    #: that a metric vision model was the best source available — which is exactly what a
    #: bare-hand head-mounted rig has for hand pose. Conflating the two would force real
    #: vision-model output to be labelled `vision_fallback` (a heuristic), which is its own
    #: kind of lie.
    measured_provenance_ceiling: Provenance | None = None

    def measures(self, channel: Channel) -> bool:
        return channel in self.measured


RIG_REGISTRY: dict[RigType, RigSpec] = {
    RigType.HEAD_MOUNTED: RigSpec(
        rig=RigType.HEAD_MOUNTED,
        description="Egocentric head-mounted RGB + head IMU. Bare hands.",
        measured=frozenset(),  # nothing. Every quantity is inferred from vision.
        cameras=("head",),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="unidepth_v2",
        # measures nothing -> may claim NO measured provenance.
        measured_provenance_ceiling=None,
    ),
    RigType.UMI_GRIPPER: RigSpec(
        rig=RigType.UMI_GRIPPER,
        description="UMI handheld gripper: wrist RGB (fisheye) + aperture encoder + wrist IMU.",
        measured=frozenset({Channel.GRASP}),  # real aperture encoder
        cameras=("wrist",),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="unidepth_v2",
        measured_provenance_ceiling=Provenance.GRIPPER_APERTURE,
    ),
    RigType.STEREO: RigSpec(
        rig=RigType.STEREO,
        description="Calibrated stereo pair with a real metric depth stream.",
        measured=frozenset({Channel.DEPTH}),
        cameras=("stereo_left", "stereo_right"),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="foundation_stereo",
        measured_provenance_ceiling=None,  # a depth sensor, but no grasp/contact hardware
    ),
    RigType.GLOVE: RigSpec(
        rig=RigType.GLOVE,
        description="Instrumented glove: per-finger flex/joint sensors + IMU + optional tactile.",
        measured=frozenset({Channel.FINGER_POSE, Channel.GRASP, Channel.CONTACT}),
        cameras=("head",),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="unidepth_v2",
        measured_provenance_ceiling=Provenance.MEASURED_HUMAN,
    ),
    RigType.TELEOP_ROBOT: RigSpec(
        rig=RigType.TELEOP_ROBOT,
        description="Teleoperated robot: native joint encoders, gripper state, EE force.",
        measured=frozenset(
            {Channel.JOINT_STATE, Channel.GRASP, Channel.FINGER_POSE, Channel.CONTACT}
        ),
        cameras=("top", "wrist"),
        ego_motion=EgoMotionMethod.NONE,  # fixed cameras; robot kinematics give the frame
        depth_model="foundation_stereo",
        measured_provenance_ceiling=Provenance.MEASURED_ROBOTSPACE,
    ),
    # The 6th rig, new in the Master Spec (§L0). Its exoskeleton encoders report joint
    # angles ALREADY IN ROBOT HAND SPACE — which is why L5 lets it bypass finger
    # retargeting entirely (§L5 "DexUMI bypass"), and why it sits at the top of the
    # trust ordering. This rig is the reason `finger_joints_robotspace` exists as a
    # field distinct from `finger_joints_human`.
    RigType.DEXUMI_EXOSKELETON: RigSpec(
        rig=RigType.DEXUMI_EXOSKELETON,
        description="DexUMI exoskeleton: robot-space encoder joints + tactile. Human hand as UMI.",
        measured=frozenset(
            {Channel.FINGER_POSE, Channel.FINGER_POSE_ROBOTSPACE, Channel.CONTACT, Channel.GRASP}
        ),
        cameras=("wrist",),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="unidepth_v2",
        measured_provenance_ceiling=Provenance.MEASURED_ROBOTSPACE,
    ),
}


def get_rig(rig: RigType) -> RigSpec:
    return RIG_REGISTRY[rig]
