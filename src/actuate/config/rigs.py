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


class SensorStreamSpec(BaseModel):
    """One non-video sensor stream a rig produces, as it appears on disk.

    `patterns` are glob patterns relative to the session directory. They are the
    contract ingest verifies against: a required stream with no match fails ingest
    with an explicit absence reason; a stream whose patterns match MORE than one
    file fails as ambiguous rather than letting ingest silently pick one; and a
    file matching any rig's sensor patterns that no declared stream claims fails
    as an undeclared sensor. All three failure modes are the same bug class — a
    session ingesting cleanly while a sensor is discarded or missing.

    For rigs with no real capture session yet, the patterns are provisional naming
    conventions — lock them against the first real capture of that rig.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    patterns: tuple[str, ...]
    required: bool = True


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

    #: Non-video sensor sidecars this rig produces on disk. Ingest asserts every
    #: required stream is present and that nothing sensor-shaped goes unclaimed —
    #: see SensorStreamSpec. Empty means the rig ships video only.
    sensor_streams: tuple[SensorStreamSpec, ...] = ()

    def measures(self, channel: Channel) -> bool:
        return channel in self.measured


#: IMU sidecar patterns shared by every IMU-bearing rig. One name so the
#: convention can't fork per rig.
_IMU_PATTERNS = ("imu*.json", "imu*.csv")

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
        # required=False: the real corpus has valid head-mounted sessions with no
        # IMU sidecar, and SLAM honestly falls back to vision-only rotation.
        sensor_streams=(
            SensorStreamSpec(name="imu_head", patterns=_IMU_PATTERNS, required=False),
        ),
    ),
    RigType.UMI_GRIPPER: RigSpec(
        rig=RigType.UMI_GRIPPER,
        description="UMI handheld gripper: wrist RGB (fisheye) + aperture encoder + wrist IMU.",
        measured=frozenset({Channel.GRASP}),  # real aperture encoder
        cameras=("wrist",),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="unidepth_v2",
        measured_provenance_ceiling=Provenance.GRIPPER_APERTURE,
        # The aperture encoder is the rig's whole reason to exist (its GRASP channel
        # is hardware-measured) — a UMI session without it is not a UMI session.
        sensor_streams=(
            SensorStreamSpec(name="imu_wrist", patterns=_IMU_PATTERNS),
            SensorStreamSpec(
                name="aperture",
                patterns=("aperture*.json", "aperture*.csv", "gripper_aperture*.csv"),
            ),
        ),
    ),
    RigType.STEREO: RigSpec(
        rig=RigType.STEREO,
        description="Calibrated stereo pair with a real metric depth stream.",
        measured=frozenset({Channel.DEPTH}),
        cameras=("stereo_left", "stereo_right"),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="foundation_stereo",
        measured_provenance_ceiling=None,  # a depth sensor, but no grasp/contact hardware
        sensor_streams=(
            SensorStreamSpec(name="imu", patterns=_IMU_PATTERNS, required=False),
            # depth may legitimately be computed from the pair rather than shipped.
            SensorStreamSpec(name="depth", patterns=("depth.raw",), required=False),
        ),
    ),
    RigType.GLOVE: RigSpec(
        rig=RigType.GLOVE,
        description="Instrumented glove: per-finger flex/joint sensors + IMU + optional tactile.",
        measured=frozenset({Channel.FINGER_POSE, Channel.GRASP, Channel.CONTACT}),
        cameras=("head",),
        ego_motion=EgoMotionMethod.ORB_SLAM3,
        depth_model="unidepth_v2",
        measured_provenance_ceiling=Provenance.MEASURED_HUMAN,
        # The joint stream is what makes this rig measure FINGER_POSE; without it
        # the session is a bare-hand head-mounted capture wearing the wrong label.
        sensor_streams=(
            SensorStreamSpec(name="glove_joints", patterns=("glove*.json", "glove*.csv")),
            SensorStreamSpec(name="imu_glove", patterns=_IMU_PATTERNS),
            SensorStreamSpec(
                name="tactile",
                patterns=("tactile*.json", "tactile*.csv"),
                required=False,
            ),
        ),
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
        sensor_streams=(
            SensorStreamSpec(
                name="joint_states",
                patterns=("joint_states*.json", "joint_states*.csv"),
            ),
            SensorStreamSpec(
                name="gripper_state",
                patterns=("gripper_state*.json", "gripper_state*.csv"),
                required=False,
            ),
            SensorStreamSpec(
                name="ee_wrench",
                patterns=("ee_wrench*.json", "ee_wrench*.csv", "wrench*.csv"),
                required=False,
            ),
        ),
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
        # The robot-space encoder stream is why this rig tops the trust ordering
        # (measured GT for the retarget target itself) — it is not optional.
        sensor_streams=(
            SensorStreamSpec(
                name="exo_joints_robotspace",
                patterns=("exo_joints*.json", "exo_joints*.csv", "joints_robotspace*.json",
                          "joints_robotspace*.csv"),
            ),
            SensorStreamSpec(
                name="tactile",
                patterns=("tactile*.json", "tactile*.csv"),
                required=False,
            ),
            SensorStreamSpec(name="imu_wrist", patterns=_IMU_PATTERNS, required=False),
        ),
    ),
}


def get_rig(rig: RigType) -> RigSpec:
    return RIG_REGISTRY[rig]


def all_sensor_patterns() -> frozenset[str]:
    """Every sensor sidecar pattern any registered rig declares.

    This is the recognizer used to catch UNDECLARED sensors at ingest: a file
    matching any of these that no declared stream of the session's rig claims
    means a sensor is about to be silently discarded. Deriving it from the
    registry keeps the guarantee self-maintaining — registering a new rig's
    patterns automatically protects them on every other rig too.
    """
    return frozenset(
        pattern
        for spec in RIG_REGISTRY.values()
        for stream in spec.sensor_streams
        for pattern in stream.patterns
    )
