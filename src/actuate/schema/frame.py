"""Per-frame canonical record — Master Spec §3, exactly.

    t                        float64, canonical clock          L0
    rig, episode_id, frame_idx                                 L0
    images.<cam>             ref to chunked MP4                L0
    camera_pose              SE(3), world frame                L1 SLAM/MPS
    hand.<L|R>.mano          MANO params (beta fixed, theta 45 axis-angle)  L1 WiLoR
    hand.<L|R>.keypoints_3d  21x3, camera frame                L1
    hand.<L|R>.wrist_pose    SE(3), camera frame               L1
    finger_joints_human      per-finger joint angles           L2 glove / L1 vision
    finger_joints_robotspace joint angles in ROBOT hand space  L2 DexUMI (measured GT)
    object.<id>.pose         SE(3)                             L1 FoundationPose
    object.<id>.mask         RLE/ref                           L1 SAM2
    depth.<cam>              ref to depth map + uncertainty    L1
    contact.<finger>         confidence [0,1] + source enum    L2
    interaction_state        enum                              L2
    confidence.<field>       float                             all
    provenance.<field>       source tag                        all

Two invariants are enforced by validators here rather than left to convention, because
both are places where a plausible-looking default silently becomes a false claim:

  1. **Every populated field carries a provenance entry.** The whole trust model keys on
     provenance (Master Spec §5, §L4). A value that skipped it is invisible to every
     downstream confidence computation, so the schema refuses to hold one.

  2. **A field may only be populated if the rig actually measures it.** `contact.thumb`
     on a bare-hand head-mounted rig is not a low-confidence reading — it is a reading
     that does not exist. `None` means *not measured*; `0.0` means *measured, nothing
     touching*. The dexterous branch (§L5) treats reliable contact state as a hard
     prerequisite, so a fabricated one is strictly worse than an absent one. This check
     lives on CanonicalEpisode, which is the level that knows the rig.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from actuate.config import Finger, InteractionState, Provenance, RigType, Side

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


class SE3(_Base):
    """Rigid pose. Right-handed, gravity-aligned. Position in METRES.

    The schema has no place for pixel or normalized coordinates — anything entering it
    is already metric, which is the entire point of L1's metric reconstruction.
    """

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _unit_quaternion(self) -> SE3:
        w, x, y, z = self.quaternion_wxyz
        n2 = w * w + x * x + y * y + z * z
        if not (0.99 < n2 < 1.01):
            raise ValueError(f"quaternion_wxyz must be unit-norm; got |q|^2={n2:.6f}")
        return self


# --------------------------------------------------------------------------------------
# Refs — the canonical representation POINTS AT pixels, it never carries them.
# AWS Architecture §1: video is the dominant storage cost; never duplicate or re-encode.
# --------------------------------------------------------------------------------------


class ImageRef(_Base):
    """-> observation.images.<cam>. A reference into a chunked MP4, not a frame."""

    uri: str  # s3://actuate-work-<env>/... or a local path under the same layout
    frame_index: int = Field(ge=0)


class DepthRef(_Base):
    """Depth map + its uncertainty. UniDepthV2 emits per-pixel uncertainty and that
    feeds the certificate directly, so a depth ref without one is a lost trust signal."""

    uri: str
    frame_index: int = Field(ge=0)
    uncertainty_uri: str | None = None
    mean_uncertainty: float | None = None


class MaskRef(_Base):
    """RLE or a pointer to one (SAM2)."""

    uri: str | None = None
    rle: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> MaskRef:
        if (self.uri is None) == (self.rle is None):
            raise ValueError("MaskRef needs exactly one of uri or rle")
        return self


# --------------------------------------------------------------------------------------
# Hand — MANO is the INTERMEDIATE (Master Spec §3 key decisions), not the delivered target
# --------------------------------------------------------------------------------------


class MANOParams(_Base):
    """beta fixed (shape), theta FULL 45 axis-angle (pose). From WiLoR (L1).

    Master Spec is explicit that MANO is the *intermediate*: the delivered Stage-I
    pretraining target is relative-SE(3) wrist + retargeted joints on a canonical
    reference hand. Carrying MANO here is what makes that retarget possible later; it is
    not itself the action.

    ### Why 45 axis-angle, not 15-PCA (schema_version 3)

    v2 stored `theta_pca` — MANO's first 15 PCA components. Measured on the real capture with
    a correct least-squares projection (the components are NOT orthonormal, so a transpose
    inverse is wrong — see perception.hands.to_theta_pca), projecting WiLoR's native 45
    axis-angle down to the top-15 subspace loses a **median 10.3 deg / p90 17.3 deg / p99
    22.5 deg** of per-joint angle. That is material for retargeting, where contact placement
    is the whole point (§L5): a p99 of 22 deg on a finger joint moves a fingertip centimetres,
    and carrying the full 45 is free and exactly lossless. So the schema carries all **45**
    axis-angle values (15
    joints x 3). A consumer that wants the compressed form can project 45 -> N itself
    (`perception.hands.to_theta_pca`); the schema refuses to throw away information it can't
    get back. This widening is why v2 -> v3 is a version bump, not a silent change.
    """

    betas: tuple[float, ...] = Field(min_length=10, max_length=10)
    #: 45 axis-angle values = 15 hand joints x 3. WiLoR's native output. NOT PCA.
    theta: tuple[float, ...] = Field(min_length=45, max_length=45)
    global_orient: tuple[float, float, float]


class HandState(_Base):
    mano: MANOParams | None = None
    #: 21x3 in the camera frame. The fingertip retarget input (GeoRT).
    keypoints_3d: tuple[tuple[float, float, float], ...] | None = None
    wrist_pose: SE3 | None = None

    @model_validator(mode="after")
    def _twentyone_keypoints(self) -> HandState:
        if self.keypoints_3d is not None and len(self.keypoints_3d) != 21:
            raise ValueError(
                f"keypoints_3d must be 21x3 (MANO/MediaPipe topology); got {len(self.keypoints_3d)}"
            )
        return self


class ContactReading(_Base):
    """contact.<finger>: confidence in [0,1] PLUS the source it came from.

    The source is not decoration. Master Spec §L2's arbiter ordering exists precisely
    because a 0.9 from a tactile sensor and a 0.9 from a vision heuristic are not the
    same claim, and the dexterous branch must be able to tell them apart.
    """

    confidence: Unit
    source: Provenance


class ObjectState(_Base):
    pose: SE3 | None = None
    mask: MaskRef | None = None


# --------------------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------------------

#: Fields that must declare where their value came from. A populated field missing from
#: `provenance` fails validation — see CanonicalFrame._provenance_complete.
PROVENANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "camera_pose",
    "hands",
    "finger_joints_human",
    "finger_joints_robotspace",
    "objects",
    "depth",
    "contact",
    "interaction_state",
)


class CanonicalFrame(_Base):
    t: float = Field(ge=0.0, description="seconds on the canonical clock")
    rig: RigType
    episode_id: str
    frame_idx: int = Field(ge=0)

    #: -> observation.images.*  Keys are camera names from the rig registry, enforced
    #: consistent from ingestion (Master Spec §3).
    images: dict[str, ImageRef] = Field(default_factory=dict)

    camera_pose: SE3 | None = None  # world frame, from SLAM/MPS

    hands: dict[Side, HandState] = Field(default_factory=dict)

    #: Human finger joint angles. L2 (glove, measured) or L1 (vision, estimated).
    finger_joints_human: dict[Side, tuple[float, ...]] | None = None

    #: Joint angles ALREADY IN ROBOT HAND SPACE. Only a DexUMI exoskeleton produces
    #: this, and it is measured ground truth for the retarget target itself — which is
    #: why such rigs bypass finger retargeting entirely (§L5 "DexUMI bypass").
    finger_joints_robotspace: dict[Side, tuple[float, ...]] | None = None

    objects: dict[str, ObjectState] = Field(default_factory=dict)
    depth: dict[str, DepthRef] = Field(default_factory=dict)

    #: contact.<finger>, per side. None == NOT MEASURED. See module docstring.
    contact: dict[Side, dict[Finger, ContactReading]] | None = None

    interaction_state: InteractionState | None = None

    confidence: dict[str, Unit] = Field(default_factory=dict)
    provenance: dict[str, Provenance] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _provenance_complete(self) -> CanonicalFrame:
        """Refuse to hold a value whose origin is unstated."""
        missing = []
        for name in PROVENANCE_REQUIRED_FIELDS:
            value = getattr(self, name)
            populated = value is not None and (not isinstance(value, dict) or len(value) > 0)
            if populated and name not in self.provenance:
                missing.append(name)
        if missing:
            raise ValueError(
                f"frame {self.frame_idx}: populated field(s) {missing} carry no provenance "
                "entry. Every value must declare whether it was measured or inferred — the "
                "trust model keys on it, so a value without one is invisible to certification."
            )
        return self

    @model_validator(mode="after")
    def _contact_is_five_fingers(self) -> CanonicalFrame:
        if self.contact is None:
            return self
        for side, readings in self.contact.items():
            missing = set(Finger) - set(readings)
            if missing:
                raise ValueError(
                    f"frame {self.frame_idx}: contact[{side.value}] is missing "
                    f"{sorted(f.value for f in missing)}. Report all five fingers or none — "
                    "a partial contact vector silently reads as 'not touching' downstream."
                )
        return self
