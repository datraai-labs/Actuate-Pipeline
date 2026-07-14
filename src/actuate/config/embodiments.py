"""Embodiment registry — Master Spec §2.3.

The second extension point: one entry per target robot, mapping it to its kinematic
model (URDF), hand type/DoF, control modes, and camera config. A new customer robot
becomes a supported delivery target by adding an entry here — not by touching any layer.

**Honest status: no embodiment is retarget-capable yet.** L5 does not exist (Master Spec
§1 marks it net-new). Every entry below therefore has `urdf_path=None` and
`sim_validated=False`. They are declarations of intent, and `retarget_eligibility` in the
canonical schema stays `{}` until an entry has a real URDF and a passing sim replay.

The one entry that is load-bearing today is CANONICAL_REFERENCE_HAND: Master Spec §3
makes "retargeted joints on a canonical high-DoF reference hand" the *delivered Stage-I
pretraining action target*, so the exporter needs to know which hand that is. §7 item 4
lists picking it as a benchmark-before-lock decision — so it is UNRESOLVED, and the
schema must not pretend otherwise.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from actuate.config.enums import ControlMode


class HandSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dof: int
    n_fingers: int


class EmbodimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    arm_dof: int
    hand: HandSpec | None = None
    control_modes: tuple[ControlMode, ...] = (ControlMode.JOINT, ControlMode.EE)
    cameras: tuple[str, ...] = ()

    #: None == no kinematic model registered. Without one there is no IK, no sim
    #: replay, and therefore no honest claim of retarget-eligibility.
    urdf_path: str | None = None

    #: True only after a real MuJoCo/Isaac replay passed collision + joint-limit +
    #: contact-stability + no-slip on a real episode (Master Spec §L5 gate).
    sim_validated: bool = False

    @property
    def is_dexterous(self) -> bool:
        return self.hand is not None and self.hand.dof > 2

    @property
    def is_retarget_ready(self) -> bool:
        """Everything L5 needs actually exists for this robot."""
        return self.urdf_path is not None


#: Master Spec §3/§L5: MANO is the intermediate; the DELIVERED Stage-I pretraining
#: action target is relative-SE(3) wrist + retargeted joints on a canonical high-DoF
#: reference hand (~20+ DoF). §7 item 4 says: pick it via benchmark, not from memory.
#: It is not picked. Leaving this None is the honest state — the alternative is
#: hardcoding a guess that the Stage-I exporter would then bake into every delivered
#: dataset.
CANONICAL_REFERENCE_HAND: HandSpec | None = None


EMBODIMENT_REGISTRY: dict[str, EmbodimentSpec] = {
    "franka_dual": EmbodimentSpec(
        name="franka_dual",
        description="Dual Franka Panda, parallel grippers. Non-dexterous.",
        arm_dof=7,
        hand=HandSpec(name="panda_gripper", dof=1, n_fingers=2),
        cameras=("top", "wrist"),
    ),
    "unitree_g1": EmbodimentSpec(
        name="unitree_g1",
        description="Unitree G1 humanoid. Dexterous hands.",
        arm_dof=7,
        hand=HandSpec(name="unitree_three_finger", dof=7, n_fingers=3),
        cameras=("head", "wrist"),
    ),
}


def get_embodiment(name: str) -> EmbodimentSpec:
    if name not in EMBODIMENT_REGISTRY:
        raise KeyError(
            f"unknown embodiment {name!r}; registered: {sorted(EMBODIMENT_REGISTRY)}"
        )
    return EMBODIMENT_REGISTRY[name]


def retarget_ready_embodiments() -> list[str]:
    """Embodiments L5 could actually retarget to. Currently empty, and that is correct."""
    return sorted(n for n, e in EMBODIMENT_REGISTRY.items() if e.is_retarget_ready)
