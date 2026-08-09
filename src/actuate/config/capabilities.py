"""Serializable product capabilities derived from the registries.

Registry presence is not the same as validated product support. Keeping both fields prevents
the dashboard from turning an architectural placeholder into a selectable promise.
"""
from __future__ import annotations

from actuate.config.embodiments import EMBODIMENT_REGISTRY
from actuate.config.enums import RigType
from actuate.config.rigs import RIG_REGISTRY

_RIG_PRODUCT_STATUS = {
    RigType.HEAD_MOUNTED: (True, "Real-data validated for egocentric RGB capture."),
    RigType.STEREO: (
        False,
        "Registered but not selectable until calibrated stereo is validated end to end.",
    ),
    RigType.UMI_GRIPPER: (False, "Deferred; no active real-data validation."),
    RigType.GLOVE: (False, "Deferred; no active real-data validation."),
    RigType.TELEOP_ROBOT: (False, "Deferred; no active real-data validation."),
    RigType.DEXUMI_EXOSKELETON: (False, "Deferred; no active real-data validation."),
}


def product_capabilities() -> dict:
    rigs = []
    for rig_type, spec in RIG_REGISTRY.items():
        enabled, status = _RIG_PRODUCT_STATUS[rig_type]
        rigs.append({
            "id": rig_type.value,
            "description": spec.description,
            "cameras": list(spec.cameras),
            "measured_channels": sorted(channel.value for channel in spec.measured),
            "depth_model": spec.depth_model,
            "enabled": enabled,
            "status": status,
        })
    embodiments = [
        {
            "id": name,
            "description": spec.description,
            "kinematic_model": spec.urdf_path is not None,
            "real_data_validated": spec.sim_validated,
            "enabled": spec.urdf_path is not None,
        }
        for name, spec in EMBODIMENT_REGISTRY.items()
    ]
    return {"rigs": rigs, "embodiments": embodiments}
