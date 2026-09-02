"""Product capability records consumed by local orchestration clients.

The registries describe every supported data contract.  A dashboard also needs to know
which of those contracts its current upload flow can honestly execute.  Keep that policy
next to the registries so clients never maintain a second, drifting list.
"""

from __future__ import annotations

from actuate.config.embodiments import EMBODIMENT_REGISTRY
from actuate.config.enums import RigType
from actuate.config.rigs import RIG_REGISTRY


# The browser currently uploads one video and one optional IMU sidecar. Sensor-rich and
# multi-camera rigs need a source adapter that preserves their named streams, so exposing
# them as selectable here would guarantee an ingest failure.
_BROWSER_UPLOAD_RIGS = frozenset({RigType.HEAD_MOUNTED})


def product_capabilities() -> dict[str, list[dict[str, object]]]:
    """Return a JSON-safe view of the Core registries for product clients."""
    rigs: list[dict[str, object]] = []
    for rig, spec in RIG_REGISTRY.items():
        enabled = rig in _BROWSER_UPLOAD_RIGS
        status = (
            "Available for local video upload."
            if enabled
            else "Registered in Core; requires a source adapter that preserves its sensor streams."
        )
        rigs.append(
            {
                "id": rig.value,
                "description": spec.description,
                "cameras": list(spec.cameras),
                "measured_channels": sorted(channel.value for channel in spec.measured),
                "depth_model": spec.depth_model,
                "enabled": enabled,
                "status": status,
            }
        )

    embodiments: list[dict[str, object]] = []
    for name, spec in EMBODIMENT_REGISTRY.items():
        ready = spec.is_retarget_ready
        embodiments.append(
            {
                "id": name,
                "description": spec.description,
                "kinematic_model": ready,
                "real_data_validated": spec.sim_validated,
                "enabled": ready,
            }
        )

    return {"rigs": rigs, "embodiments": embodiments}
