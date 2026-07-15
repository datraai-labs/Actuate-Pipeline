"""L1 -- ego-motion. `run(store, method="auto") -> SlamResult`.

Without camera pose, the "action" on a moving-camera rig is hand motion + head motion. On
the real capture the head moves 94% of the time and contributes a median 36% of the emitted
action. This package makes that subtractable.
"""

from __future__ import annotations

from actuate.perception.slam.runner import SlamError, run
from actuate.perception.slam.vio import (
    SlamResult,
    approximate_intrinsics,
    gyro_relative_rotations,
    vision_relative_rotation,
)

__all__ = [
    "SlamError",
    "SlamResult",
    "approximate_intrinsics",
    "gyro_relative_rotations",
    "run",
    "vision_relative_rotation",
]
