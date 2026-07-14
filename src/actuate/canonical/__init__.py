"""L3 -- Canonical Representation.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L3

The frozen schema lives in `actuate.schema` (§3); its Parquet/Zarr persistence in
`actuate.io.store`. This package turns v1's processed outputs into it.

`reproject.py` implements stable-frame reprojection and REFUSES to run without ego-motion
-- on a moving-camera rig an un-reprojected wrist delta is hand motion + head motion, and
L1 SLAM is not built. The refusal is the point.
"""

from __future__ import annotations

from actuate.canonical.build import (
    DOF_NAMES,
    CanonicalBuildError,
    build_episode,
    state_and_action_vectors,
)
from actuate.canonical.reproject import (
    EgoMotionUnavailable,
    action_is_ego_contaminated,
    relative,
    reproject_future_pose,
)

__all__ = [
    "DOF_NAMES",
    "CanonicalBuildError",
    "EgoMotionUnavailable",
    "action_is_ego_contaminated",
    "build_episode",
    "relative",
    "reproject_future_pose",
    "state_and_action_vectors",
]
