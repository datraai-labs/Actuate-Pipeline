"""L1 -- metric depth, intrinsics, per-pixel uncertainty. UniDepthV2 (Master Spec §L1).

The load-bearing model of Phase 3: WiLoR, SLAM and the whole metric-3D path were all
blocked on real depth and real intrinsics.
"""

from __future__ import annotations

from actuate.perception.depth.unidepth import (
    DepthFrame,
    DepthResult,
    UniDepthEstimator,
    backproject,
    run,
    sample_depth,
    smooth_root_depth,
    solve_root_depth,
)

__all__ = [
    "DepthFrame",
    "DepthResult",
    "UniDepthEstimator",
    "backproject",
    "run",
    "sample_depth",
    "solve_root_depth",
    "smooth_root_depth",
]
