"""ORB-SLAM3 backend -- the Master Spec §L1 default. NOT BUILT.

Registered so the backend is a config switch rather than a rewrite, and so its absence is
explicit rather than implied.

Why it is not built here (checked, not assumed, on 2026-07-14):

  - No pip distribution exists (`orbslam3` on PyPI is an empty 0.0 placeholder).
  - No C++ compiler on this machine (no MSVC). Building from source needs Pangolin, DBoW2,
    Eigen and OpenCV's C++ libs -- realistically a WSL Linux toolchain and several hours.
  - It solves a strictly harder problem than we need: stable-frame reprojection needs
    RELATIVE pose over <= one action chunk (~0.5 s), never a globally consistent map and
    never loop closure.

What it would buy us, and when it is worth the build:

  - A globally consistent trajectory (needed if we ever want world-frame object persistence
    or cross-episode scene reconstruction).
  - Metric scale from visual-inertial initialisation, without needing dense depth.
  - Loop closure, which bounds long-horizon drift.

None of those are required for ego-motion subtraction, which is what Phase 3 Part A exists
to do. Revisit when a downstream layer actually needs a map.
"""

from __future__ import annotations

from pathlib import Path

from actuate.perception.slam.vio import SlamResult


def run(session_dir: Path) -> SlamResult:  # noqa: ARG001
    raise NotImplementedError(
        "ORB-SLAM3 is not built. It has no pip distribution and needs a C++ toolchain "
        "(Pangolin/DBoW2/Eigen) that this machine does not have.\n\n"
        "Use method='vio': gyro-derived rotation (a direct sensor measurement, drift-free "
        "over a 33 ms frame) plus visual translation. Reprojection needs relative pose over "
        "<= one action chunk, not a global map -- see docs and this module's docstring."
    )
