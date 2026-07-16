"""L1 -- metric depth, intrinsics, per-pixel uncertainty. Master Spec §L1.

Default is UniDepthV2 (single-image, estimates intrinsics). A temporal/video-depth challenger
(Video-Depth-Anything, or a flow-warped filter over any base) is available via `run(model=...)`
and is A/B-scored against UniDepth with `consistency.compare` -- the metric that decides whether
a video model actually beats UniDepth's per-frame noise floor (Part C).
"""

from __future__ import annotations

from pathlib import Path

from actuate.perception.depth.benchmark import (
    GATE_WRIST_JITTER_MM,
    DepthBenchmarkRow,
    benchmark_row,
    run_benchmark,
    wrist_jitter,
)
from actuate.perception.depth.consistency import (
    ConsistencyReport,
    compare,
    consistency_score,
    sample_tracked_depths,
)
from actuate.perception.depth.unidepth import (
    DepthFrame,
    DepthResult,
    UniDepthEstimator,
    backproject,
    sample_depth,
    smooth_root_depth,
    solve_root_depth,
)
from actuate.perception.depth.unidepth import run as _unidepth_run

_TEMPORAL_MODELS = {"video_depth_anything", "vda", "flow_filter", "temporal"}


def run(session_dir: Path, model: str = "auto", **kw):
    """Depth as a `DepthResult`. One entry, a `model=` switch:

    - "auto"/"unidepth_v2"/"vitl"/"vits"  -> UniDepthV2 (single-image; estimates intrinsics)
    - "moge2"                             -> MoGe-2 (single-image; metric + focal; Kaggle)
    - "video_depth_anything"/"vda"        -> Video-Depth-Anything (temporal; Kaggle)
    - "flow_filter"/"temporal"            -> flow-warped temporal EMA over a `base=` DepthResult

    All return the same `DepthResult`, so any consumer is agnostic to which model produced it.
    Temporal video depth is best paired with a metric anchor -- see `temporal.anchor_scale`.
    """
    if model in ("moge2", "moge-2", "moge"):
        from actuate.perception.depth import moge

        return moge.run(session_dir, max_frames=kw.get("max_frames"))
    if model in _TEMPORAL_MODELS:
        from actuate.perception.depth import temporal

        backend = "flow_filter" if model in ("flow_filter", "temporal") else "video_depth_anything"
        return temporal.run(
            session_dir, backend=backend,
            base=kw.get("base"), intrinsics=kw.get("intrinsics"),
            encoder=kw.get("encoder", "vits"), max_frames=kw.get("max_frames"),
            alpha=kw.get("alpha", 0.5),
        )
    return _unidepth_run(session_dir, model=model, **kw)


__all__ = [
    "DepthFrame",
    "DepthResult",
    "UniDepthEstimator",
    "backproject",
    "run",
    "sample_depth",
    "solve_root_depth",
    "smooth_root_depth",
    "ConsistencyReport",
    "compare",
    "consistency_score",
    "sample_tracked_depths",
    "GATE_WRIST_JITTER_MM",
    "DepthBenchmarkRow",
    "benchmark_row",
    "run_benchmark",
    "wrist_jitter",
]
