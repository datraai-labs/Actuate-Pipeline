"""Temporal depth: the A/B consistency metric and the flow-warped temporal filter.

The Video-Depth-Anything backend needs a GPU + the model, so it is validated on Kaggle (see
kaggle/). What is validated here: the consistency SCORE (the A/B decision function) and that the
flow_filter actually reduces temporal wobble -- both pure/CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from actuate.perception.depth import consistency_score
from actuate.perception.depth.temporal import _flow_filter
from actuate.perception.depth.unidepth import DepthFrame, DepthResult


def test_consistency_score_is_zero_for_a_perfectly_stable_scene():
    r = consistency_score(np.full((10, 50), 0.6))
    assert r.median_wobble_pct < 1e-6
    assert r.full_swing_pct < 1e-6


def test_consistency_score_recovers_the_injected_wobble_in_mm():
    """A static scene at 0.6 m with 2% noise should score ~2% / ~12 mm -- the metric must map
    to physical millimetres, since that is what gets compared against hand motion (Part C)."""
    rng = np.random.default_rng(0)
    d = 0.6 * (1 + rng.normal(0, 0.02, (12, 60)))
    r = consistency_score(d)
    assert 1.0 < r.median_wobble_pct < 3.0
    assert 8.0 < r.wobble_mm < 18.0            # ~0.02 * 600 mm, in the Part C ballpark
    assert r.mean_depth_m == pytest.approx(0.6, abs=0.02)


def test_consistency_score_rejects_a_single_frame():
    with pytest.raises(ValueError):
        consistency_score(np.full((1, 10), 0.6))


def test_flow_filter_reduces_temporal_wobble_on_a_static_scene():
    """A static (but textured) scene with per-frame depth noise: the flow-warped EMA must read
    a pixel more consistently over time than the raw model did."""
    rng = np.random.default_rng(1)
    T, H, W = 10, 64, 80
    texture = (rng.random((H, W, 3)) * 255).astype(np.uint8)   # same every frame -> zero flow
    frames = [texture.copy() for _ in range(T)]

    base = DepthResult(intrinsics=np.eye(3), model="synthetic")
    for i in range(T):
        base.frames[i] = DepthFrame(
            depth_m=(0.6 * (1 + rng.normal(0, 0.02, (H, W)))).astype(np.float32),
            confidence=np.ones((H, W), np.float32), intrinsics=np.eye(3),
        )

    filt = _flow_filter(base, frames, alpha=0.4)

    def center_wobble(res):
        v = np.array([res.frames[i].depth_m[H // 2, W // 2] for i in range(T)])
        return np.std(np.diff(v)) / v.mean()

    assert center_wobble(filt) < center_wobble(base)
    # and it stays interface-compatible: same frames, a DepthResult, notes recording the method
    assert set(filt.frames) == set(base.frames)
    assert "temporal" in filt.notes


def test_flow_filter_requires_a_base():
    from actuate.perception.depth import run as depth_run

    with pytest.raises(ValueError, match="base"):
        depth_run("nonexistent_session", model="flow_filter")
