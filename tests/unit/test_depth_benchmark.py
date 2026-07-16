"""Depth benchmark: keyframe scale-anchoring and the wrist-z-jitter gate metric.

The models (MoGe-2, Video-Depth-Anything) run on Kaggle; validated here are the two pure pieces
that make the A/B a decision rather than a vibe: the affine scale-anchor (must give the anchor's
scale while keeping the video model's consistency) and the wrist-jitter metric (must fall on a
consistent depth and rise on a noisy one, in millimetres).
"""

from __future__ import annotations

import numpy as np

from actuate.config import Side
from actuate.perception.depth.benchmark import (
    GATE_WRIST_JITTER_MM,
    run_benchmark,
    wrist_jitter,
)
from actuate.perception.depth.temporal import anchor_scale
from actuate.perception.depth.unidepth import DepthFrame, DepthResult

H, W, T = 48, 64, 12
K = np.array([[50, 0, W / 2], [0, 50, H / 2], [0, 0, 1]], dtype=np.float64)


def _const_depth(val, model="m", intr=K):
    r = DepthResult(intrinsics=intr, model=model)
    for i in range(T):
        r.frames[i] = DepthFrame(depth_m=np.full((H, W), val, np.float32),
                                 confidence=np.ones((H, W), np.float32), intrinsics=K)
    return r


def _noisy_depth(mean, sigma, seed=0, model="m"):
    rng = np.random.default_rng(seed)
    r = DepthResult(intrinsics=K, model=model)
    for i in range(T):
        r.frames[i] = DepthFrame(
            depth_m=(mean * (1 + rng.normal(0, sigma, (H, W)))).astype(np.float32),
            confidence=np.ones((H, W), np.float32), intrinsics=K)
    return r


def test_anchor_scale_gives_anchor_scale_and_keeps_video_consistency():
    # video: temporally consistent (constant) but wrong scale (0.25). anchor: right scale, noisy.
    video = _const_depth(0.25, model="vda", intr=None)
    anchor = _noisy_depth(0.6, 0.03, model="unidepth")

    out = anchor_scale(video, anchor, keyframe_stride=4)

    mean_after = np.mean([out.frames[i].depth_m.mean() for i in range(T)])
    jit_anchor = np.std(np.diff([anchor.frames[i].depth_m.mean() for i in range(T)]))
    jit_out = np.std(np.diff([out.frames[i].depth_m.mean() for i in range(T)]))
    assert abs(mean_after - 0.6) < 0.03          # took the anchor's metric scale
    assert jit_out < jit_anchor                  # kept the video model's low jitter
    assert out.intrinsics is not None            # intrinsics come from the anchor


class _HF:
    def __init__(self):
        self.side = Side.RIGHT
        self.keypoints_3d = np.zeros((21, 3))    # flat -> root-relative dz = 0
        self.keypoints_2d = np.tile([W / 2, H / 2], (21, 1))


class _HR:
    def __init__(self):
        self.frames = {i: [_HF()] for i in range(T)}


def test_wrist_jitter_is_low_on_consistent_depth_high_on_noisy():
    raw_c, _, med_c = wrist_jitter(_const_depth(0.6), _HR())
    raw_n, _, _ = wrist_jitter(_noisy_depth(0.6, 0.03), _HR())
    assert raw_c < 1.0                           # a stable depth places the wrist stably
    assert raw_n > raw_c                          # noise shows up as jitter
    assert abs(med_c - 0.6) < 0.05               # metric scale reported in metres


def test_run_benchmark_reports_the_gate_verdict():
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    report = run_benchmark(
        {"UniDepthV2": _noisy_depth(0.6, 0.03), "temporal": _const_depth(0.6)},
        _HR(), frames, baseline="UniDepthV2",
    )
    assert "GATE" in report
    assert f"< {GATE_WRIST_JITTER_MM:.0f} mm" in report
    # the consistent 'temporal' model must be reported as passing the gate
    assert "temporal" in report
