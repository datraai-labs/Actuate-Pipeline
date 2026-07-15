"""Part F visualization: the pure conventions, and log_episode against the real Rerun API.

The conventions (skeleton topology, colours, coordinate frame) are viewer-free and get exact
tests. `log_episode` is exercised against a real in-file Rerun recording built from synthetic
perception results -- no GPU, no models -- so the logging path itself is covered even though the
full "open the viewer on the real capture" gate is a manual/GPU step.
"""

from __future__ import annotations

import numpy as np
import pytest

from actuate.config import Finger, InteractionState, Side
from actuate.viz.conventions import (
    FINGERTIPS,
    HAND_COLORS,
    HAND_EDGES,
    STATE_COLORS,
    contact_color,
    skeleton_strips,
)


def test_hand_edges_form_a_valid_tree_over_21_keypoints():
    # every index used is in [0, 20]
    idx = {i for e in HAND_EDGES for i in e}
    assert idx <= set(range(21))
    # 20 edges for a 21-node tree (5 fingers x 4 segments)
    assert len(HAND_EDGES) == 20
    # every non-wrist keypoint has exactly one parent (it's a tree rooted at the wrist)
    children = [b for _, b in HAND_EDGES]
    assert sorted(children) == list(range(1, 21))
    assert all(a == 0 or a in children for a, _ in HAND_EDGES)


def test_fingertips_are_the_leaf_indices():
    assert set(FINGERTIPS.values()) == {4, 8, 12, 16, 20}
    # a fingertip is never a parent
    parents = {a for a, _ in HAND_EDGES}
    assert not (set(FINGERTIPS.values()) & parents)


def test_state_colors_follow_the_spec_green_blue_red():
    assert STATE_COLORS[InteractionState.STATIC][1] > 150   # green channel dominant
    assert STATE_COLORS[InteractionState.MOVING][0] > 150   # red channel dominant
    assert STATE_COLORS[InteractionState.GRASPED_R][2] > 150  # blue channel dominant


def test_hand_colors_distinguish_left_from_right():
    assert HAND_COLORS[Side.LEFT] != HAND_COLORS[Side.RIGHT]


def test_contact_color_is_brighter_for_higher_confidence():
    dim = contact_color(0.0)
    bright = contact_color(1.0)
    assert bright[1] > dim[1]                       # green ramp
    assert contact_color(float("nan")) == dim       # NaN clamps to the dim end, never crashes
    # out-of-range clamps
    assert contact_color(5.0) == contact_color(1.0)


def test_skeleton_strips_skip_non_finite_points():
    pos = np.zeros((21, 3))
    pos[8] = [np.nan, 0, 0]  # drop an index fingertip
    strips = skeleton_strips(pos)
    # the edge (7,8) touches the NaN point and must be dropped; 19 of 20 remain
    assert len(strips) == 19
    # each surviving strip is two 3-vectors
    assert all(len(s) == 2 and len(s[0]) == 3 for s in strips)


# --------------------------------------------------------------------------------------
# log_episode against a real Rerun recording (no GPU)
# --------------------------------------------------------------------------------------


class _HF:
    def __init__(self, side):
        self.side = side
        # a small open hand: keypoints spread in a plane, root at index 0
        kp = np.zeros((21, 3))
        for i in range(21):
            kp[i] = [0.01 * (i % 5), 0.01 * (i // 5), 0.0]
        self.keypoints_3d = kp
        # 2D keypoints in-bounds for the small (48x64) test images, clustered near centre so
        # depth sampling at the wrist actually hits the depth map.
        self.keypoints_2d = np.tile([30.0, 24.0], (21, 1)) + np.random.default_rng(0).uniform(
            -3, 3, (21, 2)
        )
        self.bbox = (24.0, 18.0, 36.0, 30.0)


class _Frames:
    def __init__(self, frames, intrinsics=None):
        self.frames = frames
        if intrinsics is not None:
            self.intrinsics = intrinsics


def test_log_episode_writes_every_modality_to_a_real_rrd(tmp_path):
    rr = pytest.importorskip("rerun")
    from actuate.fusion import ContactPoint, FrameFusion
    from actuate.perception.depth.unidepth import DepthFrame
    from actuate.perception.objects.objects import ObjectFrame
    from actuate.perception.objects.rle import encode_rle
    from actuate.viz import log_episode

    N = 5
    H, W = 48, 64
    K = np.array([[50, 0, W / 2], [0, 50, H / 2], [0, 0, 1]], dtype=np.float64)

    hands = _Frames({i: [_HF(Side.RIGHT)] for i in range(N)})
    depth = _Frames(
        {i: DepthFrame(depth_m=np.full((H, W), 0.6), confidence=np.ones((H, W)), intrinsics=K)
         for i in range(N)},
        intrinsics=K,
    )
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 10:20] = True
    objects = _Frames(
        {i: [ObjectFrame(track_id=1, label="document", score=0.9, bbox=(10, 10, 20, 20),
                         mask_rle=encode_rle(mask))] for i in range(N)}
    )
    from actuate.config import Provenance

    fusion = _Frames(
        {i: FrameFusion(
            interaction_state=InteractionState.STATIC,
            grasp={Side.RIGHT: 0.1},
            grasp_provenance=Provenance.VISION_FALLBACK,
            contact={Side.RIGHT: {f: ContactPoint(0.1, Provenance.VISION_FALLBACK) for f in Finger}},
            finger_joints_provenance={Side.RIGHT: Provenance.VISION_PRIMARY},
        ) for i in range(N)}
    )
    video = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(N)]

    out = tmp_path / "ep.rrd"
    rr.init("actuate-test")
    counts = log_episode(hands=hands, depth=depth, objects=objects, fusion=fusion,
                         video_frames=video, intrinsics=K)
    rr.save(str(out))

    # every modality logged once per frame, and the recording is a real non-trivial file
    for key in ("video", "depth", "hand", "objects", "state", "contact"):
        assert counts[key] == N, f"{key} logged {counts[key]} times, expected {N}"
    assert counts["action"] == N - 1  # no action on the first frame (no previous root)
    assert out.exists() and out.stat().st_size > 1000


def test_stage_cache_runs_on_miss_loads_on_hit_reruns_on_change(tmp_path):
    """The `actuate viz --cache` behaviour: never re-run a stage whose inputs are unchanged."""
    from actuate.cli.viz import _stage_cached

    calls = {"n": 0}

    def run_fn():
        calls["n"] += 1
        return {"v": calls["n"]}

    # no --cache -> always runs
    _, src = _stage_cached(tmp_path, "depth", "n=20", use_cache=False, force=False, run_fn=run_fn)
    assert src == "ran" and calls["n"] == 1

    # first --cache -> miss -> runs + writes
    _, src = _stage_cached(tmp_path, "depth", "n=20", use_cache=True, force=False, run_fn=run_fn)
    assert src == "ran" and calls["n"] == 2

    # same key -> hit -> loads, does NOT run; returns the cached value
    r, src = _stage_cached(tmp_path, "depth", "n=20", use_cache=True, force=False, run_fn=run_fn)
    assert src == "cache" and calls["n"] == 2 and r["v"] == 2

    # changed input (different frame count) -> miss -> runs
    _, src = _stage_cached(tmp_path, "depth", "n=40", use_cache=True, force=False, run_fn=run_fn)
    assert src == "ran" and calls["n"] == 3

    # --force -> re-runs even on a hit
    _, src = _stage_cached(tmp_path, "depth", "n=20", use_cache=True, force=True, run_fn=run_fn)
    assert src == "ran" and calls["n"] == 4

    # a corrupt cache file falls through to a re-run rather than crashing
    for bad in (tmp_path / ".actuate_cache").glob("depth_*.pkl"):
        bad.write_bytes(b"not a pickle")
    _, src = _stage_cached(tmp_path, "depth", "n=20", use_cache=True, force=False, run_fn=run_fn)
    assert src == "ran"


def test_log_episode_places_hand_at_metric_depth_not_at_origin(tmp_path):
    """The hand root must land near the depth plane (~0.6 m), not at the camera origin --
    this is the 'not floating' requirement, checked numerically on the placement math."""
    pytest.importorskip("rerun")
    from actuate.perception.depth import backproject, solve_root_depth
    from actuate.perception.depth.unidepth import DepthFrame

    H, W = 48, 64
    K = np.array([[50, 0, W / 2], [0, 50, H / 2], [0, 0, 1]], dtype=np.float64)
    h = _HF(Side.RIGHT)
    df = DepthFrame(depth_m=np.full((H, W), 0.6), confidence=np.ones((H, W)), intrinsics=K)
    z = solve_root_depth(df.depth_m, df.confidence, h.keypoints_2d, h.keypoints_3d)
    root_cam = backproject(h.keypoints_2d[0], z, K)
    assert 0.4 < root_cam[2] < 0.8, "hand root should sit at the ~0.6 m depth plane"
