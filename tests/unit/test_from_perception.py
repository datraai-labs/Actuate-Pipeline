"""The L1/L2 -> L3 wiring (build_from_perception) and the root-depth smoothing boundary fix.

build_from_perception is what makes the schema-v3 MANO reach the LeRobot exporter, so it gets a
test that the episode is v3, carries the full 45-component MANO, produces the 53-dim state, and
places the wrist at metric depth. The smoothing test pins the boundary-frame bug that would
otherwise make the first/last frames' wrist depth read too shallow.
"""

from __future__ import annotations

import json

import numpy as np

from actuate.canonical import (
    build_from_perception,
    episode_dof_names,
    state_and_action_vectors,
)
from actuate.config import InteractionState, Provenance, RigType, Side
from actuate.fusion import FrameFusion, FusionReport
from actuate.perception.depth import smooth_root_depth
from actuate.perception.depth.unidepth import DepthFrame, DepthResult


class _HF:
    def __init__(self, i: int):
        self.side = Side.RIGHT
        self.betas = np.zeros(10)
        self.hand_pose = np.full(45, 0.1)          # 45 axis-angle
        self.global_orient = np.array([0.1, 0.2, 0.3])
        kp = np.zeros((21, 3))
        kp[:, 0] = np.linspace(0, 0.1, 21)
        kp[:, 1] = 0.01 * i
        self.keypoints_3d = kp
        self.keypoints_2d = np.tile([32.0, 24.0], (21, 1))


class _HR:
    def __init__(self, n):
        self.frames = {i: [_HF(i)] for i in range(n)}


def _fusion(n):
    report = FusionReport(
        rig=RigType.HEAD_MOUNTED,
        states=[InteractionState.MOVING] * n,
    )
    for i in range(n):
        report.frames[i] = FrameFusion(
            interaction_state=InteractionState.MOVING,
            grasp={Side.RIGHT: i / max(n - 1, 1)},
            grasp_provenance=Provenance.VISION_FALLBACK,
            contact={},
            finger_joints_provenance={},
        )
    return report


def _session(tmp_path, n):
    (tmp_path / "session_meta.json").write_text(
        json.dumps({"session_id": "syn", "fps_nominal": 30, "frame_count": n})
    )
    return tmp_path


def test_build_from_perception_produces_v3_mano_and_53dim_state(tmp_path):
    n, H, W = 8, 48, 64
    K = np.array([[50, 0, W / 2], [0, 50, H / 2], [0, 0, 1]], dtype=np.float64)
    depth = DepthResult(intrinsics=K)
    for i in range(n):
        depth.frames[i] = DepthFrame(
            depth_m=np.full((H, W), 0.6), confidence=np.ones((H, W)), intrinsics=K
        )

    ep = build_from_perception(
        _session(tmp_path, n), "a" * 64, hands=_HR(n), depth=depth, fusion=_fusion(n),
        rig=RigType.HEAD_MOUNTED, task="synthetic", artifact_dir=tmp_path / "run",
    )

    assert ep.schema_version >= 3      # v3 introduced the 45-MANO this test pins below
    hand = next(iter(ep.frames[0].hands.values()))
    assert len(hand.mano.theta) == 45                       # full axis-angle, not 15-PCA
    # wrist placed at metric depth (~0.6 m), not floating at the origin
    assert 0.4 < hand.wrist_pose.position_m[2] < 0.8

    dof = episode_dof_names(ep)
    state, _, valid = state_and_action_vectors(ep)
    assert len(dof) == 53                                   # 8 wrist+grasp + 45 MANO
    assert state.shape[1] == 53
    assert valid.sum() >= n - 1
    # Dense depth survives beyond the ephemeral perception cache and every canonical frame
    # points to the correct row in the durable artifact.
    depth_path = tmp_path / "run" / "artifacts" / "depth" / "head.npz"
    assert depth_path.exists()
    with np.load(depth_path, allow_pickle=False) as arrays:
        assert arrays["depth_m"].shape == (n, H, W)
        assert arrays["confidence"].shape == (n, H, W)
        assert arrays["frame_indices"].tolist() == list(range(n))
    assert ep.frames[3].depth["head"].uri == "artifacts/depth/head.npz"
    assert ep.frames[3].depth["head"].frame_index == 3
    # UniDepth's relative-confidence raster is preserved in the NPZ but not relabeled as a
    # calibrated unit-interval probability in the customer certificate.
    assert "depth" not in ep.frames[3].confidence
    assert "hands" not in ep.frames[3].confidence  # the detector supplied no real score


def test_build_records_current_commercial_constraints_despite_cache_era_notes(tmp_path):
    """Old cached wording must not erase the active models' current license provenance."""
    n, height, width = 3, 24, 32
    intrinsics = np.array(
        [[30, 0, width / 2], [0, 30, height / 2], [0, 0, 1]], dtype=np.float64
    )
    depth = DepthResult(intrinsics=intrinsics)
    for i in range(n):
        depth.frames[i] = DepthFrame(
            depth_m=np.full((height, width), 0.6),
            confidence=np.ones((height, width)),
            intrinsics=intrinsics,
        )
    hands = _HR(n)
    hands.notes = {"hand_model": "wilor", "licence": "obsolete cache-era wording"}

    episode = build_from_perception(
        _session(tmp_path, n),
        "e" * 64,
        hands=hands,
        depth=depth,
        fusion=_fusion(n),
        rig=RigType.HEAD_MOUNTED,
        task="pick up cup",
    )

    constraints = episode.derivation_notes["usage_constraints"]
    assert "CC-BY-NC-ND-4.0" in constraints
    assert "UniDepth" in constraints and "CC-BY-NC-4.0" in constraints
    assert "obsolete cache-era wording" not in constraints


def test_stereo_uses_registry_camera_names_not_fictional_wrist(tmp_path):
    n = 3
    ep = build_from_perception(
        _session(tmp_path, n), "d" * 64, hands=_HR(n), fusion=_fusion(n),
        rig=RigType.STEREO, task="synthetic",
    )
    assert set(ep.frames[0].images) == {"stereo_left", "stereo_right"}
    assert "wrist" not in ep.frames[0].images


def test_build_from_perception_carries_object_masks(tmp_path):
    """Objects wire into the canonical frame as ObjectState (mask carried, pose None)."""
    import numpy as np

    from actuate.perception.objects.objects import ObjectFrame, ObjectResult
    from actuate.perception.objects.rle import decode_rle, encode_rle

    n = 4
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 10:20] = True
    objects = ObjectResult()
    for i in range(n):
        objects.frames[i] = [
            ObjectFrame(track_id=2, label="document", score=0.9, bbox=(10, 10, 20, 20),
                        mask_rle=encode_rle(mask))
        ]

    ep = build_from_perception(
        _session(tmp_path, n), "c" * 64, hands=_HR(n), objects=objects,
        rig=RigType.HEAD_MOUNTED, task="t",
    )
    obj = ep.frames[0].objects["2"]
    assert obj.pose is None                                  # 6-DoF not fabricated
    assert obj.mask.rle is not None
    assert np.array_equal(decode_rle(obj.mask.rle), mask)    # mask carried exactly
    from actuate.config import Provenance

    assert ep.frames[0].provenance["objects"] is Provenance.VISION_PRIMARY


def test_build_from_perception_does_not_carry_contact_on_a_barehand_rig(tmp_path):
    """The schema forbids contact on a rig that measures none; the wiring must respect it."""
    n = 4
    ep = build_from_perception(
        _session(tmp_path, n), "b" * 64, hands=_HR(n), rig=RigType.HEAD_MOUNTED, task="t",
    )
    for f in ep.frames:
        assert f.contact is None                            # not fabricated from vision


def test_smooth_root_depth_does_not_pull_boundary_frames_shallow():
    """Boundary fix: a constant depth must stay constant everywhere, including the first/last
    frames. A plain convolve(..., 'same') zero-pads and would read the edges ~4/7 too shallow."""
    z = np.full(8, 0.6)
    out = smooth_root_depth(z, window=7)
    assert np.allclose(out, 0.6), f"boundary frames drifted: {out}"
    # the broken (zero-padded) version would have made out[0] ~= 0.6*4/7 = 0.343
    assert out[0] > 0.55 and out[-1] > 0.55
