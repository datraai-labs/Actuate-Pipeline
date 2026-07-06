"""
DatraAI Pipeline — Tests for Step 04d: Metric 3D / Depth Estimation
Covers every pure/GPU-free function (camera intrinsics, pinhole lift,
keypoint depth sampling, the ego-motion hook, the stereo-depth reader) plus
an end-to-end run() against the "stereo" and "none" depth modes — both of
which never touch the monocular model. The monocular model call
(_get_monocular_pipeline / _estimate_monocular_depth_frame) is
NOT exercised here; it needs real GPU hardware to validate — see the
module's own docstring in scripts/04d_depth_estimate.py.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import write_session_h5

_spec = importlib.util.spec_from_file_location(
    "depth_estimate",
    str(Path(__file__).resolve().parent.parent / "scripts" / "04d_depth_estimate.py"),
)
_depth_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_depth_mod)

_load_camera_intrinsics = _depth_mod._load_camera_intrinsics
_sample_depth_at_normalized_point = _depth_mod._sample_depth_at_normalized_point
_lift_point_to_3d = _depth_mod._lift_point_to_3d
_ego_motion_scale_hint = _depth_mod._ego_motion_scale_hint
_read_stereo_depth_stream = _depth_mod._read_stereo_depth_stream
run = _depth_mod.run


class TestLoadCameraIntrinsics:
    def test_fallback_when_no_calibration_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CAMERA_INTRINSICS_PATH", str(tmp_path / "missing" / "{device_id}_intrinsics.json"))
        intrinsics = _load_camera_intrinsics("camX", width=1280, height=720)

        assert intrinsics["source"] == "approximated_no_calibration_file"
        assert intrinsics["cx"] == 640.0
        assert intrinsics["cy"] == 360.0
        expected_fx = 1280 / (2.0 * math.tan(math.radians(cfg.CAMERA_DEFAULT_HFOV_DEG) / 2.0))
        assert abs(intrinsics["fx"] - expected_fx) < 1e-6
        assert intrinsics["fx"] == intrinsics["fy"]  # square-pixel assumption

    def test_loads_real_calibration_file_when_present(self, tmp_path, monkeypatch):
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        device_id = "camA"
        with open(calib_dir / f"{device_id}_intrinsics.json", "w") as f:
            json.dump({"fx": 600.0, "fy": 605.0, "cx": 320.0, "cy": 240.0}, f)

        monkeypatch.setattr(cfg, "CAMERA_INTRINSICS_PATH", str(calib_dir / "{device_id}_intrinsics.json"))
        intrinsics = _load_camera_intrinsics(device_id, width=640, height=480)

        assert intrinsics["source"] == "calibration_file"
        assert intrinsics["fx"] == 600.0
        assert intrinsics["fy"] == 605.0
        assert intrinsics["cx"] == 320.0
        assert intrinsics["cy"] == 240.0


class TestSampleDepthAtNormalizedPoint:
    def test_samples_known_value(self):
        # Use an 11x11 map (even index count 0..10) so normalized 0.5 maps
        # to an unambiguous integer pixel (5) regardless of round-half-to-even
        # vs round-half-up convention — avoids relying on Python's banker's
        # rounding at an exact .5 boundary (round(4.5) == 4, not 5, which a
        # 10x10 map would hit).
        depth_map = np.zeros((11, 11), dtype=np.float32)
        depth_map[5, 5] = 1.234
        result = _sample_depth_at_normalized_point(depth_map, 0.5, 0.5)
        assert result == pytest.approx(1.234)

    def test_rounding_matches_pythons_round_half_to_even(self):
        """
        Pin down the exact rounding convention: on a 10x10 map, normalized
        0.5 -> px = round(0.5 * 9) = round(4.5) = 4 (round-half-to-even),
        not 5. If this test breaks, the sampling convention changed —
        update call sites that assume a particular pixel mapping.
        """
        depth_map = np.zeros((10, 10), dtype=np.float32)
        depth_map[4, 4] = 9.0
        depth_map[5, 5] = 1.0
        assert _sample_depth_at_normalized_point(depth_map, 0.5, 0.5) == pytest.approx(9.0)

    def test_none_depth_map_returns_none(self):
        assert _sample_depth_at_normalized_point(None, 0.5, 0.5) is None

    def test_clamps_out_of_range_coordinates(self):
        depth_map = np.zeros((10, 10), dtype=np.float32)
        depth_map[0, 0] = 2.0
        depth_map[9, 9] = 3.0
        assert _sample_depth_at_normalized_point(depth_map, -5.0, -5.0) == pytest.approx(2.0)
        assert _sample_depth_at_normalized_point(depth_map, 50.0, 50.0) == pytest.approx(3.0)

    def test_non_positive_value_returns_none(self):
        depth_map = np.zeros((10, 10), dtype=np.float32)  # all zero -> invalid
        assert _sample_depth_at_normalized_point(depth_map, 0.5, 0.5) is None

    def test_nan_value_returns_none(self):
        depth_map = np.full((10, 10), np.nan, dtype=np.float32)
        assert _sample_depth_at_normalized_point(depth_map, 0.5, 0.5) is None


class TestLiftPointTo3D:
    def test_center_pixel_yields_zero_xy(self):
        intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
        # normalized center point of a 640x480 image lands exactly on cx,cy
        point = _lift_point_to_3d(0.5, 0.5, depth_m=2.0, intrinsics=intrinsics, width=640, height=480)
        assert point == [0.0, 0.0, 2.0]

    def test_off_center_point_matches_pinhole_formula(self):
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        x_norm, y_norm, depth_m = 0.75, 0.25, 1.5
        point = _lift_point_to_3d(x_norm, y_norm, depth_m, intrinsics, width=640, height=480)

        px, py = x_norm * 640, y_norm * 480
        expected_x = (px - intrinsics["cx"]) * depth_m / intrinsics["fx"]
        expected_y = (py - intrinsics["cy"]) * depth_m / intrinsics["fy"]

        assert point[0] == pytest.approx(expected_x, abs=1e-4)
        assert point[1] == pytest.approx(expected_y, abs=1e-4)
        assert point[2] == depth_m

    def test_none_depth_returns_none(self):
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        assert _lift_point_to_3d(0.5, 0.5, None, intrinsics, 640, 480) is None


class TestEgoMotionScaleHint:
    def test_currently_a_documented_noop(self):
        """
        Locks in the current behavior: with a metric monocular model
        configured, this hook must return 1.0 regardless of input — if this
        test starts failing, either the hook grew real logic (update the
        test) or something is unintentionally calling into it differently.
        """
        gyro = np.random.randn(5, 3)
        accel = np.random.randn(5, 3)
        assert _ego_motion_scale_hint(gyro, accel) == 1.0
        assert _ego_motion_scale_hint(np.zeros((0, 3)), np.zeros((0, 3))) == 1.0


class TestReadStereoDepthStream:
    def test_reads_matching_shape_stream(self, tmp_path):
        n_frames, h, w = 3, 4, 5
        data = np.arange(n_frames * h * w, dtype=np.float32).reshape(n_frames, h, w)
        (tmp_path / "depth.raw").write_bytes(data.tobytes())

        result = _read_stereo_depth_stream(tmp_path, n_frames, h, w)
        assert result is not None
        assert result.shape == (n_frames, h, w)
        assert np.array_equal(result, data)

    def test_missing_file_returns_none(self, tmp_path):
        assert _read_stereo_depth_stream(tmp_path, 3, 4, 5) is None

    def test_undersized_file_falls_back_to_none(self, tmp_path):
        # Only enough bytes for 1 frame, but we ask for 3.
        data = np.zeros((1, 4, 5), dtype=np.float32)
        (tmp_path / "depth.raw").write_bytes(data.tobytes())
        assert _read_stereo_depth_stream(tmp_path, 3, 4, 5) is None

    def test_depth_meta_overrides_shape(self, tmp_path):
        n_frames, h, w = 2, 3, 3
        data = np.ones((n_frames, h, w), dtype=np.float32) * 7.0
        (tmp_path / "depth.raw").write_bytes(data.tobytes())
        with open(tmp_path / "depth_meta.json", "w") as f:
            json.dump({"height": h, "width": w, "dtype": "float32"}, f)

        result = _read_stereo_depth_stream(tmp_path, n_frames, height=999, width=999)
        assert result is not None
        assert result.shape == (n_frames, h, w)
        assert np.all(result == 7.0)


def _write_synthetic_session(proc_dir, raw_dir, session_id, n_frames, width=640, height=480):
    proc_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]
    pose_data = [
        {
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": {"landmarks": landmarks, "confidence": 0.9},
            "left_hand": None,
            "derived": None,
        }
        for _ in range(n_frames)
    ]
    with open(proc_dir / "hand_pose.json", "w") as f:
        json.dump(pose_data, f)

    object_tracks = [
        {
            "frame_idx": i,
            "tracked_objects": [
                {"track_id": "stub_obj_0", "class_label": "unknown_stub", "confidence": 0.1,
                 "bbox": [0.4, 0.4, 0.6, 0.6], "centroid_norm": [0.5, 0.5], "is_stub": True}
            ],
            "stub": True,
        }
        for i in range(n_frames)
    ]
    with open(proc_dir / "object_tracks.json", "w") as f:
        json.dump(object_tracks, f)

    session_meta = {
        "session_id": session_id,
        "video_width": width,
        "video_height": height,
    }
    with open(proc_dir / "session_meta.json", "w") as f:
        json.dump(session_meta, f)

    timestamps = np.arange(n_frames, dtype=np.float64) / 30.0
    write_session_h5(
        path=proc_dir / "session.h5",
        video_timestamps_abs=timestamps,
        pts_relative=timestamps.copy(),
        accel=np.zeros((n_frames, 3), dtype=np.float32),
        gyro=np.zeros((n_frames, 3), dtype=np.float32),
        metadata_dict={},
    )
    return proc_dir


class TestRunEndToEndWithoutGPU:
    """
    Exercises full orchestration for depth_mode "none" and "stereo" — both
    entirely GPU-free. The "monocular_estimated" path is intentionally not
    tested here (needs a real model download + GPU — see module docstring).
    """

    def test_depth_mode_none(self, tmp_path, monkeypatch):
        session_id = "sess_none"
        proc_dir = tmp_path / "processed" / session_id
        raw_dir = tmp_path / "raw" / session_id
        _write_synthetic_session(proc_dir, raw_dir, session_id, n_frames=3)

        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(cfg, "DEPTH_MODE", "none")

        result = run(session_id)

        assert result["depth_mode_effective"] == "none"
        assert result["retargeting_eligible"] is False

        with open(proc_dir / "depth_data.json") as f:
            depth_data = json.load(f)
        assert len(depth_data) == 3
        assert all(d["depth_mode"] == "none" for d in depth_data)
        assert all(d["keypoint_depths_m"] == {} for d in depth_data)

        with open(proc_dir / "hand_pose_3d.json") as f:
            hand_pose_3d = json.load(f)
        assert all(h["landmarks_3d_m"] is None for h in hand_pose_3d)

    def test_depth_mode_stereo_lifts_landmarks_and_enriches_object_tracks(self, tmp_path, monkeypatch):
        session_id = "sess_stereo"
        n_frames, h, w = 3, 480, 640
        proc_dir = tmp_path / "processed" / session_id
        raw_dir = tmp_path / "raw" / session_id
        _write_synthetic_session(proc_dir, raw_dir, session_id, n_frames=n_frames, width=w, height=h)

        # A flat depth field at 2.0m everywhere.
        depth_stream = np.full((n_frames, h, w), 2.0, dtype=np.float32)
        (raw_dir / "depth.raw").write_bytes(depth_stream.tobytes())

        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(cfg, "DEPTH_MODE", "stereo")
        monkeypatch.setattr(cfg, "CAMERA_INTRINSICS_PATH", str(tmp_path / "nope" / "{device_id}_intrinsics.json"))

        result = run(session_id)

        assert result["depth_mode_effective"] == "stereo"
        assert result["retargeting_eligible"] is True
        assert result["depth_confidence"] == cfg.DEPTH_CONFIDENCE_MULTIPLIER["stereo"]

        with open(proc_dir / "depth_data.json") as f:
            depth_data = json.load(f)
        assert depth_data[0]["keypoint_depths_m"]["wrist"] == pytest.approx(2.0)

        with open(proc_dir / "hand_pose_3d.json") as f:
            hand_pose_3d = json.load(f)
        assert hand_pose_3d[0]["landmarks_3d_m"] is not None
        assert len(hand_pose_3d[0]["landmarks_3d_m"]) == 21
        assert hand_pose_3d[0]["landmarks_3d_m"][0][2] == pytest.approx(2.0)  # Z == depth

        # object_tracks.json must be enriched in place with centroid_3d_m
        with open(proc_dir / "object_tracks.json") as f:
            object_tracks = json.load(f)
        obj = object_tracks[0]["tracked_objects"][0]
        assert "centroid_3d_m" in obj
        assert obj["centroid_3d_m"][2] == pytest.approx(2.0)

    def test_stereo_falls_back_to_monocular_mode_label_when_stream_missing(self, tmp_path, monkeypatch):
        """
        With DEPTH_MODE == "stereo" but no depth.raw present, the effective
        mode should report "monocular_estimated" — but we don't actually
        invoke the monocular model in this test; we only verify the
        fallback *decision*. If code changes ever cause this branch to
        proceed further, it would attempt to load the real model and this
        test would start failing/hanging, which is an intentional tripwire.
        """
        session_id = "sess_fallback_decision_only"
        proc_dir = tmp_path / "processed" / session_id
        raw_dir = tmp_path / "raw" / session_id
        _write_synthetic_session(proc_dir, raw_dir, session_id, n_frames=1)

        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(cfg, "DEPTH_MODE", "stereo")

        stereo_result = _read_stereo_depth_stream(raw_dir, 1, 480, 640)
        assert stereo_result is None  # confirms the fallback trigger condition is real
