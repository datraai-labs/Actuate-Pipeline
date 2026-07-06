"""
DatraAI Pipeline — Tests for Step 06: Phase Segmentation
Tests _frame_relevant_confidence's pure logic and run()'s per-segment
mean_confidence propagation (v2 addendum §9).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

_spec = importlib.util.spec_from_file_location(
    "phase_segment",
    str(Path(__file__).resolve().parent.parent / "scripts" / "06_phase_segment.py"),
)
_phase_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase_mod)

_frame_relevant_confidence = _phase_mod._frame_relevant_confidence
run = _phase_mod.run


class TestFrameRelevantConfidence:
    def test_averages_active_relevant_primitives(self):
        flags = {"power_grasp": True, "lateral_pinch": False}
        confidences = {"power_grasp": 0.9, "lateral_pinch": 0.1}
        assert _frame_relevant_confidence(flags, confidences, "grasp") == 0.9

    def test_averages_multiple_active_relevant_primitives(self):
        flags = {"power_grasp": True, "lateral_pinch": True}
        confidences = {"power_grasp": 0.8, "lateral_pinch": 0.4}
        assert _frame_relevant_confidence(flags, confidences, "grasp") == pytest.approx(0.6)

    def test_falls_back_to_full_relevant_set_when_none_active(self):
        """Frame's final phase was inherited via smoothing — none of the relevant primitives fired on THIS frame."""
        flags = {"power_grasp": False, "lateral_pinch": False}
        confidences = {"power_grasp": 0.3, "lateral_pinch": 0.7}
        assert _frame_relevant_confidence(flags, confidences, "grasp") == 0.5

    def test_unknown_phase_returns_zero(self):
        assert _frame_relevant_confidence({}, {}, "not_a_real_phase") == 0.0

    def test_active_manipulation_pools_rotation_and_grasp(self):
        flags = {"wrist_pronate": True, "power_grasp": True, "lateral_pinch": False, "wrist_supinate": False, "wrist_flex": False}
        confidences = {"wrist_pronate": 1.0, "power_grasp": 0.6, "lateral_pinch": 0.0, "wrist_supinate": 0.0, "wrist_flex": 0.0}
        assert _frame_relevant_confidence(flags, confidences, "active_manipulation") == 0.8


def _prim_frame(frame_idx, raw_flags, primitive_confidences, timestamp_sec=None):
    fps = 30.0
    return {
        "frame_idx": frame_idx,
        "timestamp_sec": round(frame_idx / fps, 4) if timestamp_sec is None else timestamp_sec,
        "active_primitives": [p for p, v in raw_flags.items() if v],
        "raw_flags": raw_flags,
        "primitive_confidences": primitive_confidences,
    }


_ZERO_FLAGS = {p: False for p in [
    "wrist_pronate", "wrist_supinate", "wrist_flex", "reach_onset", "power_grasp",
    "lateral_pinch", "contact_onset", "contact_release", "finger_curl", "finger_extend",
    "idle", "transport",
]}
_ZERO_CONF = {p: 0.0 for p in _ZERO_FLAGS}


class TestRunMeanConfidencePropagation:
    """
    Two grasp segments with genuinely different underlying detection
    quality must NOT collapse to the same mean_confidence — a marginal
    (low-confidence) grasp segment must score visibly lower than an
    unambiguous (high-confidence) one, propagated all the way from
    primitives.json's per-frame primitive_confidences.
    """

    def test_marginal_segment_scores_lower_than_unambiguous_segment(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_phase_conf"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        primitives = []
        # Frames 0-14: idle (filler, keeps segments apart and above min duration)
        for i in range(0, 15):
            flags = dict(_ZERO_FLAGS); flags["idle"] = True
            conf = dict(_ZERO_CONF); conf["idle"] = 0.9
            primitives.append(_prim_frame(i, flags, conf))
        # Frames 15-29: grasp segment with LOW confidence (marginal detections)
        for i in range(15, 30):
            flags = dict(_ZERO_FLAGS); flags["power_grasp"] = True
            conf = dict(_ZERO_CONF); conf["power_grasp"] = 0.15
            primitives.append(_prim_frame(i, flags, conf))
        # Frames 30-44: idle filler again
        for i in range(30, 45):
            flags = dict(_ZERO_FLAGS); flags["idle"] = True
            conf = dict(_ZERO_CONF); conf["idle"] = 0.9
            primitives.append(_prim_frame(i, flags, conf))
        # Frames 45-59: grasp segment with HIGH confidence (unambiguous detections)
        for i in range(45, 60):
            flags = dict(_ZERO_FLAGS); flags["power_grasp"] = True
            conf = dict(_ZERO_CONF); conf["power_grasp"] = 0.95
            primitives.append(_prim_frame(i, flags, conf))

        with open(proc_dir / "primitives.json", "w") as f:
            json.dump(primitives, f)

        result = run(session_id)
        grasp_segments = [s for s in result["segments"] if s["phase"] == "grasp"]
        assert len(grasp_segments) == 2, f"expected 2 grasp segments, got {len(grasp_segments)}"

        marginal_seg, unambiguous_seg = grasp_segments[0], grasp_segments[1]
        assert marginal_seg["mean_confidence"] == pytest.approx(0.15, abs=1e-6)
        assert unambiguous_seg["mean_confidence"] == pytest.approx(0.95, abs=1e-6)
        assert marginal_seg["mean_confidence"] < unambiguous_seg["mean_confidence"]

    def test_every_segment_has_mean_confidence_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_phase_conf_field"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        primitives = []
        for i in range(0, 20):
            flags = dict(_ZERO_FLAGS); flags["idle"] = True
            conf = dict(_ZERO_CONF); conf["idle"] = 0.5
            primitives.append(_prim_frame(i, flags, conf))
        with open(proc_dir / "primitives.json", "w") as f:
            json.dump(primitives, f)

        result = run(session_id)
        assert all("mean_confidence" in s for s in result["segments"])

    def test_missing_primitive_confidences_field_degrades_gracefully(self, tmp_path, monkeypatch):
        """Older primitives.json without primitive_confidences must not crash run() — falls back to 0.0."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_phase_no_conf"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        primitives = []
        for i in range(0, 20):
            flags = dict(_ZERO_FLAGS); flags["idle"] = True
            primitives.append({
                "frame_idx": i,
                "timestamp_sec": round(i / 30.0, 4),
                "active_primitives": ["idle"],
                "raw_flags": flags,
                # no "primitive_confidences" key at all
            })
        with open(proc_dir / "primitives.json", "w") as f:
            json.dump(primitives, f)

        result = run(session_id)
        assert all(s["mean_confidence"] == 0.0 for s in result["segments"])
