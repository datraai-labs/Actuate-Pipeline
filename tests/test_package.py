"""
DatraAI Pipeline — Tests for Step 11: Package
Tests _assemble_action_labels' per-episode nesting (v2 addendum §6) and
run()'s episode-level manifest counting across a batch.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

_spec = importlib.util.spec_from_file_location(
    "package",
    str(Path(__file__).resolve().parent.parent / "scripts" / "11_package.py"),
)
_pkg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pkg_mod)

_assemble_action_labels = _pkg_mod._assemble_action_labels
run = _pkg_mod.run


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _make_one_episode_session(proc_dir, session_id, task="bolt_tightening", confidence=0.9, eis=80):
    proc_dir.mkdir(parents=True, exist_ok=True)
    episodes = {
        "session_id": session_id,
        "episodes": [
            {"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": 49,
             "start_sec": 0.0, "end_sec": 1.6667, "duration_sec": 1.6667},
        ],
    }
    _write_json(proc_dir / "episodes.json", episodes)
    _write_json(proc_dir / "task_label.json", {
        "session_id": session_id,
        "episodes": [{"episode_id": f"{session_id}_ep00", "L1_task": task, "confidence": confidence}],
    })
    _write_json(proc_dir / "phases.json", {
        "session_id": session_id,
        "segments": [
            {"phase": "grasp", "start_frame": 0, "end_frame": 24, "start_sec": 0.0, "end_sec": 0.8333, "mean_confidence": 0.82},
            {"phase": "release", "start_frame": 25, "end_frame": 49, "start_sec": 0.8333, "end_sec": 1.6667, "mean_confidence": 0.44},
        ],
    })
    _write_json(proc_dir / "primitives.json", [
        {"frame_idx": i, "active_primitives": ["power_grasp"]} for i in range(50)
    ])
    _write_json(proc_dir / "validation_report.json", {"overall_valid": True})
    _write_json(proc_dir / "quality_certificate.json", {
        "session_id": session_id,
        "episodes": [{"episode_id": f"{session_id}_ep00", "EIS": eis, "flags": []}],
        "session_mean_EIS": eis,
    })
    _write_json(proc_dir / "session_meta.json", {"duration_seconds": 1.6667})
    _write_json(proc_dir / "qa_report.json", {"overall_passed": True})
    (proc_dir / "redacted_compressed.mp4").write_bytes(b"fake video bytes")
    (proc_dir / "session.h5").write_bytes(b"fake h5 bytes")
    _write_json(proc_dir / "hand_pose.json", [])
    _write_json(proc_dir / "language_grounding.json", {"session_id": session_id, "episodes": []})
    _write_json(proc_dir / "privacy_report.json", {})
    return proc_dir


class TestAssembleActionLabels:
    """
    Regression coverage for the bug found via real-data verification: the
    per-episode "L1_task" field must be the plain task-label STRING (matching
    every other script in the pipeline — 07_task_classify.py, 08_validate.py,
    09_language_ground.py all treat it as a string), not the entire
    classification dict from task_label.json.
    """

    def test_l1_task_is_a_plain_string_not_the_classification_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_pkg_one_ep"
        proc_dir = _make_one_episode_session(tmp_path / session_id, session_id, task="bolt_tightening", confidence=0.87)

        result = _assemble_action_labels(proc_dir, session_id)

        ep = result["episodes"][0]
        assert ep["L1_task"] == "bolt_tightening"
        assert isinstance(ep["L1_task"], str)
        assert ep["L1_task_confidence"] == 0.87

    def test_episode_nesting_shape_and_filtered_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_pkg_shape"
        proc_dir = _make_one_episode_session(tmp_path / session_id, session_id, eis=73)

        result = _assemble_action_labels(proc_dir, session_id)

        assert result["session_id"] == session_id
        assert len(result["episodes"]) == 1
        ep = result["episodes"][0]
        assert ep["episode_id"] == f"{session_id}_ep00"
        assert ep["start_frame"] == 0
        assert ep["end_frame"] == 49
        assert len(ep["L2_phases"]) == 2  # both phase segments overlap the episode
        assert len(ep["L3_primitives"]) == 50  # all 50 primitive frames fall inside
        assert ep["quality"] == {"EIS": 73, "flags": []}
        assert result["session_mean_EIS"] == 73

    def test_missing_task_label_entry_defaults_to_unknown_string(self, tmp_path, monkeypatch):
        """An episode absent from task_label.json must fail closed to the string "unknown", not an empty dict."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_pkg_missing_task"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)
        _write_json(proc_dir / "episodes.json", {
            "session_id": session_id,
            "episodes": [{"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": 9,
                          "start_sec": 0.0, "end_sec": 0.3, "duration_sec": 0.3}],
        })
        _write_json(proc_dir / "task_label.json", {"session_id": session_id, "episodes": []})

        result = _assemble_action_labels(proc_dir, session_id)
        ep = result["episodes"][0]
        assert ep["L1_task"] == "unknown"
        assert ep["L1_task_confidence"] == 0.0


class TestConfidenceTree:
    """v2 addendum §9 — action_labels.json exposes a nested confidence_tree per episode."""

    def test_confidence_tree_shape_and_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_pkg_conf_tree"
        proc_dir = _make_one_episode_session(tmp_path / session_id, session_id, confidence=0.83)

        result = _assemble_action_labels(proc_dir, session_id)
        ep = result["episodes"][0]

        assert "confidence_tree" in ep
        tree = ep["confidence_tree"]
        assert tree["task_level_confidence"] == 0.83
        assert tree["frame_level_available"] is True

        segs = tree["segment_level_confidences"]
        assert len(segs) == 2
        assert segs[0] == {"phase": "grasp", "start_frame": 0, "end_frame": 24, "mean_confidence": 0.82}
        assert segs[1] == {"phase": "release", "start_frame": 25, "end_frame": 49, "mean_confidence": 0.44}

    def test_marginal_and_unambiguous_segments_are_not_flattened_to_the_same_value(self, tmp_path, monkeypatch):
        """The whole point of §9: segment_level_confidences must show real variance, not a constant placeholder."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_pkg_conf_variance"
        proc_dir = _make_one_episode_session(tmp_path / session_id, session_id)

        result = _assemble_action_labels(proc_dir, session_id)
        segs = result["episodes"][0]["confidence_tree"]["segment_level_confidences"]
        values = [s["mean_confidence"] for s in segs]
        assert len(set(values)) > 1, f"expected varying confidences, got {values}"

    def test_frame_level_primitive_confidences_survive_into_l3_primitives(self, tmp_path, monkeypatch):
        """frame_level_available: true is only honest if L3_primitives entries actually carry primitive_confidences."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_pkg_frame_level"
        proc_dir = _make_one_episode_session(tmp_path / session_id, session_id)
        # Overwrite primitives.json with entries that DO carry primitive_confidences (as 05_primitives.py now writes).
        _write_json(proc_dir / "primitives.json", [
            {"frame_idx": i, "active_primitives": ["power_grasp"], "primitive_confidences": {"power_grasp": 0.5}}
            for i in range(50)
        ])

        result = _assemble_action_labels(proc_dir, session_id)
        ep = result["episodes"][0]
        assert ep["confidence_tree"]["frame_level_available"] is True
        assert all("primitive_confidences" in p for p in ep["L3_primitives"])


class TestRunBatchManifest:
    def test_episode_count_and_task_distribution_summed_across_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "DELIVERY_DIR", tmp_path / "delivery")

        _make_one_episode_session(tmp_path / "processed" / "sess_a", "sess_a", task="bolt_tightening", eis=90)
        _make_one_episode_session(tmp_path / "processed" / "sess_b", "sess_b", task="bolt_tightening", eis=60)

        result = run(["sess_a", "sess_b"], batch_id="test_batch")

        assert result["session_count"] == 2
        assert result["episode_count"] == 2
        assert result["task_distribution"] == {"bolt_tightening": 2}
        assert result["mean_EIS"] == 75.0  # (90 + 60) / 2

    def test_session_without_redacted_video_is_skipped_not_delivered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "DELIVERY_DIR", tmp_path / "delivery")

        proc_dir = _make_one_episode_session(tmp_path / "processed" / "sess_unredacted", "sess_unredacted")
        (proc_dir / "redacted_compressed.mp4").unlink()  # simulate 03b never ran

        result = run(["sess_unredacted"], batch_id="test_batch_skip")

        assert result["session_count"] == 0
        assert result["episode_count"] == 0
        assert not (tmp_path / "delivery" / "test_batch_skip" / "sessions" / "sess_unredacted").exists()

    def test_missing_processed_dir_is_skipped_not_delivered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "DELIVERY_DIR", tmp_path / "delivery")
        (tmp_path / "processed").mkdir(parents=True)

        result = run(["sess_nonexistent"], batch_id="test_batch_missing")

        assert result["session_count"] == 0
        assert result["sessions"] == []
