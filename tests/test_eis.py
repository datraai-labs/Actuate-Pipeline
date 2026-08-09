"""
DatraAI Pipeline — Tests for Step 10: EIS (Episode Integrity Score)
Tests EIS computation with synthetic component scores, and run()'s
per-episode orchestration (v2 addendum §6).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import compute_eis from 10_eis.py via importlib
import importlib.util

import config as cfg
from utils.hdf5_writer import write_session_h5

_spec = importlib.util.spec_from_file_location(
    "eis",
    str(Path(__file__).resolve().parent.parent / "scripts" / "10_eis.py"),
)
_eis_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eis_mod)

compute_eis = _eis_mod.compute_eis
run = _eis_mod.run
_compute_label_confidence = _eis_mod._compute_label_confidence


class TestEISPerfectSession:
    """Test EIS with perfect component scores."""

    def test_eis_perfect_session(self):
        """All components at maximum → EIS = 100."""
        result = compute_eis(
            max_drift_ms=0.0,        # perfect sync
            blur_score=200.0,        # max normalized blur
            hand_presence_rate=1.0,  # 100% hand presence
            label_confidence=1.0,    # perfect confidence
            causal_passed=True,      # no causal issues
        )

        assert result["eis"] == 100, f"Expected EIS=100, got {result['eis']}"

    def test_eis_near_perfect(self):
        """Slightly imperfect session should score high but < 100."""
        result = compute_eis(
            max_drift_ms=0.5,
            blur_score=180.0,
            hand_presence_rate=0.95,
            label_confidence=0.92,
            causal_passed=True,
        )

        assert 85 <= result["eis"] <= 99, f"Expected EIS 85-99, got {result['eis']}"


class TestEISQuarantineThreshold:
    """Test EIS quarantine threshold behavior."""

    def test_eis_low_scores_quarantine(self):
        """Very poor scores → EIS < 70."""
        result = compute_eis(
            max_drift_ms=4.5,        # bad sync
            blur_score=30.0,         # very blurry
            hand_presence_rate=0.2,  # low hand presence
            label_confidence=0.3,    # low confidence
            causal_passed=False,     # causal failures
            causal_inversion_count=8,
            causal_total_events=10,
        )

        assert result["eis"] < 70, f"Expected EIS < 70 (quarantine), got {result['eis']}"

    def test_eis_zero_everything(self):
        """All zeros → EIS = 0."""
        result = compute_eis(
            max_drift_ms=10.0,       # drift at max penalty
            blur_score=0.0,          # no blur score
            hand_presence_rate=0.0,  # no hands
            label_confidence=0.0,    # no confidence
            causal_passed=False,
            causal_inversion_count=10,
            causal_total_events=10,
        )

        assert result["eis"] == 0, f"Expected EIS=0, got {result['eis']}"


class TestEISComponentWeights:
    """Test that EIS components are weighted correctly."""

    def test_eis_weights_sum_to_one(self):
        """Verify that EIS weights sum to 1.0."""
        total = (
            cfg.SYNC_WEIGHT
            + cfg.BLUR_WEIGHT
            + cfg.HAND_PRESENCE_WEIGHT
            + cfg.LABEL_CONFIDENCE_WEIGHT
            + cfg.CAUSAL_CHECK_WEIGHT
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_eis_sync_weight_dominates(self):
        """Sync weight (0.30) should be the largest single weight."""
        assert cfg.SYNC_WEIGHT >= cfg.BLUR_WEIGHT
        assert cfg.SYNC_WEIGHT >= cfg.HAND_PRESENCE_WEIGHT
        assert cfg.SYNC_WEIGHT >= cfg.LABEL_CONFIDENCE_WEIGHT
        assert cfg.SYNC_WEIGHT >= cfg.CAUSAL_CHECK_WEIGHT

    def test_eis_only_sync_bad(self):
        """Bad sync but everything else perfect → EIS = 70."""
        result = compute_eis(
            max_drift_ms=5.0,        # score = 0
            blur_score=200.0,        # score = 1.0
            hand_presence_rate=1.0,  # score = 1.0
            label_confidence=1.0,    # score = 1.0
            causal_passed=True,      # score = 1.0
        )

        # Expected: 0*0.30 + 1*0.20 + 1*0.20 + 1*0.15 + 1*0.15 = 0.70 → 70
        assert result["eis"] == 70, f"Expected EIS=70, got {result['eis']}"

    def test_eis_only_blur_bad(self):
        """Bad blur but everything else perfect → EIS = 80."""
        result = compute_eis(
            max_drift_ms=0.0,        # score = 1.0
            blur_score=0.0,          # score = 0
            hand_presence_rate=1.0,  # score = 1.0
            label_confidence=1.0,    # score = 1.0
            causal_passed=True,      # score = 1.0
        )

        # Expected: 1*0.30 + 0*0.20 + 1*0.20 + 1*0.15 + 1*0.15 = 0.80 → 80
        assert result["eis"] == 80, f"Expected EIS=80, got {result['eis']}"

    def test_eis_verified_formula(self):
        """Manually verify the weighted sum formula."""
        result = compute_eis(
            max_drift_ms=1.0,        # score = 1 - 1/5 = 0.8
            blur_score=100.0,        # score = 100/200 = 0.5
            hand_presence_rate=0.7,  # score = 0.7
            label_confidence=0.6,    # score = 0.6
            causal_passed=True,      # score = 1.0
        )

        expected_raw = (
            0.8 * 0.30    # sync
            + 0.5 * 0.20  # blur
            + 0.7 * 0.20  # hand
            + 0.6 * 0.15  # label
            + 1.0 * 0.15  # causal
        )
        expected_eis = round(expected_raw * 100)

        assert result["eis"] == expected_eis, (
            f"Expected EIS={expected_eis}, got {result['eis']}. "
            f"Raw: expected={expected_raw}, got={result['eis_raw']}"
        )

    def test_eis_components_structure(self):
        """Verify the output structure contains all expected components."""
        result = compute_eis(
            max_drift_ms=1.0,
            blur_score=100.0,
            hand_presence_rate=0.8,
            label_confidence=0.7,
            causal_passed=True,
        )

        assert "eis" in result
        assert "components" in result
        components = result["components"]
        assert "sync_drift" in components
        assert "blur" in components
        assert "hand_presence" in components
        assert "label_confidence" in components
        assert "causal_check" in components

        for comp_name, comp in components.items():
            assert "score" in comp, f"Component {comp_name} missing 'score'"
            assert "weight" in comp, f"Component {comp_name} missing 'weight'"
            assert 0.0 <= comp["score"] <= 1.0, (
                f"Component {comp_name} score {comp['score']} out of [0,1] range"
            )

    def test_eis_causal_partial_failure(self):
        """Partial causal failures should reduce score proportionally."""
        # 3 out of 10 inversions → score = 1 - 3/10 = 0.7
        result = compute_eis(
            max_drift_ms=0.0,
            blur_score=200.0,
            hand_presence_rate=1.0,
            label_confidence=1.0,
            causal_passed=False,
            causal_inversion_count=3,
            causal_total_events=10,
        )

        causal_score = result["components"]["causal_check"]["score"]
        assert abs(causal_score - 0.7) < 0.01, f"Expected causal score 0.7, got {causal_score}"

    def test_eis_bounded(self):
        """EIS should always be in [0, 100]."""
        # Test with extreme values
        for drift in [0, 1, 5, 10, 100]:
            for blur in [0, 50, 200, 500]:
                for hp in [0.0, 0.5, 1.0]:
                    result = compute_eis(
                        max_drift_ms=drift,
                        blur_score=blur,
                        hand_presence_rate=hp,
                        label_confidence=0.5,
                        causal_passed=True,
                    )
                    assert 0 <= result["eis"] <= 100, (
                        f"EIS {result['eis']} out of bounds for "
                        f"drift={drift}, blur={blur}, hp={hp}"
                    )


class TestRunPerEpisodeOrchestration:
    """
    A two-episode session where episode 0 has 100% hand presence / high
    label confidence and episode 1 has 0% hand presence / low label
    confidence, but sync/blur/causal (session-wide, not restructured by
    §6) are shared and identical for both. Confirms hand_presence_rate and
    label_confidence are genuinely computed per-episode (not a session
    average leaking into both), while the shared components stay constant.
    """

    def _write_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    def _write_session(self, proc_dir, session_id, monkeypatch, tmp_path, max_drift_ms=1.0):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)

        import numpy as np

        n = 200
        video_ts = np.arange(n, dtype=np.float64) / 30.0
        write_session_h5(
            proc_dir / "session.h5",
            video_timestamps_abs=video_ts,
            pts_relative=video_ts,
            accel=np.zeros((n, 3), dtype=np.float32),
            gyro=np.zeros((n, 3), dtype=np.float32),
            metadata_dict={"sync_stats": {"max_drift_ms": max_drift_ms}},
        )

        self._write_json(proc_dir / "qa_report.json", {
            "overall_passed": True,
            "checks": {"blur": {"score": 150.0}},
        })
        self._write_json(proc_dir / "validation_report.json", {
            "overall_valid": True,
            "checks": {"causal_ordering": {"passed": True, "inversion_count": 0, "total_events": 1}},
        })

    def test_hand_presence_and_label_confidence_are_per_episode(self, tmp_path, monkeypatch):
        session_id = "sess_eis_multi_ep"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)
        self._write_session(proc_dir, session_id, monkeypatch, tmp_path)

        episodes = {
            "session_id": session_id,
            "episodes": [
                {"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": 49,
                 "start_sec": 0.0, "end_sec": 1.6667, "duration_sec": 1.6667},
                {"episode_id": f"{session_id}_ep01", "start_frame": 100, "end_frame": 149,
                 "start_sec": 3.3333, "end_sec": 5.0, "duration_sec": 1.6667},
            ],
        }
        task_label = {
            "session_id": session_id,
            "episodes": [
                {"episode_id": f"{session_id}_ep00", "L1_task": "bolt_tightening", "confidence": 0.95},
                {"episode_id": f"{session_id}_ep01", "L1_task": "unknown", "confidence": 0.1},
            ],
        }
        # Episode 0: hands detected on every frame in [0,49].
        # Episode 1: hands detected on NONE of the frames in [100,149].
        hand_pose = (
            [{"frame_idx": i, "hands_detected": True} for i in range(50)]
            + [{"frame_idx": i, "hands_detected": False} for i in range(100, 150)]
        )

        self._write_json(proc_dir / "episodes.json", episodes)
        self._write_json(proc_dir / "task_label.json", task_label)
        self._write_json(proc_dir / "hand_pose.json", hand_pose)

        result = run(session_id)

        assert len(result["episodes"]) == 2
        ep0, ep1 = result["episodes"]

        assert ep0["episode_id"] == f"{session_id}_ep00"
        assert ep0["components"]["hand_presence"]["rate"] == 1.0
        assert ep0["components"]["label_confidence"]["score"] == 0.95

        assert ep1["episode_id"] == f"{session_id}_ep01"
        assert ep1["components"]["hand_presence"]["rate"] == 0.0
        assert ep1["components"]["label_confidence"]["score"] == 0.1

        # Shared session-wide components (sync/blur/causal) must be
        # IDENTICAL across both episodes since those upstream checks were
        # not restructured per-episode by §6.
        assert ep0["components"]["sync_drift"] == ep1["components"]["sync_drift"]
        assert ep0["components"]["blur"] == ep1["components"]["blur"]
        assert ep0["components"]["causal_check"] == ep1["components"]["causal_check"]

        # Episode 1's near-zero hand presence + low confidence must drag
        # its EIS below episode 0's — proving the per-episode components
        # actually feed into a genuinely different score, not a shared one.
        assert ep1["EIS"] < ep0["EIS"]

        assert result["session_mean_EIS"] == round((ep0["EIS"] + ep1["EIS"]) / 2, 1)

    def test_missing_task_label_entry_defaults_confidence_to_zero(self, tmp_path, monkeypatch):
        """An episode with no matching task_label.json entry must fail closed (0.0 label confidence), not crash."""
        session_id = "sess_eis_no_task_entry"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)
        self._write_session(proc_dir, session_id, monkeypatch, tmp_path)

        episodes = {
            "session_id": session_id,
            "episodes": [
                {"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": 29,
                 "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            ],
        }
        # task_label.json has NO episodes at all (simulates an upstream gap).
        task_label = {"session_id": session_id, "episodes": []}

        self._write_json(proc_dir / "episodes.json", episodes)
        self._write_json(proc_dir / "task_label.json", task_label)
        self._write_json(proc_dir / "hand_pose.json", [])

        result = run(session_id)
        assert result["episodes"][0]["components"]["label_confidence"]["score"] == 0.0


class TestComputeLabelConfidencePureFunction:
    """v2 addendum §9 — label_confidence is now a real weighted average across task + segment confidences, not the bare task-classification score."""

    def test_no_segments_falls_back_to_task_confidence(self):
        assert _compute_label_confidence(0.85, []) == 0.85

    def test_weighted_average_matches_frame_count_weighting(self):
        task_confidence = 1.0
        segments = [
            {"start_frame": 0, "end_frame": 9, "mean_confidence": 0.0},   # 10 frames
            {"start_frame": 10, "end_frame": 29, "mean_confidence": 1.0},  # 20 frames
        ]
        # total_seg_frames = 30 -> task entry weighted 30, segments weighted
        # 10 and 20 respectively. weighted_sum = 1.0*30 + 0.0*10 + 1.0*20 = 50
        # weight_total = 30+10+20 = 60 -> 50/60 = 0.8333
        result = _compute_label_confidence(task_confidence, segments)
        assert result == pytest.approx(0.8333, abs=1e-4)

    def test_low_segment_confidence_pulls_down_high_task_confidence(self):
        """A confident task classification riding on genuinely marginal phase segmentation must show a lower blended score than the bare task confidence."""
        high_task_confidence = 0.95
        marginal_segments = [{"start_frame": 0, "end_frame": 99, "mean_confidence": 0.1}]
        result = _compute_label_confidence(high_task_confidence, marginal_segments)
        assert result < high_task_confidence

    def test_missing_mean_confidence_field_defaults_to_zero(self):
        segments = [{"start_frame": 0, "end_frame": 9}]  # no mean_confidence key at all
        result = _compute_label_confidence(1.0, segments)
        # total_seg_frames=10, weighted_sum = 1.0*10 + 0.0*10 = 10, weight_total=20 -> 0.5
        assert result == pytest.approx(0.5)


class TestRunLabelConfidenceIntegration:
    def _base_session(self, tmp_path, monkeypatch, session_id):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        import numpy as np
        n = 60
        video_ts = np.arange(n, dtype=np.float64) / 30.0
        write_session_h5(
            proc_dir / "session.h5",
            video_timestamps_abs=video_ts,
            pts_relative=video_ts,
            accel=np.zeros((n, 3), dtype=np.float32),
            gyro=np.zeros((n, 3), dtype=np.float32),
            metadata_dict={"sync_stats": {"max_drift_ms": 0.0}},
        )
        with open(proc_dir / "qa_report.json", "w") as f:
            json.dump({"overall_passed": True, "checks": {"blur": {"score": 150.0}}}, f)
        with open(proc_dir / "validation_report.json", "w") as f:
            json.dump({"overall_valid": True, "checks": {"causal_ordering": {"passed": True, "inversion_count": 0, "total_events": 1}}}, f)
        with open(proc_dir / "hand_pose.json", "w") as f:
            json.dump([{"frame_idx": i, "hands_detected": True} for i in range(n)], f)
        with open(proc_dir / "episodes.json", "w") as f:
            json.dump({
                "session_id": session_id,
                "episodes": [{"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": n - 1,
                              "start_sec": 0.0, "end_sec": (n - 1) / 30.0, "duration_sec": (n - 1) / 30.0}],
            }, f)
        return proc_dir

    def test_label_confidence_reflects_segment_mean_confidence_not_just_task_score(self, tmp_path, monkeypatch):
        session_id = "sess_eis_label_conf"
        proc_dir = self._base_session(tmp_path, monkeypatch, session_id)

        with open(proc_dir / "task_label.json", "w") as f:
            json.dump({"session_id": session_id, "episodes": [
                {"episode_id": f"{session_id}_ep00", "L1_task": "bolt_tightening", "confidence": 1.0},
            ]}, f)
        # A perfect task-confidence score, but the phase segmentation
        # underneath it is entirely low-confidence — the blended
        # label_confidence must be visibly pulled down from 1.0, not just
        # echo the bare task-classification score.
        with open(proc_dir / "phases.json", "w") as f:
            json.dump({"session_id": session_id, "segments": [
                {"phase": "grasp", "start_frame": 0, "end_frame": 59, "mean_confidence": 0.1},
            ]}, f)

        result = run(session_id)
        label_conf = result["episodes"][0]["components"]["label_confidence"]["score"]
        assert label_conf < 1.0
        assert label_conf == pytest.approx(0.55, abs=1e-3)  # (1.0*60 + 0.1*60) / 120

    def test_missing_phases_json_falls_back_to_bare_task_confidence(self, tmp_path, monkeypatch):
        session_id = "sess_eis_label_conf_no_phases"
        proc_dir = self._base_session(tmp_path, monkeypatch, session_id)

        with open(proc_dir / "task_label.json", "w") as f:
            json.dump({"session_id": session_id, "episodes": [
                {"episode_id": f"{session_id}_ep00", "L1_task": "bolt_tightening", "confidence": 0.77},
            ]}, f)
        # No phases.json written at all.

        result = run(session_id)
        label_conf = result["episodes"][0]["components"]["label_confidence"]["score"]
        assert label_conf == pytest.approx(0.77)
