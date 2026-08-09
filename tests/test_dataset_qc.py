"""
DatraAI Pipeline — Tests for utils/dataset_qc.py and scripts/11b_dataset_qc.py
(v2 addendum §11)

SYNTHETIC MULTI-SESSION FIXTURES ONLY — this repo has exactly one real
session (session_001, confirmed off-taxonomy; see
docs/PIPELINE_STATUS.md's business-blocker note), so §11's dedup/balance/
diversity/split logic can only be CORRECTNESS-verified here, not validated
against a real multi-session dataset. Don't read these tests as evidence
the pipeline has been checked against real dataset-scale data — they
prove the arithmetic and control flow are right, nothing more.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib

import config as cfg
from utils.dataset_qc import (
    compute_diversity_summary,
    compute_task_imbalance_ratio,
    dedup_sessions,
    hamming_similarity,
    phash_frame,
    stratified_split,
)

dataset_qc_script = importlib.import_module("scripts.11b_dataset_qc")


class TestPhashFrame:
    def test_identical_frames_produce_identical_hash(self):
        frame = np.random.RandomState(0).randint(0, 255, size=(64, 64)).astype(np.uint8)
        assert phash_frame(frame) == phash_frame(frame.copy())

    def test_solid_black_and_solid_white_differ(self):
        black = np.zeros((64, 64), dtype=np.uint8)
        white = np.full((64, 64), 255, dtype=np.uint8)
        # Degenerate case: a solid-color frame's mean == every pixel, so
        # "> mean" is all-False regardless of color — both hash to 0. This
        # documents that phash_frame needs actual contrast to discriminate,
        # not a claim it distinguishes any two different images.
        assert phash_frame(black) == phash_frame(white)

    def test_distinct_textured_frames_produce_different_hashes(self):
        rng = np.random.RandomState(1)
        frame_a = rng.randint(0, 255, size=(64, 64)).astype(np.uint8)
        frame_b = rng.randint(0, 255, size=(64, 64)).astype(np.uint8)
        assert phash_frame(frame_a) != phash_frame(frame_b)


class TestHammingSimilarity:
    def test_identical_hashes_have_similarity_one(self):
        assert hamming_similarity(0b1010, 0b1010, bits=4) == 1.0

    def test_fully_opposite_hashes_have_similarity_zero(self):
        assert hamming_similarity(0b0000, 0b1111, bits=4) == 0.0

    def test_partial_difference(self):
        assert hamming_similarity(0b1000, 0b0000, bits=4) == 0.75


class TestDedupSessions:
    def test_no_near_duplicates_keeps_all(self):
        session_hashes = [("s1", 0b0000), ("s2", 0b1111)]
        result = dedup_sessions(session_hashes, threshold=0.95, bits=4)
        assert result["kept"] == ["s1", "s2"]
        assert result["removed"] == []

    def test_near_duplicate_flags_later_session_as_removed(self):
        session_hashes = [("s1", 0b0000), ("s2", 0b0000), ("s3", 0b1111)]
        result = dedup_sessions(session_hashes, threshold=0.95, bits=4)
        assert result["kept"] == ["s1", "s3"]
        assert result["removed"] == ["s2"]
        assert result["duplicate_of"]["s2"] == "s1"
        assert result["similarities"]["s2"] == 1.0

    def test_below_threshold_similarity_not_flagged(self):
        # bits=4, differing by 1 bit -> similarity 0.75, below a 0.9 threshold
        session_hashes = [("s1", 0b0000), ("s2", 0b1000)]
        result = dedup_sessions(session_hashes, threshold=0.9, bits=4)
        assert result["kept"] == ["s1", "s2"]
        assert result["removed"] == []

    def test_matches_earliest_kept_session_not_a_later_one(self):
        session_hashes = [("s1", 0b0000), ("s2", 0b1111), ("s3", 0b1111)]
        result = dedup_sessions(session_hashes, threshold=0.95, bits=4)
        assert result["duplicate_of"]["s3"] == "s2"

    def test_empty_input_returns_empty_result(self):
        result = dedup_sessions([], threshold=0.9)
        assert result["kept"] == []
        assert result["removed"] == []


class TestComputeTaskImbalanceRatio:
    def test_single_task_is_undefined_not_a_number(self):
        assert compute_task_imbalance_ratio({"bolt_tightening": 10}) is None

    def test_no_tasks_is_undefined(self):
        assert compute_task_imbalance_ratio({}) is None

    def test_perfectly_balanced_two_tasks_ratio_is_one(self):
        assert compute_task_imbalance_ratio({"bolt_tightening": 5, "box_seal": 5}) == 1.0

    def test_imbalanced_tasks_ratio_reflects_max_over_min(self):
        assert compute_task_imbalance_ratio({"bolt_tightening": 20, "box_seal": 5}) == 4.0

    def test_zero_count_tasks_excluded_from_ratio(self):
        # A task present in the taxonomy but with 0 episodes shouldn't
        # count as the "min" and make the ratio meaninglessly huge.
        assert compute_task_imbalance_ratio({"bolt_tightening": 20, "box_seal": 5, "tool_change": 0}) == 4.0


class TestComputeDiversitySummary:
    def test_counts_and_dedups_object_classes(self):
        summary = compute_diversity_summary({"bolt": 5, "box": 3}, ["worker_1", "worker_1", "worker_2"])
        assert summary["unique_object_classes"] == ["bolt", "box"]
        assert summary["object_class_diversity_count"] == 2
        assert summary["unique_worker_ids"] == ["worker_1", "worker_2"]
        assert summary["worker_diversity_count"] == 2

    def test_none_worker_ids_excluded_not_counted_as_unknown(self):
        summary = compute_diversity_summary({}, [None, None, "worker_1"])
        assert summary["unique_worker_ids"] == ["worker_1"]
        assert summary["worker_diversity_count"] == 1

    def test_no_object_tracks_gives_zero_diversity_not_a_crash(self):
        summary = compute_diversity_summary({}, [])
        assert summary["object_class_diversity_count"] == 0
        assert summary["worker_diversity_count"] == 0


def _episodes(task_counts: dict) -> list:
    """Build synthetic episode dicts: {task: count} -> [{"episode_id":..., "L1_task":...}, ...]."""
    episodes = []
    for task, count in task_counts.items():
        for i in range(count):
            episodes.append({"episode_id": f"{task}_ep{i:03d}", "L1_task": task})
    return episodes


class TestStratifiedSplit:
    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError):
            stratified_split(_episodes({"bolt_tightening": 10}), {"train": 0.5, "val": 0.4}, "L1_task")

    def test_split_sizes_match_ratios_within_rounding(self):
        episodes = _episodes({"bolt_tightening": 100})
        splits = stratified_split(episodes, {"train": 0.8, "val": 0.1, "test": 0.1}, "L1_task")
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_every_episode_assigned_exactly_once(self):
        episodes = _episodes({"bolt_tightening": 37, "box_seal": 13})
        splits = stratified_split(episodes, {"train": 0.8, "val": 0.1, "test": 0.1}, "L1_task")
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert sorted(all_ids) == sorted(e["episode_id"] for e in episodes)
        assert len(all_ids) == len(set(all_ids))

    def test_stratifies_proportionally_per_task_not_globally(self):
        """
        A rare task's episodes must still appear in every split
        proportionally, not all land in one split because the rare task
        happened to sort after a big common task in a global (non-
        stratified) split.
        """
        episodes = _episodes({"bolt_tightening": 90, "tool_change": 10})
        splits = stratified_split(episodes, {"train": 0.8, "val": 0.1, "test": 0.1}, "L1_task")
        for split_name in ("train", "val", "test"):
            tool_change_in_split = [e for e in splits[split_name] if e.startswith("tool_change")]
            assert len(tool_change_in_split) > 0, f"tool_change missing entirely from {split_name}"

    def test_deterministic_across_repeated_calls(self):
        episodes = _episodes({"bolt_tightening": 23, "box_seal": 17})
        splits_a = stratified_split(episodes, {"train": 0.8, "val": 0.1, "test": 0.1}, "L1_task")
        splits_b = stratified_split(episodes, {"train": 0.8, "val": 0.1, "test": 0.1}, "L1_task")
        assert splits_a == splits_b

    def test_missing_stratify_key_falls_back_to_unknown_bucket(self):
        episodes = [{"episode_id": "ep1"}, {"episode_id": "ep2"}]
        splits = stratified_split(episodes, {"train": 1.0}, "L1_task")
        assert sorted(splits["train"]) == ["ep1", "ep2"]


class TestRunEndToEnd:
    """
    Synthetic multi-session batch, exercising scripts/11b_dataset_qc.py's
    run() against a fabricated delivery/ + processed/ tree — this is the
    "multi-session" scenario that doesn't exist in real data yet.
    """

    def _write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _write_solid_color_video(self, path, color_value, size=(32, 32), frame_count=5):
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10, size)
        frame = np.full((size[1], size[0], 3), color_value, dtype=np.uint8)
        for _ in range(frame_count):
            writer.write(frame)
        writer.release()

    def test_run_produces_report_and_extends_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "DELIVERY_DIR", tmp_path / "delivery")
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "DEDUP_SIMILARITY_THRESHOLD", 0.90)
        monkeypatch.setattr(cfg, "TRAIN_VAL_TEST_SPLIT", {"train": 0.8, "val": 0.1, "test": 0.1})
        monkeypatch.setattr(cfg, "SPLIT_STRATIFY_BY", "L1_task")
        monkeypatch.setattr(dataset_qc_script, "cfg", cfg)

        batch_id = "batch_synthetic_test"
        delivery_dir = cfg.DELIVERY_DIR / batch_id

        sessions = {
            "session_a": {"color": 30, "task_counts": {"bolt_tightening": 10}},
            "session_b": {"color": 220, "task_counts": {"box_seal": 10}},
        }

        session_manifests = []
        for session_id, spec in sessions.items():
            self._write_solid_color_video(
                delivery_dir / "sessions" / session_id / "compressed.mp4", spec["color"]
            )
            episodes = _episodes(spec["task_counts"])
            session_manifests.append({
                "session_id": session_id,
                "episode_count": len(episodes),
                "episodes": episodes,
            })
            self._write_json(
                cfg.PROCESSED_DIR / session_id / "session_meta.json",
                {"worker_id": f"worker_{session_id}"},
            )
            self._write_json(
                cfg.PROCESSED_DIR / session_id / "object_tracks.json",
                [{"frame_idx": 0, "tracked_objects": [{"class_label": "bolt"}]}],
            )

        task_distribution = {}
        for spec in sessions.values():
            for task, count in spec["task_counts"].items():
                task_distribution[task] = task_distribution.get(task, 0) + count

        self._write_json(delivery_dir / "dataset_manifest.json", {
            "batch_id": batch_id,
            "sessions": session_manifests,
            "task_distribution": task_distribution,
        })

        report = dataset_qc_script.run(batch_id)

        assert report["session_count"] == 2
        assert report["task_imbalance_ratio"] == 1.0
        assert report["diversity_summary"]["worker_diversity_count"] == 2
        assert sum(report["splits"].values()) == 20

        with open(delivery_dir / "dataset_manifest.json", encoding="utf-8") as f:
            updated_manifest = json.load(f)
        assert "dedup_removed_count" in updated_manifest
        assert "task_imbalance_ratio" in updated_manifest
        assert "diversity_summary" in updated_manifest
        assert "splits" in updated_manifest

        assert (delivery_dir / "dataset_qc_report.json").exists()

    def test_missing_manifest_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "DELIVERY_DIR", tmp_path / "delivery")
        monkeypatch.setattr(dataset_qc_script, "cfg", cfg)
        with pytest.raises(FileNotFoundError):
            dataset_qc_script.run("nonexistent_batch")

    def test_single_session_batch_sets_caveat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "DELIVERY_DIR", tmp_path / "delivery")
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "TRAIN_VAL_TEST_SPLIT", {"train": 1.0})
        monkeypatch.setattr(cfg, "SPLIT_STRATIFY_BY", "L1_task")
        monkeypatch.setattr(dataset_qc_script, "cfg", cfg)

        batch_id = "batch_single_session"
        delivery_dir = cfg.DELIVERY_DIR / batch_id
        self._write_solid_color_video(delivery_dir / "sessions" / "only_session" / "compressed.mp4", 100)
        self._write_json(delivery_dir / "dataset_manifest.json", {
            "batch_id": batch_id,
            "sessions": [{
                "session_id": "only_session",
                "episode_count": 1,
                "episodes": [{"episode_id": "ep1", "L1_task": "unknown"}],
            }],
            "task_distribution": {"unknown": 1},
        })

        report = dataset_qc_script.run(batch_id)

        assert report["single_real_session_caveat"] is not None
        assert report["task_imbalance_ratio"] is None
