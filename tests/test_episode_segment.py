"""
DatraAI Pipeline — Tests for Step 06b: Multi-Episode Segmentation
Synthetic segment fixtures, no real video/model calls required.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

_spec = importlib.util.spec_from_file_location(
    "episode_segment",
    str(Path(__file__).resolve().parent.parent / "scripts" / "06b_episode_segment.py"),
)
_episode_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_episode_mod)

_split_into_episodes = _episode_mod._split_into_episodes
_build_episode_entry = _episode_mod._build_episode_entry
run = _episode_mod.run


def _seg(phase, start_frame, end_frame, fps=30.0):
    """Build one phases.json-style segment (30fps timing by default)."""
    return {
        "phase": phase,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": round(start_frame / fps, 4),
        "end_sec": round(end_frame / fps, 4),
        "duration_frames": end_frame - start_frame + 1,
    }


class TestSplitIntoEpisodesSingleEpisode:
    """
    The most common real session shape: no idle gap exceeds the threshold,
    so the whole session must be exactly ONE episode — not zero, not
    truncated. This is the exact edge case flagged for explicit
    verification.
    """

    def test_no_qualifying_gap_yields_exactly_one_group_spanning_everything(self):
        # Several idle segments, all well under the 8.0s default threshold —
        # matches the real session_001 shape (longest real gap was 5.9s).
        segments = [
            _seg("grasp", 0, 14),
            _seg("idle", 15, 48),      # 34 frames / 30fps ≈ 1.13s — well under threshold
            _seg("active_manipulation", 49, 99),
            _seg("idle", 100, 250),    # 151 frames ≈ 5.03s — still under 8.0s
            _seg("release", 251, 300),
        ]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)

        assert len(groups) == 1
        assert groups[0] == segments  # every segment included, nothing dropped
        assert groups[0][0]["start_frame"] == 0
        assert groups[0][-1]["end_frame"] == 300

    def test_single_segment_session(self):
        segments = [_seg("grasp", 0, 29)]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)
        assert len(groups) == 1
        assert groups[0] == segments

    def test_empty_segments_returns_empty_no_crash(self):
        assert _split_into_episodes([], gap_threshold_sec=8.0) == []

    def test_run_against_real_session_001_shape_yields_one_episode(self):
        """
        Regression guard using the REAL session_001 phases.json shape: 76
        segments, longest real idle gap 5.9s (well under the 8.0s default).
        Under the true production config, this session must yield exactly
        one episode — not synthetic data, the actual recorded gap durations.
        """
        real_phases_path = (
            Path(__file__).resolve().parent.parent / "processed" / "session_001" / "phases.json"
        )
        if not real_phases_path.exists():
            import pytest
            pytest.skip("processed/session_001/phases.json not present in this checkout")

        with open(real_phases_path) as f:
            real_segments = json.load(f)["segments"]

        groups = _split_into_episodes(real_segments, gap_threshold_sec=cfg.EPISODE_GAP_THRESHOLD_SEC)
        assert len(groups) == 1, (
            f"Expected exactly one episode for session_001 under the real "
            f"{cfg.EPISODE_GAP_THRESHOLD_SEC}s threshold (longest real gap is 5.9s), got {len(groups)}"
        )
        assert groups[0][0]["start_frame"] == 0
        assert groups[0][-1]["end_frame"] == real_segments[-1]["end_frame"]


class TestSplitIntoEpisodesMultipleEpisodes:
    def test_single_qualifying_gap_splits_into_two_episodes(self):
        segments = [
            _seg("grasp", 0, 29),
            _seg("active_manipulation", 30, 89),
            _seg("idle", 90, 389),   # 300 frames = 10.0s > 8.0s threshold -> boundary
            _seg("grasp", 390, 419),
            _seg("release", 420, 449),
        ]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)

        assert len(groups) == 2
        # First episode: everything before the gap.
        assert groups[0][0]["start_frame"] == 0
        assert groups[0][-1]["end_frame"] == 89
        # Second episode: everything after the gap. The gap itself belongs
        # to neither episode.
        assert groups[1][0]["start_frame"] == 390
        assert groups[1][-1]["end_frame"] == 449

    def test_two_qualifying_gaps_split_into_three_episodes(self):
        segments = [
            _seg("grasp", 0, 29),
            _seg("idle", 30, 329),      # 10.0s gap
            _seg("grasp", 330, 359),
            _seg("idle", 360, 660),     # ~10.03s gap
            _seg("grasp", 661, 690),
        ]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)
        assert len(groups) == 3
        assert groups[0][-1]["end_frame"] == 29
        assert groups[1][0]["start_frame"] == 330
        assert groups[1][-1]["end_frame"] == 359
        assert groups[2][0]["start_frame"] == 661

    def test_qualifying_gap_at_very_start_produces_no_leading_empty_episode(self):
        segments = [
            _seg("idle", 0, 299),   # 10.0s gap right at the start
            _seg("grasp", 300, 329),
            _seg("release", 330, 359),
        ]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)
        assert len(groups) == 1
        assert groups[0][0]["start_frame"] == 300

    def test_qualifying_gap_at_very_end_produces_no_trailing_episode(self):
        segments = [
            _seg("grasp", 0, 29),
            _seg("release", 30, 59),
            _seg("idle", 60, 359),   # trailing 10.0s idle — not a new episode
        ]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)
        assert len(groups) == 1
        assert groups[0][-1]["end_frame"] == 59

    def test_boundary_exactly_at_threshold_does_not_split(self):
        # duration == threshold is NOT "> threshold" -> not a qualifying gap.
        segments = [
            _seg("grasp", 0, 29),
            _seg("idle", 30, 269),   # exactly 8.0s (240 frames / 30fps)
            _seg("grasp", 270, 299),
        ]
        groups = _split_into_episodes(segments, gap_threshold_sec=8.0)
        assert len(groups) == 1


class TestBuildEpisodeEntry:
    def test_fields_match_group_boundaries(self):
        group = [_seg("grasp", 100, 129), _seg("release", 130, 159)]
        entry = _build_episode_entry("session_001", 0, group)
        assert entry["episode_id"] == "session_001_ep00"
        assert entry["start_frame"] == 100
        assert entry["end_frame"] == 159
        assert entry["duration_sec"] == round(group[-1]["end_sec"] - group[0]["start_sec"], 4)

    def test_episode_index_formatting(self):
        group = [_seg("grasp", 0, 29)]
        entry = _build_episode_entry("sess", 7, group)
        assert entry["episode_id"] == "sess_ep07"


class TestRunDropsShortFragmentsAndFallsBack:
    def test_short_fragment_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "MIN_EPISODE_DURATION_SEC", 2.0)
        monkeypatch.setattr(cfg, "EPISODE_GAP_THRESHOLD_SEC", 8.0)

        session_id = "sess_short_fragment"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        segments = [
            _seg("grasp", 0, 14),        # 0.5s fragment before a qualifying gap -> dropped
            _seg("idle", 15, 314),       # 10.0s gap
            _seg("grasp", 315, 344),     # second group: 315-434 = 120 frames = 4.0s -> kept
            _seg("release", 345, 434),
        ]
        with open(proc_dir / "phases.json", "w") as f:
            json.dump({"session_id": session_id, "segments": segments}, f)

        result = run(session_id)

        # The 0.5s fragment (frames 0-14) must be dropped; only the
        # second, longer (4.0s) group survives.
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["start_frame"] == 315
        assert result["episodes"][0]["end_frame"] == 434

    def test_all_fragments_short_falls_back_to_whole_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "MIN_EPISODE_DURATION_SEC", 100.0)  # impossibly high
        monkeypatch.setattr(cfg, "EPISODE_GAP_THRESHOLD_SEC", 8.0)

        session_id = "sess_all_short"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        segments = [_seg("grasp", 0, 29), _seg("release", 30, 59)]
        with open(proc_dir / "phases.json", "w") as f:
            json.dump({"session_id": session_id, "segments": segments}, f)

        result = run(session_id)

        # Every candidate group is below MIN_EPISODE_DURATION_SEC — must NOT
        # produce zero episodes; falls back to the whole session as one.
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["start_frame"] == 0
        assert result["episodes"][0]["end_frame"] == 59

    def test_missing_phases_json_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_missing"
        (tmp_path / session_id).mkdir(parents=True)
        try:
            run(session_id)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
