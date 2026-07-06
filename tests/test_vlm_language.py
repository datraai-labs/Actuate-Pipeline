"""
DatraAI Pipeline — Tests for utils/vlm_language.py (v2 addendum §7, revised)
Pure-function tests only — no real API calls (generate_instruction_vlm /
check_instruction_hallucination take an injected client, exercised via a
fake in tests/test_language_ground.py's VLM-orchestration tests instead).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.vlm_language import (
    sample_representative_frames,
    estimate_cost_usd,
    should_spotcheck,
    check_task_disagreement,
)


def _seg(phase, start_frame, end_frame):
    return {"phase": phase, "start_frame": start_frame, "end_frame": end_frame}


class TestSampleRepresentativeFrames:
    def test_includes_start_and_end(self):
        episode = {"start_frame": 0, "end_frame": 299}
        segments = [_seg("grasp", 0, 149), _seg("release", 150, 299)]
        frames = sample_representative_frames(episode, segments)
        assert frames[0] == 0
        assert frames[-1] == 299

    def test_spans_key_phase_transitions(self):
        """Interior frames must be drawn from reach->grasp->manipulate->release, not arbitrary timestamps."""
        episode = {"start_frame": 0, "end_frame": 299}
        segments = [
            _seg("reach", 0, 29),
            _seg("grasp", 30, 89),
            _seg("active_manipulation", 90, 249),
            _seg("release", 250, 299),
        ]
        frames = sample_representative_frames(episode, segments)
        # Every interior frame should fall inside a key-phase segment's range.
        key_ranges = [(30, 89), (90, 249), (250, 299)]
        for f in frames[1:-1]:
            assert any(lo <= f <= hi for lo, hi in key_ranges), f"frame {f} not inside any key-phase segment"

    def test_frame_count_within_bounds(self):
        episode = {"start_frame": 0, "end_frame": 999}
        segments = [
            _seg("reach", 0, 99),
            _seg("grasp", 100, 199),
            _seg("active_manipulation", 200, 299),
            _seg("release", 300, 399),
            _seg("idle", 400, 999),
        ]
        frames = sample_representative_frames(episode, segments)
        assert cfg.VLM_MIN_SAMPLE_FRAMES <= len(frames) or len(frames) == len(set([episode["start_frame"], episode["end_frame"]]))
        assert len(frames) <= cfg.VLM_MAX_SAMPLE_FRAMES

    def test_no_key_phase_segments_falls_back_to_evenly_spaced(self):
        """An episode with no grasp/active_manipulation/release segments (e.g. all idle/reach) still gets multiple frames."""
        episode = {"start_frame": 0, "end_frame": 299}
        segments = [_seg("reach", 0, 149), _seg("idle", 150, 299)]
        frames = sample_representative_frames(episode, segments)
        assert len(frames) >= cfg.VLM_MIN_SAMPLE_FRAMES
        assert frames[0] == 0
        assert frames[-1] == 299

    def test_degenerate_single_frame_episode(self):
        episode = {"start_frame": 5, "end_frame": 5}
        assert sample_representative_frames(episode, []) == [5]

    def test_frames_are_sorted_and_unique(self):
        episode = {"start_frame": 0, "end_frame": 500}
        segments = [
            _seg("grasp", 0, 100),
            _seg("active_manipulation", 100, 200),
            _seg("active_manipulation", 200, 300),
            _seg("release", 300, 500),
        ]
        frames = sample_representative_frames(episode, segments)
        assert frames == sorted(set(frames))

    def test_many_key_phase_segments_still_capped_at_max(self):
        episode = {"start_frame": 0, "end_frame": 1000}
        segments = [_seg("grasp", i * 50, i * 50 + 40) for i in range(20)]
        frames = sample_representative_frames(episode, segments)
        assert len(frames) <= cfg.VLM_MAX_SAMPLE_FRAMES


class TestEstimateCostUsd:
    def test_zero_usage_zero_cost(self):
        assert estimate_cost_usd({"input_tokens": 0, "output_tokens": 0}) == 0.0

    def test_matches_pricing_table(self):
        pricing = cfg.VLM_PRICING_USD_PER_MTOK
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        expected = pricing["input"] + pricing["output"]
        assert estimate_cost_usd(usage) == pytest.approx(expected)

    def test_output_tokens_cost_more_than_input(self):
        """Output pricing should be higher per-token (matches published Claude pricing shape)."""
        input_only = estimate_cost_usd({"input_tokens": 1000, "output_tokens": 0})
        output_only = estimate_cost_usd({"input_tokens": 0, "output_tokens": 1000})
        assert output_only > input_only


class TestShouldSpotcheck:
    def test_rate_zero_never_checks(self):
        for i in range(20):
            assert should_spotcheck(f"ep_{i}", 0.0) is False

    def test_rate_one_always_checks(self):
        for i in range(20):
            assert should_spotcheck(f"ep_{i}", 1.0) is True

    def test_deterministic_for_same_episode_id(self):
        assert should_spotcheck("session_001_ep00", 0.3) == should_spotcheck("session_001_ep00", 0.3)

    def test_roughly_matches_rate_across_many_episodes(self):
        """Not an exact proportion test (hash-based), but should land in a sane ballpark for a large sample."""
        n = 2000
        rate = 0.2
        checked = sum(1 for i in range(n) if should_spotcheck(f"episode_{i}", rate))
        fraction = checked / n
        assert 0.1 < fraction < 0.3


class TestCheckTaskDisagreement:
    def test_matching_task_no_disagreement(self):
        result = check_task_disagreement("bolt_tightening", "bolt_tightening")
        assert result["disagree"] is False

    def test_case_and_spacing_normalized(self):
        result = check_task_disagreement("Bolt Tightening", "bolt_tightening")
        assert result["disagree"] is False

    def test_genuinely_different_task_flags_disagreement(self):
        result = check_task_disagreement("material transfer", "bolt_tightening")
        assert result["disagree"] is True

    def test_unknown_classifier_task_does_not_flag(self):
        """If the classifier itself couldn't decide (needs_human_review -> 'unknown'), don't pile on a disagreement flag."""
        result = check_task_disagreement("bolt_tightening", "unknown")
        assert result["disagree"] is False

    def test_empty_vlm_guess_does_not_flag(self):
        result = check_task_disagreement("", "bolt_tightening")
        assert result["disagree"] is False
