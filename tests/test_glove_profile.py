"""
DatraAI Pipeline — Tests for utils/glove_profile.py (v2 addendum §2)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.glove_profile import resolve_grasp_thresholds


class TestResolveGraspThresholds:
    def test_none_glove_type_matches_raw_config_defaults(self):
        result = resolve_grasp_thresholds("none")
        assert result["power_grasp_dist"] == cfg.POWER_GRASP_DIST
        assert result["lateral_pinch_dist"] == cfg.LATERAL_PINCH_DIST
        assert result["multiplier"] == 1.0

    def test_thin_glove_scales_up_thresholds(self):
        result = resolve_grasp_thresholds("thin")
        assert result["power_grasp_dist"] > cfg.POWER_GRASP_DIST
        assert result["lateral_pinch_dist"] > cfg.LATERAL_PINCH_DIST
        assert result["multiplier"] == cfg.GLOVE_THRESHOLD_MULTIPLIERS["thin"]

    def test_thick_glove_scales_more_than_thin(self):
        thin = resolve_grasp_thresholds("thin")
        thick = resolve_grasp_thresholds("thick")
        assert thick["power_grasp_dist"] > thin["power_grasp_dist"]
        assert thick["lateral_pinch_dist"] > thin["lateral_pinch_dist"]

    def test_missing_glove_type_falls_back_to_default_not_error(self):
        result = resolve_grasp_thresholds(None)
        assert result["glove_type"] == cfg.GLOVE_TYPE_DEFAULT
        assert result["multiplier"] == 1.0

    def test_unrecognized_glove_type_falls_back_to_default(self):
        result = resolve_grasp_thresholds("exosuit_gauntlet")
        assert result["glove_type"] == cfg.GLOVE_TYPE_DEFAULT
        assert result["multiplier"] == 1.0

    def test_thresholds_scale_proportionally_with_multiplier(self):
        result = resolve_grasp_thresholds("thick")
        expected_power = round(cfg.POWER_GRASP_DIST * cfg.GLOVE_THRESHOLD_MULTIPLIERS["thick"], 6)
        expected_pinch = round(cfg.LATERAL_PINCH_DIST * cfg.GLOVE_THRESHOLD_MULTIPLIERS["thick"], 6)
        assert result["power_grasp_dist"] == expected_power
        assert result["lateral_pinch_dist"] == expected_pinch
