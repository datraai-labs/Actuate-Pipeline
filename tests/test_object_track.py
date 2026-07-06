"""
DatraAI Pipeline — Tests for Step 04c: Object Detection & Tracking (v2 addendum §3)
Pure-logic tests only — no GPU/model dependency (Grounding DINO / SAM2 are
exercised separately against real session_001 footage; see
docs/PIPELINE_STATUS.md §3 for that real-data verification report).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "object_track",
    str(Path(__file__).resolve().parent.parent / "scripts" / "04c_object_track.py"),
)
_ot_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ot_mod)

iou = _ot_mod.iou
dominant_hand_center_norm = _ot_mod.dominant_hand_center_norm
filter_detections_near_hand = _ot_mod.filter_detections_near_hand
match_or_create_track_ids = _ot_mod.match_or_create_track_ids
mask_to_bbox_and_centroid = _ot_mod.mask_to_bbox_and_centroid


class TestIoU:
    def test_identical_boxes_iou_one(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_partial_overlap(self):
        # [0,0,10,10] area=100, [5,5,15,15] area=100, intersection [5,5,10,10] area=25
        # union = 100+100-25=175 -> iou=25/175
        result = iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert result == pytest.approx(25 / 175)

    def test_degenerate_zero_area_box(self):
        assert iou([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0


class TestDominantHandCenterNorm:
    def test_no_hands_detected_returns_none(self):
        assert dominant_hand_center_norm({"hands_detected": False}) is None
        assert dominant_hand_center_norm(None) is None

    def test_missing_dominant_hand_key_returns_none(self):
        assert dominant_hand_center_norm({"hands_detected": True, "dominant_hand": None}) is None

    def test_computes_mean_of_landmarks(self):
        landmarks = [[0.4, 0.6, 0.0]] * 21  # constant -> mean is trivially known
        pose_frame = {
            "hands_detected": True,
            "dominant_hand": "right",
            "right_hand": {"landmarks": landmarks},
        }
        result = dominant_hand_center_norm(pose_frame)
        assert result == pytest.approx((0.4, 0.6))


class TestFilterDetectionsNearHand:
    def test_no_hand_yields_empty(self):
        detections = [{"box": [0, 0, 10, 10], "label": "bolt", "score": 0.9}]
        assert filter_detections_near_hand(detections, None, 0.3, 100, 100) == []

    def test_keeps_detection_within_radius(self):
        detections = [{"box": [40, 40, 60, 60], "label": "bolt", "score": 0.9}]  # centroid (0.5, 0.5)
        result = filter_detections_near_hand(detections, (0.5, 0.5), 0.3, 100, 100)
        assert len(result) == 1

    def test_excludes_detection_outside_radius(self):
        detections = [{"box": [0, 0, 10, 10], "label": "bolt", "score": 0.9}]  # centroid (0.05, 0.05)
        result = filter_detections_near_hand(detections, (0.9, 0.9), 0.3, 100, 100)
        assert result == []

    def test_mixed_detections_partially_filtered(self):
        near = {"box": [45, 45, 55, 55], "label": "bolt", "score": 0.9}   # centroid (0.5, 0.5)
        far = {"box": [0, 0, 5, 5], "label": "box", "score": 0.8}         # centroid (0.025, 0.025)
        result = filter_detections_near_hand([near, far], (0.5, 0.5), 0.3, 100, 100)
        assert result == [near]


class TestMatchOrCreateTrackIds:
    def test_no_existing_tracks_all_new(self):
        detections = [{"box": [0, 0, 10, 10], "label": "bolt", "score": 0.9}]
        assigned, next_id = match_or_create_track_ids(detections, {}, iou_min=0.3, next_track_id=1)
        assert assigned == [(1, detections[0])]
        assert next_id == 2

    def test_matches_overlapping_existing_track(self):
        detections = [{"box": [1, 1, 11, 11], "label": "bolt", "score": 0.9}]  # near-identical to existing track's last box
        existing = {5: [0, 0, 10, 10]}
        assigned, next_id = match_or_create_track_ids(detections, existing, iou_min=0.3, next_track_id=6)
        assert assigned[0][0] == 5  # matched the existing track_id, not a new one
        assert next_id == 6  # counter untouched since no new track was created

    def test_low_overlap_creates_new_track(self):
        detections = [{"box": [50, 50, 60, 60], "label": "bolt", "score": 0.9}]  # far from existing track
        existing = {5: [0, 0, 10, 10]}
        assigned, next_id = match_or_create_track_ids(detections, existing, iou_min=0.3, next_track_id=6)
        assert assigned[0][0] == 6  # a genuinely new track
        assert next_id == 7

    def test_two_existing_tracks_do_not_collide(self):
        """Two detections matching two different existing tracks must each get their own track_id, not both claim the same one."""
        det_a = {"box": [1, 1, 11, 11], "label": "bolt", "score": 0.9}
        det_b = {"box": [101, 101, 111, 111], "label": "box", "score": 0.8}
        existing = {5: [0, 0, 10, 10], 8: [100, 100, 110, 110]}
        assigned, next_id = match_or_create_track_ids([det_a, det_b], existing, iou_min=0.3, next_track_id=9)
        ids = {tid for tid, _ in assigned}
        assert ids == {5, 8}
        assert next_id == 9


class TestMaskToBboxAndCentroid:
    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 100), dtype=bool)
        assert mask_to_bbox_and_centroid(mask, 100, 100) is None

    def test_single_pixel_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 60] = True  # row=y=50, col=x=60
        result = mask_to_bbox_and_centroid(mask, 100, 100)
        assert result["centroid_norm"] == pytest.approx([0.60, 0.50])

    def test_rectangular_mask_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 30:40] = True  # rows(y) 10-19, cols(x) 30-39
        result = mask_to_bbox_and_centroid(mask, 100, 100)
        x1, y1, x2, y2 = result["bbox"]
        assert x1 == pytest.approx(0.30)
        assert y1 == pytest.approx(0.10)
        assert x2 == pytest.approx(0.39)
        assert y2 == pytest.approx(0.19)
