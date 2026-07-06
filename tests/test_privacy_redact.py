"""
DatraAI Pipeline — Tests for Step 03b: Privacy / PII Redaction
Covers every pure/GPU-free function: blur application, histogram similarity,
confidence tiering, the cross-sample text-region tracker. Face detection
(MediaPipe) and OCR (EasyOCR) are NOT exercised here — see the module's own
docstring for what needs real validation before trusting those paths.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

_spec = importlib.util.spec_from_file_location(
    "privacy_redact",
    str(Path(__file__).resolve().parent.parent / "scripts" / "03b_privacy_redact.py"),
)
_privacy_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_privacy_mod)

_blur_region = _privacy_mod._blur_region
_compute_reference_histogram = _privacy_mod._compute_reference_histogram
_face_similarity = _privacy_mod._face_similarity
classify_text_confidence = _privacy_mod.classify_text_confidence
TextRegionTracker = _privacy_mod.TextRegionTracker
_clamp_bbox = _privacy_mod._clamp_bbox
_check_ocr_propagation_invariant = _privacy_mod._check_ocr_propagation_invariant


def _random_frame(h=100, w=100, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)


class TestBlurRegion:
    def test_region_actually_changes(self):
        frame = _random_frame()
        original = frame.copy()
        _blur_region(frame, (10, 10, 60, 60), kernel_size=15)
        assert not np.array_equal(frame[10:60, 10:60], original[10:60, 10:60])

    def test_outside_region_untouched(self):
        frame = _random_frame()
        original = frame.copy()
        _blur_region(frame, (10, 10, 60, 60), kernel_size=15)
        assert np.array_equal(frame[0:10, :], original[0:10, :])
        assert np.array_equal(frame[:, 0:10], original[:, 0:10])

    def test_bbox_clamped_to_frame_bounds(self):
        frame = _random_frame(h=50, w=50)
        # Wildly out-of-range bbox should not raise or corrupt shape.
        result = _blur_region(frame, (-100, -100, 500, 500), kernel_size=15)
        assert result.shape == (50, 50, 3)

    def test_degenerate_bbox_is_noop(self):
        frame = _random_frame()
        original = frame.copy()
        _blur_region(frame, (30, 30, 20, 20), kernel_size=15)  # x2<x1, y2<y1
        assert np.array_equal(frame, original)

    def test_even_kernel_size_forced_odd(self):
        frame = _random_frame()
        # Should not raise even with an even kernel size input.
        result = _blur_region(frame, (10, 10, 60, 60), kernel_size=16)
        assert result is not None

    def test_tiny_region_does_not_crash(self):
        frame = _random_frame()
        result = _blur_region(frame, (10, 10, 12, 12), kernel_size=51)
        assert result.shape == frame.shape


class TestFaceSimilarity:
    def test_no_reference_returns_zero(self):
        crop = _random_frame(h=20, w=20)
        assert _face_similarity(crop, None) == 0.0

    def test_empty_crop_returns_zero(self):
        ref_hist = _compute_reference_histogram(_random_frame(h=20, w=20))
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        assert _face_similarity(empty_crop, ref_hist) == 0.0

    def test_identical_image_scores_high_similarity(self):
        # Solid-color image against itself should correlate near 1.0.
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        ref_hist = _compute_reference_histogram(img)
        similarity = _face_similarity(img.copy(), ref_hist)
        assert similarity > 0.99

    def test_returns_a_float(self):
        ref_hist = _compute_reference_histogram(_random_frame(seed=1))
        similarity = _face_similarity(_random_frame(seed=2), ref_hist)
        assert isinstance(similarity, float)


class TestClassifyTextConfidence:
    def test_below_min_is_ignored(self):
        assert classify_text_confidence(cfg.PRIVACY_OCR_MIN_CONFIDENCE - 0.01) == "ignore"

    def test_middle_band_is_blur_and_flag(self):
        mid = (cfg.PRIVACY_OCR_MIN_CONFIDENCE + cfg.PRIVACY_OCR_BLUR_CONFIDENCE) / 2.0
        assert classify_text_confidence(mid) == "blur_and_flag"

    def test_at_or_above_blur_confidence_is_confident_blur(self):
        assert classify_text_confidence(cfg.PRIVACY_OCR_BLUR_CONFIDENCE) == "blur"
        assert classify_text_confidence(1.0) == "blur"

    def test_exactly_at_min_confidence_is_not_ignored(self):
        # >= min, not strictly >, per the implementation's boundary.
        assert classify_text_confidence(cfg.PRIVACY_OCR_MIN_CONFIDENCE) != "ignore"


class TestTextRegionTracker:
    def test_active_bboxes_after_update(self):
        tracker = TextRegionTracker(grace_frames=5)
        tracker.update([(0, 0, 10, 10), (20, 20, 30, 30)], current_frame=0)
        assert set(tracker.active_bboxes()) == {(0, 0, 10, 10), (20, 20, 30, 30)}

    def test_region_persists_within_grace_period(self):
        tracker = TextRegionTracker(grace_frames=5)
        tracker.update([(0, 0, 10, 10)], current_frame=0)
        tracker.prune(current_frame=5)  # exactly at the grace boundary
        assert (0, 0, 10, 10) in tracker.active_bboxes()

    def test_region_expires_after_grace_period(self):
        tracker = TextRegionTracker(grace_frames=5)
        tracker.update([(0, 0, 10, 10)], current_frame=0)
        tracker.prune(current_frame=6)  # one past the grace boundary
        assert tracker.active_bboxes() == []

    def test_redetection_refreshes_the_timer(self):
        tracker = TextRegionTracker(grace_frames=5)
        tracker.update([(0, 0, 10, 10)], current_frame=0)
        tracker.update([(0, 0, 10, 10)], current_frame=4)  # re-seen before expiry
        tracker.prune(current_frame=8)  # would have expired if not refreshed (0+5<8)
        assert (0, 0, 10, 10) in tracker.active_bboxes()

    def test_gap_between_ocr_samples_stays_covered(self):
        """
        The whole point of the tracker: OCR runs only every
        PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES frames, but every frame in
        between should still see the last known regions blurred. This uses
        the REAL configured PRIVACY_OCR_PROPAGATION_FRAMES value, not a
        value chosen to make the test pass — if config.py's values ever
        regress on the invariant, this test (and
        _check_ocr_propagation_invariant) will both catch it.
        """
        tracker = TextRegionTracker(grace_frames=cfg.PRIVACY_OCR_PROPAGATION_FRAMES)
        tracker.update([(5, 5, 15, 15)], current_frame=0)
        for frame_idx in range(1, cfg.PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES):
            tracker.prune(frame_idx)
            assert (5, 5, 15, 15) in tracker.active_bboxes(), f"lost coverage at frame {frame_idx}"


class TestOcrPropagationInvariant:
    """
    Guards against reintroducing the exact bug found while writing these
    tests: PRIVACY_OCR_PROPAGATION_FRAMES shorter than
    PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES silently leaves a window of frames
    unredacted between OCR samples.
    """

    def test_current_config_satisfies_invariant(self):
        # Must not raise against the actual config.py values.
        _check_ocr_propagation_invariant()

    def test_raises_when_propagation_shorter_than_sample_interval(self, monkeypatch):
        monkeypatch.setattr(cfg, "REDACT_TEXT_AND_BADGES", True)
        monkeypatch.setattr(cfg, "PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES", 30)
        monkeypatch.setattr(cfg, "PRIVACY_OCR_PROPAGATION_FRAMES", 15)
        with pytest.raises(ValueError):
            _check_ocr_propagation_invariant()

    def test_skipped_entirely_when_text_redaction_disabled(self, monkeypatch):
        monkeypatch.setattr(cfg, "REDACT_TEXT_AND_BADGES", False)
        monkeypatch.setattr(cfg, "PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES", 30)
        monkeypatch.setattr(cfg, "PRIVACY_OCR_PROPAGATION_FRAMES", 0)
        _check_ocr_propagation_invariant()  # must not raise — text redaction is off


class TestClampBbox:
    def test_normal_bbox_unchanged(self):
        assert _clamp_bbox(10, 10, 50, 50, width=100, height=100) == (10, 10, 50, 50)

    def test_negative_coords_clamped_to_zero(self):
        assert _clamp_bbox(-10, -10, 50, 50, width=100, height=100) == (0, 0, 50, 50)

    def test_oversized_coords_clamped_to_frame(self):
        assert _clamp_bbox(10, 10, 500, 500, width=100, height=100) == (10, 10, 100, 100)

    def test_floats_are_cast_to_int(self):
        result = _clamp_bbox(10.7, 10.2, 50.9, 50.1, width=100, height=100)
        assert all(isinstance(v, int) for v in result)


def _write_synthetic_video(path, n_frames, width=64, height=48, fps=30.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(n_frames):
        writer.write(np.full((height, width, 3), 200, dtype=np.uint8))
    writer.release()


class TestRunReportsPerTierBreakdownAndFlaggedRegions:
    """
    Integration test for run()'s orchestration — monkeypatches the detector
    functions so no real model is loaded (fully hermetic/fast), while
    verifying the exact reporting logic this conversation's review flagged
    as missing: a per-tier breakdown of raw OCR detections (not inflated by
    cross-frame propagation) and an actionable, inspectable trace for every
    "blur_and_flag" detection (frame_idx + bbox + confidence).
    """

    def test_tier_counts_and_flagged_regions(self, tmp_path, monkeypatch):
        session_id = "sess_ocr_report_test"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)
        _write_synthetic_video(proc_dir / "compressed.mp4", n_frames=cfg.PRIVACY_OCR_SAMPLE_INTERVAL_FRAMES + 5)

        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "REDACT_TEXT_AND_BADGES", True)
        monkeypatch.setattr(cfg, "REDACT_BYSTANDER_FACES", True)

        # No faces at all — isolate this test to the text-redaction path.
        class _DummyFaceDetector:
            def close(self):
                pass

        monkeypatch.setattr(_privacy_mod, "_get_face_detector", lambda: _DummyFaceDetector())
        monkeypatch.setattr(_privacy_mod, "_detect_faces", lambda detector, frame: [])

        # One detection per confidence tier, only on the first OCR sample
        # (frame 0): below MIN -> ignore, between MIN and BLUR -> blur_and_flag,
        # at/above BLUR -> confidently blurred. No detections on later samples.
        ignore_conf = cfg.PRIVACY_OCR_MIN_CONFIDENCE - 0.05
        flag_conf = (cfg.PRIVACY_OCR_MIN_CONFIDENCE + cfg.PRIVACY_OCR_BLUR_CONFIDENCE) / 2.0
        blur_conf = cfg.PRIVACY_OCR_BLUR_CONFIDENCE

        call_count = {"n": 0}

        def fake_run_ocr(reader, frame):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [
                    {"bbox": (0, 0, 5, 5), "confidence": ignore_conf},
                    {"bbox": (10, 10, 20, 20), "confidence": flag_conf},
                    {"bbox": (30, 30, 40, 40), "confidence": blur_conf},
                ]
            return []

        monkeypatch.setattr(_privacy_mod, "_get_ocr_reader", lambda: object())
        monkeypatch.setattr(_privacy_mod, "_run_ocr", fake_run_ocr)

        report = _privacy_mod.run(session_id)

        tiers = report["text_detection_tiers"]
        assert tiers["ignored_below_min_confidence"] == 1
        assert tiers["blurred_and_flagged_for_review"] == 1
        assert tiers["confidently_blurred"] == 1

        # The flagged region must be inspectable: frame, bbox, confidence.
        assert len(report["flagged_regions"]) == 1
        flagged = report["flagged_regions"][0]
        assert flagged["frame_idx"] == 0
        assert flagged["bbox"] == [10, 10, 20, 20]
        assert flagged["confidence"] == pytest.approx(flag_conf, abs=1e-4)

        # frames_flagged_for_review counts FRAMES with >=1 flagged
        # detection, not raw detection count — only frame 0 qualifies here.
        assert report["frames_flagged_for_review"] == 1
