"""
DatraAI Pipeline — Tests for Step 07: Task Classification (v2 addendum §6)
Tests _classify_episode's pure scoring logic and run()'s per-episode
filtering/orchestration against synthetic multi-episode fixtures.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.episode_utils import filter_frames

_spec = importlib.util.spec_from_file_location(
    "task_classify",
    str(Path(__file__).resolve().parent.parent / "scripts" / "07_task_classify.py"),
)
_task_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task_mod)

_classify_episode = _task_mod._classify_episode
_dominant_object_class = _task_mod._dominant_object_class
run = _task_mod.run


def _frame(idx, active):
    return {"frame_idx": idx, "active_primitives": active}


class TestClassifyEpisodePureFunction:
    def test_empty_primitives_yields_unknown(self):
        result = _classify_episode([])
        assert result["L1_task"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["needs_human_review"] is True

    def test_strong_signature_match_is_confident(self):
        # Build enough frames to fully satisfy bolt_tightening's signature.
        signature = cfg.TASK_SIGNATURES["bolt_tightening"]
        frames = []
        idx = 0
        for prim, count in signature.items():
            for _ in range(count):
                frames.append(_frame(idx, [prim]))
                idx += 1
        result = _classify_episode(frames)
        assert result["L1_task"] == "bolt_tightening"
        assert result["confidence"] >= 0.99

    def test_deliberate_tie_across_two_tasks_returns_unknown(self):
        """
        Regression test for a real bug: session_001's real primitive counts
        scored 5 tasks tied at 1.0, and the previous code silently picked
        whichever came first in TASK_SIGNATURES' dict order (a
        confident-looking but arbitrary, wrong label). Construct frames
        that fully and equally satisfy two DISJOINT task signatures
        (assembly_insert and packaging_fold share no required primitives)
        so both land at score 1.0 — the classifier must not pick either
        one; it must fall back to unknown with needs_human_review.
        """
        assembly_sig = cfg.TASK_SIGNATURES["assembly_insert"]  # wrist_flex, power_grasp, contact_onset
        packaging_sig = cfg.TASK_SIGNATURES["packaging_fold"]  # finger_curl, finger_extend
        assert set(assembly_sig) & set(packaging_sig) == set(), "fixture assumption: signatures must be disjoint"

        frames = []
        idx = 0
        for prim, count in {**assembly_sig, **packaging_sig}.items():
            for _ in range(count):
                frames.append(_frame(idx, [prim]))
                idx += 1

        result = _classify_episode(frames)

        # Sanity check the trap is real: both tasks must actually score at
        # (or within the tie margin of) the top score before asserting the
        # tie-handling behavior — otherwise this test would pass vacuously.
        assembly_score = result["all_scores"]["assembly_insert"]
        packaging_score = result["all_scores"]["packaging_fold"]
        max_score = max(result["all_scores"].values())
        assert assembly_score == max_score
        assert packaging_score == max_score

        assert result["L1_task"] == "unknown"
        assert result["needs_human_review"] is True
        assert result["L1_task"] != "assembly_insert"
        assert result["L1_task"] != "packaging_fold"

    def test_low_confidence_forces_unknown_and_review_flag(self):
        # A single frame of one required primitive, well short of every
        # signature's required count -> every score should land under 0.5.
        result = _classify_episode([_frame(0, ["reach_onset"])])
        assert result["needs_human_review"] is True
        assert result["L1_task"] == "unknown"

    def test_primitive_counts_reflect_input_frames_only(self):
        frames = [_frame(0, ["power_grasp", "contact_onset"]), _frame(1, ["power_grasp"])]
        result = _classify_episode(frames)
        assert result["primitive_counts"]["power_grasp"] == 2
        assert result["primitive_counts"]["contact_onset"] == 1


def _clean_signature_fixture(task_name):
    """A frame set containing ONLY task_name's own required primitives, at exactly the required counts — nothing else."""
    sig = cfg.TASK_SIGNATURES[task_name]
    frames = []
    idx = 0
    for prim, count in sig.items():
        for _ in range(count):
            frames.append(_frame(idx, [prim]))
            idx += 1
    return frames


def _single_primitive_fixture(prim, count):
    return [_frame(i, [prim]) for i in range(count)]


class TestSyntheticSignatureDiscriminability:
    """
    ⚠️ SYNTHETIC DISCRIMINABILITY TESTING — NOT REAL-DATA VALIDATION.

    These tests confirm TASK_SIGNATURES' internal consistency using
    hand-constructed primitive-count fixtures: does each signature, when
    fed EXACTLY its own required primitives and nothing else, uniquely win
    without tying against an unrelated task? Does a single, generic
    primitive alone (with every other required primitive completely
    absent) correctly fail to cross the confidence threshold?

    This says NOTHING about whether any of these signatures actually
    fires correctly on real footage of a worker performing that task —
    no such footage exists in this repo to test against (see
    docs/PIPELINE_STATUS.md's "KNOWN DATA ISSUE" note). Conflating this
    kind of synthetic self-consistency test with real-data validation is
    exactly the mistake that let session_001 — later found to depict
    paperwork sorting, not any defined task — pass as a validation
    reference for as long as it did. Don't repeat that here.
    """

    @pytest.mark.parametrize("task_name", list(cfg.TASK_SIGNATURES.keys()))
    def test_clean_fixture_uniquely_wins_its_own_task(self, task_name):
        result = _classify_episode(_clean_signature_fixture(task_name))
        assert result["L1_task"] == task_name, (
            f"{task_name}'s own clean fixture should classify as {task_name}, "
            f"got {result['L1_task']!r} (all_scores={result['all_scores']})"
        )
        assert result["needs_human_review"] is False
        assert result["confidence"] == pytest.approx(1.0)

    def test_original_2primitive_false_positives_now_return_unknown(self):
        """
        Regression test for a real discriminability bug found via this same
        synthetic-fixture method: the original (pre-2026-07-06)
        TASK_SIGNATURES had 5 tasks defined by only 2 required primitives,
        so satisfying just ONE of them (with the other completely absent)
        averaged to exactly 0.5 — the classifier's confidence threshold —
        and confidently fired anyway. `power_grasp=20` alone (zero
        transport) fired material_transfer; `lateral_pinch`/`idle`/
        `finger_curl` alone did the same for their tasks. Every
        TASK_SIGNATURES entry now requires >=3 primitives specifically to
        close this gap — confirm a single generic primitive alone no
        longer confidently classifies anything.
        """
        probes = [
            ("power_grasp", 20),   # used to confidently fire material_transfer
            ("transport", 80),     # shared by 3 tasks, must not fire any alone
            ("finger_extend", 40), # used to confidently fire box_seal
            ("lateral_pinch", 30), # used to confidently fire label_apply
            ("idle", 100),         # used to confidently fire inspection_visual
            ("finger_curl", 50),   # used to confidently fire packaging_fold
            ("reach_onset", 20),   # shared by 3 tasks, must not fire any alone
        ]
        for prim, count in probes:
            result = _classify_episode(_single_primitive_fixture(prim, count))
            assert result["L1_task"] == "unknown", (
                f"{prim}={count} alone should NOT confidently classify any "
                f"task, got {result['L1_task']!r} (confidence={result['confidence']}, "
                f"all_scores={result['all_scores']})"
            )
            assert result["needs_human_review"] is True

    def test_no_two_clean_fixtures_tie_against_each_other(self):
        """
        Broader sweep: for every pair of tasks, the fixture cleanly built
        for task A must not also tie for task B — i.e. no clean fixture
        should trigger the >=2-tasks-within-TASK_TIE_MARGIN fallback.
        (test_clean_fixture_uniquely_wins_its_own_task already covers this
        indirectly via needs_human_review is False, but this makes the
        "why" explicit: no OTHER task's score sits within the tie margin.)
        """
        for task_name in cfg.TASK_SIGNATURES:
            result = _classify_episode(_clean_signature_fixture(task_name))
            own_score = result["all_scores"][task_name]
            competitors_within_margin = [
                t for t, s in result["all_scores"].items()
                if t != task_name and (own_score - s) <= cfg.TASK_TIE_MARGIN
            ]
            assert competitors_within_margin == [], (
                f"{task_name}'s clean fixture ties with {competitors_within_margin} "
                f"(all_scores={result['all_scores']})"
            )


class TestRunPerEpisodeFiltering:
    """
    Confirm run() actually scopes each episode's classification to its own
    frame range — a two-episode session where episode 0's frames signal one
    task and episode 1's frames signal a different (or empty) one must NOT
    have episode 1's classification contaminated by episode 0's primitives
    (or vice versa) via an unfiltered/global primitive count.
    """

    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_multi_ep"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)
        return session_id, proc_dir

    def test_two_episodes_scored_independently(self, tmp_path, monkeypatch):
        session_id, proc_dir = self._setup(tmp_path, monkeypatch)

        # Episode 0 (frames 0-199): strongly matches bolt_tightening.
        bolt_sig = cfg.TASK_SIGNATURES["bolt_tightening"]
        primitives = []
        idx = 0
        for prim, count in bolt_sig.items():
            for _ in range(count):
                primitives.append(_frame(idx, [prim]))
                idx += 1
        assert idx < 200, "fixture assumption: episode 0 fits in frames 0-199"

        # Episode 1 (frames 300-349): no primitives at all -> must classify
        # as unknown/needs_review, NOT inherit episode 0's strong match.
        # (no frames appended for this range)

        episodes = {
            "session_id": session_id,
            "episode_gap_threshold_sec": 8.0,
            "min_episode_duration_sec": 2.0,
            "episodes": [
                {"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": 199,
                 "start_sec": 0.0, "end_sec": 6.6333, "duration_sec": 6.6333},
                {"episode_id": f"{session_id}_ep01", "start_frame": 300, "end_frame": 349,
                 "start_sec": 10.0, "end_sec": 11.6333, "duration_sec": 1.6333},
            ],
        }
        with open(proc_dir / "episodes.json", "w") as f:
            json.dump(episodes, f)
        with open(proc_dir / "primitives.json", "w") as f:
            json.dump(primitives, f)

        result = run(session_id)

        assert len(result["episodes"]) == 2
        ep0, ep1 = result["episodes"]
        assert ep0["episode_id"] == f"{session_id}_ep00"
        assert ep0["L1_task"] == "bolt_tightening"
        assert ep0["needs_human_review"] is False

        assert ep1["episode_id"] == f"{session_id}_ep01"
        assert ep1["L1_task"] == "unknown"
        assert ep1["needs_human_review"] is True
        # The critical contamination check: episode 1 must show zero
        # primitive counts, proving filter_frames actually excluded
        # episode 0's frames rather than scoring against the whole session.
        assert ep1["primitive_counts"] == {}

    def test_missing_primitives_json_raises(self, tmp_path, monkeypatch):
        session_id, proc_dir = self._setup(tmp_path, monkeypatch)
        with open(proc_dir / "episodes.json", "w") as f:
            json.dump({"session_id": session_id, "episodes": []}, f)
        try:
            run(session_id)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_filter_frames_boundary_is_inclusive(self):
        # Sanity-check the shared utility run() depends on: frames exactly
        # at start_frame/end_frame must be included, not off-by-one dropped.
        frames = [_frame(9, ["a"]), _frame(10, ["b"]), _frame(20, ["c"]), _frame(21, ["d"])]
        filtered = filter_frames(frames, 10, 20)
        assert [f["frame_idx"] for f in filtered] == [10, 20]


def _obj_frame(frame_idx, tracked_objects):
    return {"frame_idx": frame_idx, "stub": False, "tracked_objects": tracked_objects}


def _tracked(class_label, track_id=1, confidence=0.8):
    return {
        "track_id": track_id, "class_label": class_label, "confidence": confidence,
        "bbox": [0.4, 0.4, 0.6, 0.6], "centroid_norm": [0.5, 0.5], "is_stub": False,
    }


class TestDominantObjectClass:
    def test_no_object_tracks_returns_none(self):
        primitives = [_frame(0, ["power_grasp"])]
        assert _dominant_object_class(primitives, []) is None

    def test_ignores_frames_without_active_primitives(self):
        primitives = [_frame(0, [])]  # no active primitives -> not "active manipulation"
        tracks = [_obj_frame(0, [_tracked("bolt")])]
        assert _dominant_object_class(primitives, tracks) is None

    def test_picks_most_frequent_class_across_active_frames(self):
        primitives = [_frame(i, ["power_grasp"]) for i in range(3)]
        tracks = [
            _obj_frame(0, [_tracked("bolt")]),
            _obj_frame(1, [_tracked("bolt")]),
            _obj_frame(2, [_tracked("box")]),
        ]
        assert _dominant_object_class(primitives, tracks) == "bolt"

    def test_missing_track_for_a_frame_is_skipped_not_an_error(self):
        primitives = [_frame(0, ["power_grasp"]), _frame(1, ["power_grasp"])]
        tracks = [_obj_frame(0, [_tracked("box")])]  # no entry for frame 1
        assert _dominant_object_class(primitives, tracks) == "box"


class TestObjectMatchBonus:
    """v2 addendum §3 — the object-identity bonus must actually change the classifier's decision, not just decorate the output."""

    def test_bonus_breaks_a_primitive_only_tie(self):
        """
        Regression scenario found via real analysis: with only transport
        (80) + power_grasp (20) present — reach_onset omitted —
        material_transfer and pick_and_place tie at 0.667 on primitives
        alone, correctly returning unknown. A dominant_object_class in
        material_transfer's expected classes must break that tie in its
        favor.
        """
        frames = []
        idx = 0
        for prim, count in {"transport": 80, "power_grasp": 20}.items():
            for _ in range(count):
                frames.append(_frame(idx, [prim]))
                idx += 1

        without_bonus = _classify_episode(frames)
        assert without_bonus["L1_task"] == "unknown"  # confirms the tie is real before adding the bonus

        with_bonus = _classify_episode(frames, dominant_object_class="box")  # "box" is in material_transfer's expected classes
        assert with_bonus["L1_task"] == "material_transfer"
        assert with_bonus["object_bonus_applied"]["material_transfer"] == pytest.approx(cfg.OBJECT_MATCH_BONUS)

    def test_bonus_is_capped_at_one(self):
        signature = cfg.TASK_SIGNATURES["bolt_tightening"]
        frames = []
        idx = 0
        for prim, count in signature.items():
            for _ in range(count):
                frames.append(_frame(idx, [prim]))
                idx += 1
        result = _classify_episode(frames, dominant_object_class="bolt")  # bolt_tightening already at 1.0
        assert result["all_scores"]["bolt_tightening"] == 1.0
        assert result["object_bonus_applied"]["bolt_tightening"] == pytest.approx(0.0)  # nothing left to add

    def test_no_matching_object_class_applies_no_bonus(self):
        frames = [_frame(0, ["power_grasp"])]
        result = _classify_episode(frames, dominant_object_class="a_completely_unrelated_thing")
        assert result["object_bonus_applied"] == {}

    def test_none_object_class_applies_no_bonus(self):
        frames = [_frame(0, ["power_grasp"])]
        result = _classify_episode(frames, dominant_object_class=None)
        assert result["object_bonus_applied"] == {}
        assert result["dominant_object_class"] is None

    def test_output_reports_dominant_object_class(self):
        frames = [_frame(0, ["power_grasp"])]
        result = _classify_episode(frames, dominant_object_class="tool")
        assert result["dominant_object_class"] == "tool"


class TestRunWiresObjectBonusEndToEnd:
    def test_object_tracks_json_present_flows_through_to_classification(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_object_bonus_e2e"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        # Same tie scenario as TestObjectMatchBonus.test_bonus_breaks_a_primitive_only_tie,
        # run through the full run() orchestration this time.
        primitives = []
        idx = 0
        for prim, count in {"transport": 80, "power_grasp": 20}.items():
            for _ in range(count):
                primitives.append(_frame(idx, [prim]))
                idx += 1
        with open(proc_dir / "primitives.json", "w") as f:
            json.dump(primitives, f)

        episodes = {
            "session_id": session_id,
            "episodes": [{"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": idx - 1,
                          "start_sec": 0.0, "end_sec": idx / 30.0, "duration_sec": idx / 30.0}],
        }
        with open(proc_dir / "episodes.json", "w") as f:
            json.dump(episodes, f)

        object_tracks = [_obj_frame(i, [_tracked("box")]) for i in range(idx)]
        with open(proc_dir / "object_tracks.json", "w") as f:
            json.dump(object_tracks, f)

        result = run(session_id)
        ep = result["episodes"][0]
        assert ep["L1_task"] == "material_transfer"
        assert ep["dominant_object_class"] == "box"

    def test_missing_object_tracks_json_is_a_no_op(self, tmp_path, monkeypatch):
        """A session with no object_tracks.json (pre-§3 processed dir) must classify exactly as before — no crash, no bonus."""
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_no_object_tracks"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        signature = cfg.TASK_SIGNATURES["bolt_tightening"]
        primitives = []
        idx = 0
        for prim, count in signature.items():
            for _ in range(count):
                primitives.append(_frame(idx, [prim]))
                idx += 1
        with open(proc_dir / "primitives.json", "w") as f:
            json.dump(primitives, f)
        episodes = {
            "session_id": session_id,
            "episodes": [{"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": idx - 1,
                          "start_sec": 0.0, "end_sec": idx / 30.0, "duration_sec": idx / 30.0}],
        }
        with open(proc_dir / "episodes.json", "w") as f:
            json.dump(episodes, f)
        # No object_tracks.json written at all.

        result = run(session_id)
        ep = result["episodes"][0]
        assert ep["L1_task"] == "bolt_tightening"
        assert ep["dominant_object_class"] is None
        assert ep["object_bonus_applied"] == {}
