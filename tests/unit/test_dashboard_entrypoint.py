"""The Actuate-dashboard bridge maps a real CanonicalEpisode onto the dashboard worker's
per-episode JSON contract (quality_certificate.json + language_grounding.json +
phase_segmentation.json). These pin the field names the dashboard's _ingest_episodes reads,
and the honest handling of the 0-hands case (needs_review, not a fabricated clean episode)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dashboard_entrypoint",
                                               REPO / "dashboard_entrypoint.py")
dash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dash)


def _fake_episode(*, n_frames: int, quality: int, subtasks=(), retarget=None):
    comps = SimpleNamespace(sync_integrity=0.9, calibration_completeness=0.3,
                            perception_confidence=0.8, contact_consistency=None,
                            ik_convergence_rate=0.84)
    meta = SimpleNamespace(quality=quality, speed="normal", mistakes=[], components=comps)
    return SimpleNamespace(
        episode_id="cap0_ep00", task="pick up the cube", task_paraphrases=["grab the cube"],
        frames=list(range(n_frames)), subtasks=list(subtasks),
        retarget_eligibility=retarget or {},
        consent=SimpleNamespace(value="granted"),
        pii_status=SimpleNamespace(value="pending"),
        is_deliverable=False, episode_meta=meta)


def test_zero_hands_writes_needs_review_not_a_fake_episode(tmp_path):
    dash._write_episode_outputs(_fake_episode(n_frames=0, quality=1), tmp_path)
    cert = json.loads((tmp_path / "cap0_ep00" / "quality_certificate.json").read_text())
    assert cert["needs_human_review"] is True
    assert cert["retargeting_eligible"] == "needs_review"
    assert cert["n_frames"] == 0
    assert cert["eis_score"] == 20.0                       # 1/5 -> 20, not a fake high score
    assert "0 hands" in cert["note"]                       # honest, human-readable reason


def test_good_episode_maps_to_training_recommendation(tmp_path):
    sub = SimpleNamespace(instruction="reach for cube", start_frame=0, end_frame=10,
                          confidence=0.9)
    ep = _fake_episode(n_frames=30, quality=4, subtasks=[sub],
                       retarget={"franka_panda": True})
    dash._write_episode_outputs(ep, tmp_path)
    d = tmp_path / "cap0_ep00"
    cert = json.loads((d / "quality_certificate.json").read_text())
    assert cert["needs_human_review"] is False
    assert cert["retargeting_eligible"] == "eligible"
    assert cert["recommended_use"] == "training"
    # the two side files the dashboard also reads
    lang = json.loads((d / "language_grounding.json").read_text())
    assert lang["language_instruction"] == "pick up the cube"
    phases = json.loads((d / "phase_segmentation.json").read_text())["phases"]
    assert phases == [{"phase": "reach for cube", "start_frame": 0, "end_frame": 10,
                       "confidence": 0.9}]


def test_certificate_has_every_field_the_worker_reads(tmp_path):
    # _ingest_episodes reads exactly these keys -- guard them all so a rename can't break it.
    dash._write_episode_outputs(_fake_episode(n_frames=5, quality=3), tmp_path)
    cert = json.loads((tmp_path / "cap0_ep00" / "quality_certificate.json").read_text())
    for key in ("episode_id", "task_label", "task_confidence", "eis_score",
                "retargeting_eligible", "recommended_use", "object_class",
                "needs_human_review", "task_classification_disagreement", "confidence_tree"):
        assert key in cert, f"worker reads {key!r}; bridge must emit it"


def test_reporter_advances_timeline_only_on_terminal_events(capsys):
    reporter = dash._make_reporter()
    reporter("perceive", "flag", "slam unavailable")     # mid-stage note -> no token
    reporter("perceive", "done", "15 frames")            # terminal -> emits 03/04/05 once
    out = capsys.readouterr().out
    assert out.count("Running stage: 04_hand_pose") == 1  # exactly once, not per-flag
    assert "04_hand_pose completed" in out
