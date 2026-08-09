"""
DatraAI Pipeline — Tests for Step 09: Language Grounding
Tests the safe template-formatting helper (no I/O required), the
_ground_episode pure function, and run()'s per-episode orchestration
(v2 addendum §6).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

import config as cfg

_spec = importlib.util.spec_from_file_location(
    "language_ground",
    str(Path(__file__).resolve().parent.parent / "scripts" / "09_language_ground.py"),
)
_lg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lg_mod)

_SafeFormatDict = _lg_mod._SafeFormatDict
_ground_episode = _lg_mod._ground_episode
run = _lg_mod.run


class TestSafeFormatDict:
    """format_map with _SafeFormatDict must never raise KeyError."""

    def test_known_keys_substitute_normally(self):
        template = "Task {task} took {duration:.1f}s."
        result = template.format_map(_SafeFormatDict({"task": "bolt_tightening", "duration": 12.34}))
        assert result == "Task bolt_tightening took 12.3s."

    def test_missing_key_does_not_raise(self):
        template = "Grasp {grasp_type} object {object_class} at {target_location}."
        # Only grasp_type provided — object_class/target_location intentionally missing.
        result = template.format_map(_SafeFormatDict({"grasp_type": "power grasp"}))
        assert result == "Grasp power grasp object {object_class} at {target_location}."

    def test_fallback_template_with_extra_unused_fields_is_safe(self):
        """Fields not referenced by the template are simply unused, never an error."""
        template = "Perform {task} task using {dominant_hand} hand with {grasp_type}. Duration: {duration:.1f}s."
        fields = {
            "task": "idle",
            "dominant_hand": "right",
            "grasp_type": "power grasp",
            "duration": 2.0,
            "object_class": "the object",
            "target_location": "the target location",
            "outcome": "attempted",
            "success": False,
        }
        result = template.format_map(_SafeFormatDict(fields))
        assert "idle" in result and "right" in result


class TestGroundEpisodePureFunction:
    def test_dominant_hand_is_majority_across_episode_frames(self):
        episode = {"duration_sec": 10.0}
        task_entry = {"L1_task": "bolt_tightening", "confidence": 0.9}
        hand_pose_in_ep = [
            {"frame_idx": 0, "dominant_hand": "left"},
            {"frame_idx": 1, "dominant_hand": "right"},
            {"frame_idx": 2, "dominant_hand": "right"},
        ]
        result = _ground_episode(episode, task_entry, hand_pose_in_ep, [], validation=None)
        assert result["template_fields"]["dominant_hand"] == "right"

    def test_no_hand_pose_frames_defaults_to_right(self):
        episode = {"duration_sec": 5.0}
        task_entry = {"L1_task": "unknown", "confidence": 0.0}
        result = _ground_episode(episode, task_entry, [], [], validation=None)
        assert result["template_fields"]["dominant_hand"] == "right"

    def test_grasp_type_prefers_lateral_pinch_when_more_common(self):
        episode = {"duration_sec": 5.0}
        task_entry = {"L1_task": "label_apply", "confidence": 0.9}
        primitives_in_ep = [
            {"frame_idx": 0, "active_primitives": ["lateral_pinch"]},
            {"frame_idx": 1, "active_primitives": ["lateral_pinch"]},
            {"frame_idx": 2, "active_primitives": ["power_grasp"]},
        ]
        result = _ground_episode(episode, task_entry, [], primitives_in_ep, validation=None)
        assert result["template_fields"]["grasp_type"] == "pinch grasp"

    def test_grasp_type_defaults_to_power_grasp_on_tie_or_no_primitives(self):
        episode = {"duration_sec": 5.0}
        task_entry = {"L1_task": "bolt_tightening", "confidence": 0.9}
        result = _ground_episode(episode, task_entry, [], [], validation=None)
        assert result["template_fields"]["grasp_type"] == "power grasp"

    def test_success_requires_both_valid_and_confident(self):
        episode = {"duration_sec": 5.0}
        # High confidence but validation says invalid -> not success.
        task_entry = {"L1_task": "bolt_tightening", "confidence": 0.95}
        result = _ground_episode(episode, task_entry, [], [], validation={"overall_valid": False})
        assert result["template_fields"]["success"] is False

        # Valid but low confidence -> not success.
        task_entry_low_conf = {"L1_task": "bolt_tightening", "confidence": 0.4}
        result2 = _ground_episode(episode, task_entry_low_conf, [], [], validation={"overall_valid": True})
        assert result2["template_fields"]["success"] is False

        # Valid and confident -> success.
        result3 = _ground_episode(episode, task_entry, [], [], validation={"overall_valid": True})
        assert result3["template_fields"]["success"] is True

    def test_object_and_location_are_unflagged_placeholders(self):
        episode = {"duration_sec": 5.0}
        task_entry = {"L1_task": "bolt_tightening", "confidence": 0.9}
        result = _ground_episode(episode, task_entry, [], [], validation=None)
        assert result["object_grounded"] is False
        assert result["location_grounded"] is False


class TestRunPerEpisodeOrchestration:
    """
    A two-episode session where episode 0's frames indicate a right-handed
    power grasp and episode 1's frames indicate a left-handed pinch grasp —
    the per-episode instruction must reflect each episode's OWN frames, not
    a session-wide average across both.
    """

    def _write_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    def test_two_episodes_grounded_independently(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_lg_multi_ep"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

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
                {"episode_id": f"{session_id}_ep00", "L1_task": "bolt_tightening", "confidence": 0.9},
                {"episode_id": f"{session_id}_ep01", "L1_task": "label_apply", "confidence": 0.85},
            ],
        }
        hand_pose = [
            {"frame_idx": 0, "dominant_hand": "right"},
            {"frame_idx": 1, "dominant_hand": "right"},
            {"frame_idx": 100, "dominant_hand": "left"},
            {"frame_idx": 101, "dominant_hand": "left"},
        ]
        primitives = [
            {"frame_idx": 0, "active_primitives": ["power_grasp"]},
            {"frame_idx": 1, "active_primitives": ["power_grasp"]},
            {"frame_idx": 100, "active_primitives": ["lateral_pinch"]},
            {"frame_idx": 101, "active_primitives": ["lateral_pinch"]},
        ]

        self._write_json(proc_dir / "episodes.json", episodes)
        self._write_json(proc_dir / "task_label.json", task_label)
        self._write_json(proc_dir / "hand_pose.json", hand_pose)
        self._write_json(proc_dir / "primitives.json", primitives)

        result = run(session_id)

        assert len(result["episodes"]) == 2
        ep0, ep1 = result["episodes"]

        assert ep0["episode_id"] == f"{session_id}_ep00"
        assert ep0["template_fields"]["dominant_hand"] == "right"
        assert ep0["template_fields"]["grasp_type"] == "power grasp"
        assert ep0["template_fields"]["task"] == "bolt_tightening"

        assert ep1["episode_id"] == f"{session_id}_ep01"
        assert ep1["template_fields"]["dominant_hand"] == "left"
        assert ep1["template_fields"]["grasp_type"] == "pinch grasp"
        assert ep1["template_fields"]["task"] == "label_apply"

        # Cross-contamination check: episode 1's instruction is templated
        # from its own "left" hand, not episode 0's "right" hand.
        assert "left hand" in ep1["instruction"]
        assert "right hand" not in ep1["instruction"]

    def test_missing_optional_validation_report_defaults_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        session_id = "sess_lg_no_validation"
        proc_dir = tmp_path / session_id
        proc_dir.mkdir(parents=True)

        episodes = {
            "session_id": session_id,
            "episodes": [
                {"episode_id": f"{session_id}_ep00", "start_frame": 0, "end_frame": 29,
                 "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            ],
        }
        task_label = {
            "session_id": session_id,
            "episodes": [{"episode_id": f"{session_id}_ep00", "L1_task": "bolt_tightening", "confidence": 0.9}],
        }
        self._write_json(proc_dir / "episodes.json", episodes)
        self._write_json(proc_dir / "task_label.json", task_label)
        self._write_json(proc_dir / "hand_pose.json", [])
        self._write_json(proc_dir / "primitives.json", [])
        # No validation_report.json written -> run() must not crash.

        result = run(session_id)
        assert len(result["episodes"]) == 1
        # No validation report -> validation=None -> success falls back to
        # (validation_valid=True) AND confidence>0.7, per _ground_episode.
        assert result["episodes"][0]["template_fields"]["success"] is True


import anthropic
from utils import vlm_language as vlm_mod


class _FakeAnthropicClient:
    """Stand-in for anthropic.Anthropic() — never actually used since the
    vlm_language functions that would consume it are monkeypatched in
    every test below; exists only so `anthropic.Anthropic()` doesn't
    require real credentials during the test run."""


def _write_two_episode_session(tmp_path, monkeypatch, session_id, task_a="bolt_tightening", task_b="material_transfer"):
    monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
    proc_dir = tmp_path / session_id
    proc_dir.mkdir(parents=True)

    def _write(path, data):
        with open(path, "w") as f:
            json.dump(data, f)

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
            {"episode_id": f"{session_id}_ep00", "L1_task": task_a, "confidence": 0.9},
            {"episode_id": f"{session_id}_ep01", "L1_task": task_b, "confidence": 0.85},
        ],
    }
    _write(proc_dir / "episodes.json", episodes)
    _write(proc_dir / "task_label.json", task_label)
    _write(proc_dir / "hand_pose.json", [])
    _write(proc_dir / "primitives.json", [])
    _write(proc_dir / "phases.json", {"session_id": session_id, "segments": [
        {"phase": "grasp", "start_frame": 0, "end_frame": 49},
        {"phase": "grasp", "start_frame": 100, "end_frame": 149},
    ]})
    (proc_dir / "compressed.mp4").write_bytes(b"not a real video")
    return proc_dir


class TestVlmModeOrchestration:
    """
    v2 addendum §7 (revised) — VLM-hybrid language grounding. All tests
    monkeypatch the vlm_language module's API-calling functions (never hit
    the real network) so these are fast, deterministic unit tests; real
    end-to-end verification against the actual Claude API and real video
    frames happens separately against session_001.
    """

    def _fake_generate(self, instruction="A grounded instruction.", task_guess="bolt_tightening", confidence=0.9, objects=None):
        def _fn(client, structured_facts, frame_images_b64, recent_instructions):
            return {
                "parsed": {
                    "instruction": instruction,
                    "task_guess": task_guess,
                    "task_guess_confidence": confidence,
                    "objects_mentioned": objects or ["bolt"],
                },
                "prompt_system": "fake system prompt",
                "raw_response_text": json.dumps({"instruction": instruction}),
                "usage": {"input_tokens": 2000, "output_tokens": 100},
                "model": cfg.VLM_MODEL,
            }
        return _fn

    def _fake_encode(self, monkeypatch):
        monkeypatch.setattr(vlm_mod, "encode_frame_base64", lambda video_path, idx: "ZmFrZQ==")

    def test_template_mode_makes_no_vlm_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "template")
        session_id = "sess_lg_template_mode"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        result = run(session_id)
        assert all(e["generation_method"] == "template" for e in result["episodes"])
        assert "vlm_cost_summary" not in result

    def test_vlm_mode_success_populates_audit_and_cost(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "vlm")
        monkeypatch.setattr(cfg, "VLM_HALLUCINATION_SPOTCHECK_RATE", 0.0)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", self._fake_generate())
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_vlm_success"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        result = run(session_id)
        assert all(e["generation_method"] == "vlm" for e in result["episodes"])
        ep0 = result["episodes"][0]
        assert ep0["vlm_audit"]["frames_sampled"]
        assert ep0["vlm_audit"]["cost_usd"] > 0
        assert ep0["vlm_audit"]["model"] == cfg.VLM_MODEL
        assert "vlm_cost_summary" in result
        assert result["vlm_cost_summary"]["vlm_calls"] == 2

    def test_vlm_mode_raises_on_api_failure_no_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "vlm")
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())

        def _raise(*a, **kw):
            raise RuntimeError("simulated API failure")
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", _raise)
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_vlm_raises"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        with pytest.raises(RuntimeError):
            run(session_id)

    def test_hybrid_mode_falls_back_to_template_on_api_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "hybrid")
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())

        def _raise(*a, **kw):
            raise RuntimeError("simulated API failure")
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", _raise)
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_hybrid_fallback"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        result = run(session_id)  # must NOT raise
        assert all(e["generation_method"] == "template_fallback" for e in result["episodes"])
        assert all("vlm_fallback_reason" in e for e in result["episodes"])
        assert "vlm_cost_summary" not in result  # no successful VLM calls were made

    def test_task_classification_disagreement_surfaced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "vlm")
        monkeypatch.setattr(cfg, "VLM_HALLUCINATION_SPOTCHECK_RATE", 0.0)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())
        # VLM independently guesses "material_transfer" while the classifier said "bolt_tightening" for episode 0.
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", self._fake_generate(task_guess="material_transfer"))
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_disagreement"
        _write_two_episode_session(tmp_path, monkeypatch, session_id, task_a="bolt_tightening")

        result = run(session_id)
        ep0 = result["episodes"][0]
        assert ep0["task_classification_disagreement"]["disagree"] is True
        assert ep0["task_classification_disagreement"]["classifier_task"] == "bolt_tightening"
        assert ep0["task_classification_disagreement"]["vlm_task_guess"] == "material_transfer"

    def test_hallucination_spotcheck_rate_zero_never_checks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "vlm")
        monkeypatch.setattr(cfg, "VLM_HALLUCINATION_SPOTCHECK_RATE", 0.0)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", self._fake_generate())
        calls = []
        monkeypatch.setattr(vlm_mod, "check_instruction_hallucination", lambda *a, **kw: calls.append(1))
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_no_spotcheck"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        result = run(session_id)
        assert len(calls) == 0
        assert all(e["hallucination_check"]["checked"] is False for e in result["episodes"])

    def test_hallucination_spotcheck_rate_one_always_checks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "vlm")
        monkeypatch.setattr(cfg, "VLM_HALLUCINATION_SPOTCHECK_RATE", 1.0)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", self._fake_generate())

        def _fake_hallucination(client, instruction_text, frame_images_b64):
            return {
                "parsed": {"consistent": True, "unsupported_claims": []},
                "usage": {"input_tokens": 1500, "output_tokens": 50},
            }
        monkeypatch.setattr(vlm_mod, "check_instruction_hallucination", _fake_hallucination)
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_always_spotcheck"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        result = run(session_id)
        for e in result["episodes"]:
            assert e["hallucination_check"]["checked"] is True
            assert e["hallucination_check"]["consistent"] is True
            assert e["hallucination_check"]["cost_usd"] > 0
        # The hallucination-check cost must be folded into the episode's total audit cost.
        assert result["episodes"][0]["vlm_audit"]["cost_usd"] > result["episodes"][0]["hallucination_check"]["cost_usd"]

    def test_recent_instructions_passed_to_next_episode(self, tmp_path, monkeypatch):
        """Anti-repetition: the second episode's call must see the first episode's instruction in recent_instructions."""
        monkeypatch.setattr(cfg, "LANGUAGE_GEN_MODE", "vlm")
        monkeypatch.setattr(cfg, "VLM_HALLUCINATION_SPOTCHECK_RATE", 0.0)
        monkeypatch.setattr(anthropic, "Anthropic", lambda: _FakeAnthropicClient())
        seen_recent = []

        def _fn(client, structured_facts, frame_images_b64, recent_instructions):
            seen_recent.append(list(recent_instructions))
            return {
                "parsed": {
                    "instruction": f"Instruction for {structured_facts['task']}.",
                    "task_guess": structured_facts["task"],
                    "task_guess_confidence": 0.9,
                    "objects_mentioned": [],
                },
                "prompt_system": "sys",
                "raw_response_text": "{}",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "model": cfg.VLM_MODEL,
            }
        monkeypatch.setattr(vlm_mod, "generate_instruction_vlm", _fn)
        self._fake_encode(monkeypatch)
        session_id = "sess_lg_recent_instructions"
        _write_two_episode_session(tmp_path, monkeypatch, session_id)

        run(session_id)
        assert seen_recent[0] == []  # first episode has no prior instructions yet
        assert len(seen_recent[1]) == 1
        assert "bolt_tightening" in seen_recent[1][0]
