"""Part A plumbing tests -- no network, no key. The client is INJECTED (the seam v1 built
and this port kept), so a fake client exercises every code path. The judge's real
red->green (a hallucinated caption against real frames) runs in the billed gate script;
here we prove the FLAGGING machinery: sub-threshold scores flag, good scores don't, and a
missing key skips instead of failing."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

# the package re-exports the `annotate` FUNCTION under the same name as the module, so an
# `import ... as` would bind the function; importlib reaches the module itself
lang_annotate = importlib.import_module("actuate.language.annotate")

from actuate.config import RigType
from actuate.language import vlm
from actuate.language.annotate import ConsistencyScore
from actuate.schema import CanonicalEpisode


class _FakeClient:
    """Returns canned structured-output responses per schema, records every request."""

    def __init__(self, responses: dict[str, dict]):
        self._responses = responses     # keyed by schema title-ish: paraphrase/caption/judge
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        schema = kw["output_config"]["format"]["schema"]
        if "paraphrases" in schema["properties"]:
            payload = self._responses["paraphrase"]
        elif "instruction" in schema["properties"]:
            payload = self._responses["caption"]
        else:
            payload = self._responses["judge"]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )


_GOOD_JUDGE = {"hand_consistency": 0.95, "object_consistency": 0.9,
               "action_consistency": 0.9, "global_consistency": 0.92,
               "unsupported_claims": [], "verdict": "consistent"}
_BAD_JUDGE = {"hand_consistency": 0.9, "object_consistency": 0.2,
              "action_consistency": 0.85, "global_consistency": 0.5,
              "unsupported_claims": ["a red stapler"], "verdict": "inconsistent"}
_CAPTION = {"hand": "right hand flat over paper", "object": "papers on a desk",
            "action": "sorting papers", "scene": "office workbench",
            "instruction": "Sort the papers on the desk.", "task_guess": "sort the papers"}
_PARAS = {"paraphrases": ["Arrange the documents on the desk.",
                          "Put the desk's papers in order.",
                          "Organize the paperwork lying on the desk."]}


def _episode(task="sort the papers"):
    return CanonicalEpisode(episode_id="ep", capture_id="c" * 64,
                            rig=RigType.HEAD_MOUNTED, task=task)


# ---------------------------------------------------------------- key handling
def test_no_key_degrades_to_rule_based_never_raises(monkeypatch, tmp_path):
    """No key: NOT a hard skip -- rule-based paraphrases still ship (Part B fallback);
    only the VLM-dependent captions/judge are skipped."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(vlm, "make_client", lambda api_key=None: None)
    report = lang_annotate.annotate(_episode(task="sort the papers"),
                                    video=tmp_path / "missing.mp4")
    assert not report.skipped                       # degraded, not skipped
    assert len(report.paraphrases) >= 3             # rule-based variants present
    assert "rule-based" in report.skip_reason       # reason still explains the degradation
    assert report.subtasks == []                    # VLM parts skipped


def test_get_api_key_never_returns_empty(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    # empty env var must not count as a key (it would authenticate with an empty key)
    assert vlm.get_api_key() != ""


# ---------------------------------------------------------------- paraphrase
def test_paraphrase_uses_the_injected_client():
    fake = _FakeClient({"paraphrase": _PARAS})
    out = lang_annotate.paraphrase("sort the papers", n=3, client=fake)
    assert len(out) == 3
    assert len(set(out)) == 3                      # distinct
    assert fake.calls[0]["model"] == vlm.VLM_MODEL
    # no sampling params -- removed on the current model family, 400 if sent
    assert "temperature" not in fake.calls[0]


# ---------------------------------------------------------------- judge flagging
def test_judge_flags_a_hallucinated_caption():
    """The FLAGGING half of gate 3: a low object score + unsupported claim must flag."""
    fake = _FakeClient({"judge": _BAD_JUDGE})
    score = lang_annotate.judge(_CAPTION, ["b64"], {}, client=fake)
    assert score.flagged
    assert "a red stapler" in score.unsupported_claims
    assert score.min_score == pytest.approx(0.2)


def test_judge_does_not_flag_a_grounded_caption():
    """Broken-variant guard: a judge that flags everything is as useless as one that
    flags nothing."""
    fake = _FakeClient({"judge": _GOOD_JUDGE})
    score = lang_annotate.judge(_CAPTION, ["b64"], {}, client=fake)
    assert not score.flagged


def test_judge_scores_are_clamped():
    wild = dict(_GOOD_JUDGE, hand_consistency=1.7, object_consistency=-0.3)
    fake = _FakeClient({"judge": wild})
    score = lang_annotate.judge(_CAPTION, ["b64"], {}, client=fake)
    assert score.hand == 1.0 and score.object == 0.0


def test_consistency_score_verdict_alone_can_flag():
    s = ConsistencyScore(hand=0.9, object=0.9, action=0.9, global_=0.9,
                         unsupported_claims=[], verdict="inconsistent")
    assert s.flagged


# ---------------------------------------------------------------- annotate orchestration
def test_annotate_refuses_to_paraphrase_a_missing_task(tmp_path):
    fake = _FakeClient({"paraphrase": _PARAS})
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    report = lang_annotate.annotate(_episode(task=None), video=v, client=fake)
    assert report.skipped
    assert "invent a label" in report.skip_reason


def test_frame_sampling_covers_start_interior_end():
    frames = vlm.sample_representative_frames(0, 300, [
        {"phase": "grasp", "start_frame": 50, "end_frame": 100},
        {"phase": "active_manipulation", "start_frame": 100, "end_frame": 250},
    ])
    assert frames[0] == 0 and frames[-1] == 300
    assert vlm.MIN_SAMPLE_FRAMES <= len(frames) <= vlm.MAX_SAMPLE_FRAMES
    assert any(0 < f < 300 for f in frames)


def test_cost_estimate_is_positive_and_scales():
    one = vlm.estimate_annotation_cost(1)
    ten = vlm.estimate_annotation_cost(10)
    assert 0 < one < ten


def test_independent_task_read_disagreement_is_fail_visible():
    assert not vlm.task_disagreement("sort the papers", "sort the papers")
    assert vlm.task_disagreement("tighten the bolt", "sort the papers")
