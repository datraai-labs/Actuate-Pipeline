"""Part A gates: closed-vocab action intervals, no single-frame rows, actor attribution,
bimanual overlapping rows, and the mislabel red->green (fake VLM client, $0)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from actuate.config import ActionVerb, Actor, InteractionState, RigType, Side
from actuate.language.actions import (
    MIN_INTERVAL_FRAMES,
    label_actions,
    verify_vlm_label,
)
from actuate.schema import CanonicalEpisode
from actuate.schema.frame import CanonicalFrame, HandState


def _kp(x, y, z, curl=0.0):
    """21x3 keypoints: wrist at (x,y,z); a uniform curl bends fingertips toward the palm."""
    kp = np.zeros((21, 3))
    kp[0] = [x, y, z]
    # fingertips (MediaPipe tips 4,8,12,16,20) extend +Z when open, retract with curl
    for tip in (4, 8, 12, 16, 20):
        kp[tip] = [x, y, z + 0.09 * (1.0 - curl)]
    return tuple(tuple(map(float, p)) for p in kp)


def _frame(i, *, sides, grasped=False, t=None):
    hands = {s: HandState(keypoints_3d=kp) for s, kp in sides.items()}
    return CanonicalFrame(
        t=(i / 30.0 if t is None else t), rig=RigType.HEAD_MOUNTED, episode_id="ep",
        frame_idx=i, hands=hands,
        interaction_state=InteractionState.GRASPED_R if grasped else InteractionState.STATIC,
        provenance={"hands": "vision_primary", "interaction_state": "vision_primary"},
        confidence={"hands": 0.9})


def _episode(frames):
    return CanonicalEpisode(episode_id="ep", capture_id="c" * 64,
                            rig=RigType.HEAD_MOUNTED, frames=tuple(frames))


# ---------------------------------------------------------------- gate 1: intervals produced
def test_intervals_use_only_closed_vocabulary():
    frames = [_frame(i, sides={Side.RIGHT: _kp(0.1 * i, 0.0, 0.5)}) for i in range(20)]
    res = label_actions(_episode(frames))
    assert res.intervals
    assert all(isinstance(iv.action_label, ActionVerb) for iv in res.intervals)
    assert all(iv.action_label in set(ActionVerb) for iv in res.intervals)


def test_static_hand_is_idle():
    frames = [_frame(i, sides={Side.RIGHT: _kp(0.3, 0.0, 0.5)}) for i in range(15)]
    res = label_actions(_episode(frames))
    assert {iv.action_label for iv in res.intervals} == {ActionVerb.IDLE}


# ---------------------------------------------------------------- gate 2: no single-frame
def test_no_single_frame_intervals():
    # a noisy alternating trajectory that would flicker without min-dwell
    frames = []
    for i in range(30):
        x = 0.5 if i % 2 else 0.0
        frames.append(_frame(i, sides={Side.RIGHT: _kp(x, 0.0, 0.5)}))
    res = label_actions(_episode(frames))
    non_transition = [iv for iv in res.intervals
                      if iv.action_label not in (ActionVerb.GRASP, ActionVerb.RELEASE)]
    assert all(iv.end_frame - iv.start_frame + 1 >= MIN_INTERVAL_FRAMES
               for iv in non_transition), [
        (iv.action_label.value, iv.start_frame, iv.end_frame) for iv in non_transition]


# ---------------------------------------------------------------- gate 3: actor attribution
def test_actor_is_the_hand_that_acts():
    frames = [_frame(i, sides={Side.RIGHT: _kp(0.3, 0.0, 0.5)}) for i in range(12)]
    res = label_actions(_episode(frames))
    assert {iv.actor for iv in res.intervals} == {Actor.RIGHT_HAND}
    left = [_frame(i, sides={Side.LEFT: _kp(0.3, 0.0, 0.5)}) for i in range(12)]
    assert {iv.actor for iv in label_actions(_episode(left)).intervals} == {Actor.LEFT_HAND}


# ---------------------------------------------------------------- gate 4: bimanual overlap
def test_concurrent_bimanual_actions_are_separate_overlapping_rows():
    """Left holds still (idle) while right moves -- must be two actor rows, not merged."""
    frames = []
    for i in range(20):
        frames.append(_frame(i, sides={
            Side.LEFT: _kp(0.0, 0.0, 0.5),          # still -> idle
            Side.RIGHT: _kp(0.05 * i, 0.0, 0.5)}))   # moving
    res = label_actions(_episode(frames))
    actors = {iv.actor for iv in res.intervals}
    assert actors == {Actor.LEFT_HAND, Actor.RIGHT_HAND}
    # at least one left interval and one right interval overlap in frame span
    lefts = [iv for iv in res.intervals if iv.actor == Actor.LEFT_HAND]
    rights = [iv for iv in res.intervals if iv.actor == Actor.RIGHT_HAND]
    assert any(l.start_frame <= r.end_frame and r.start_frame <= l.end_frame
               for l in lefts for r in rights)


# ---------------------------------------------------------------- gate 5: mislabel red->green
class _FakeVLM:
    def __init__(self, agrees: bool, observed="idle"):
        self._agrees, self._observed = agrees, observed
        self.messages = self

    def create(self, **kw):
        payload = {"agrees": self._agrees, "observed": self._observed}
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10))


def test_vlm_catches_a_mislabeled_interval():
    """An idle segment labelled 'pour' must be rejected by the VLM check (red->green)."""
    caught = verify_vlm_label(_FakeVLM(agrees=False, observed="idle"),
                              ActionVerb.POUR, "b64frame")
    assert caught is False                     # VLM disagrees -> the label is wrong


def test_vlm_confirms_a_correct_interval():
    """Broken-variant guard: a VLM that rejects everything is useless."""
    assert verify_vlm_label(_FakeVLM(agrees=True), ActionVerb.HOLD, "b64frame") is True


# ---------------------------------------------------------------- gate 6: joinable by frame
def test_intervals_are_joinable_to_frames_by_index():
    frames = [_frame(i, sides={Side.RIGHT: _kp(0.05 * i, 0.0, 0.5)}) for i in range(20)]
    ep = label_actions(_episode(frames)).episode
    idx = {f.frame_idx for f in ep.frames}
    for iv in ep.action_intervals:
        assert iv.start_frame in idx and iv.end_frame in idx
        assert iv.end_frame >= iv.start_frame
    # schema carries them, still valid
    assert CanonicalEpisode.model_validate(json.loads(ep.model_dump_json())).action_intervals


def test_result_attaches_to_episode_and_flags_low_confidence():
    frames = [_frame(i, sides={Side.RIGHT: _kp(0.05 * i, 0.0, 0.5)}) for i in range(20)]
    res = label_actions(_episode(frames))
    assert res.episode.action_intervals == tuple(res.intervals)
    assert res.flagged_for_review == sum(1 for iv in res.intervals if iv.confidence <= 0.5)


def test_pipeline_refuses_atomic_labels_for_sparse_demo_frames(tmp_path):
    from actuate.pipeline.run import _Ctx, _stage_label_actions

    # Ten representative frames over two minutes are useful for an end-to-end smoke test,
    # but not enough to infer atomic reach/grasp/pour boundaries honestly.
    frames = [
        _frame(i * 400, t=i * 400 / 30.0, sides={Side.RIGHT: _kp(0.1 * i, 0.0, 0.5)})
        for i in range(10)
    ]
    out = tmp_path / "run"
    out.mkdir()
    canonical_path = out / "canonical.json"
    canonical_path.write_text(_episode(frames).model_dump_json(), encoding="utf-8")
    ctx = _Ctx(
        session=tmp_path,
        out=out,
        profile={},
        reporter=lambda *args: None,
        confirm=lambda prompt: False,
    )
    ctx.canonical_path = canonical_path

    _stage_label_actions(ctx)

    assert ctx.checkpoint["label_actions"]["status"] == "skipped"
    assert "sparse sampling" in ctx.checkpoint["label_actions"]["note"]
    assert not CanonicalEpisode.model_validate_json(
        canonical_path.read_text(encoding="utf-8")
    ).action_intervals
