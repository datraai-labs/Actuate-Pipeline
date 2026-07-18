"""The HaMeR fallback seam: when WiLoR's detector finds 0 hands, `hands.run` re-scans with
HaMeR and recovers them; and a missing/failing HaMeR degrades to a note, never a crash.

HaMeR itself needs detectron2/ViTPose/weights that only install on the GPU box, so we inject
stub estimators through the `estimator_for` seam -- the ORCHESTRATION (does the fallback fire,
is provenance recorded, does it degrade safely) is what these tests pin down."""

from __future__ import annotations

import json

import numpy as np
import pytest

from actuate.config import Side
from actuate.perception import hands as handsmod
from actuate.perception.hands.wilor import HandFrame

cv2 = pytest.importorskip("cv2")


def _hand_frame(side=Side.RIGHT):
    return HandFrame(
        side=side, betas=np.zeros(10), hand_pose=np.zeros((15, 3)),
        global_orient=np.zeros(3), keypoints_3d=np.zeros((21, 3)),
        keypoints_2d=np.zeros((21, 2)), root_translation_virtual=np.zeros(3),
        virtual_focal=5000.0, bbox=np.zeros(4), detection_confidence=0.9)


class _Stub:
    """A hand estimator that returns a fixed detection (or nothing) for every frame."""

    def __init__(self, finds: bool):
        self._finds = finds

    def predict(self, frame_bgr):
        return [_hand_frame()] if self._finds else []


@pytest.fixture()
def session(tmp_path):
    """A 6-frame session dir the scan loop can actually open + seek."""
    s = tmp_path / "sess"
    s.mkdir()
    vid = s / "compressed.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    for _ in range(6):
        vw.write(np.zeros((32, 32, 3), np.uint8))
    vw.release()
    (s / "session_meta.json").write_text(json.dumps({"frame_count": 6}))
    return s


def test_fallback_fires_when_primary_finds_nothing(session, monkeypatch):
    # WiLoR (primary) finds 0; HaMeR (fallback) finds hands.
    def _estimator_for(model, device):
        return _Stub(finds=(model == "hamer"))

    monkeypatch.setattr(handsmod.wilor, "estimator_for", _estimator_for)
    res = handsmod.run(session, prefilter=False, fallback="hamer")
    assert res.n_with_hands > 0                       # recovered by the fallback
    assert res.notes["hand_model"] == "hamer"         # provenance says which model
    assert "re-scanned with hamer" in res.notes["fallback"]


def test_no_fallback_when_primary_succeeds(session, monkeypatch):
    calls = []

    def _estimator_for(model, device):
        calls.append(model)
        return _Stub(finds=(model == "wilor"))         # primary already works

    monkeypatch.setattr(handsmod.wilor, "estimator_for", _estimator_for)
    res = handsmod.run(session, prefilter=False, fallback="hamer")
    assert res.n_with_hands > 0
    assert calls == ["wilor"]                          # fallback never constructed
    assert res.notes["hand_model"] == "wilor"


def test_missing_fallback_degrades_to_note_not_crash(session, monkeypatch):
    def _estimator_for(model, device):
        if model == "hamer":
            raise ImportError("hamer not installed on this box")
        return _Stub(finds=False)                      # primary also finds nothing

    monkeypatch.setattr(handsmod.wilor, "estimator_for", _estimator_for)
    res = handsmod.run(session, prefilter=False, fallback="hamer")   # must NOT raise
    assert res.n_with_hands == 0
    assert "fallback unavailable" in res.notes["fallback"]


def test_estimator_for_rejects_unknown_model():
    with pytest.raises(NotImplementedError, match="not implemented"):
        handsmod.estimator_for("mediapipe", "cpu")
