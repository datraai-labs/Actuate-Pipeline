"""The egocentric-RGBD adapter: native metric 3D hand keypoints -> canonical schema, with
real handedness/keypoints and the dataset's own task label. Mocked HF -- no download."""

from __future__ import annotations

import json

import numpy as np
import pytest

from actuate.config import Side
from actuate.sources.adapters import egocentric_rgbd as adp

cv2 = pytest.importorskip("cv2")


def _kp21(z=0.5):
    # 21 joints in a plausible hand layout (metres, camera frame)
    kp = [[0.0, 0.0, z]]                      # wrist
    for f in range(5):
        for j in range(4):
            kp.append([0.02 * f, 0.01 * (j + 1), z])
    return kp


@pytest.fixture()
def fake_hf(tmp_path, monkeypatch):
    # dense keypoints jsonl: 5 frames, both hands, real joints_3d_camera
    kp = tmp_path / "kp.jsonl"
    with kp.open("w") as fh:
        for i in range(5):
            fh.write(json.dumps({
                "global_frame_index": i, "time_s": i / 30.0, "hand_count": 2,
                "hands": [
                    {"handedness": "right", "detector_confidence": 0.9,
                     "joints_3d_camera": _kp21(0.5 + 0.01 * i)},
                    {"handedness": "left", "detector_confidence": 0.8,
                     "joints_3d_camera": _kp21(0.6)},
                ]}) + "\n")
    seg = tmp_path / "seg.jsonl"
    seg.write_text(json.dumps({"task_description_en": "apply hand cream",
                               "start_rgb_frame": 0, "end_rgb_frame": 4,
                               "subtask_label_en": "rub hands"}) + "\n")
    vid = tmp_path / "overlay.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    for _ in range(5):
        vw.write(np.zeros((64, 64, 3), np.uint8))
    vw.release()

    def _fake(repo_id, rel):
        from pathlib import Path
        if rel.endswith(".jsonl") and "keypoint" in rel:
            return kp
        if "segment" in rel:
            return seg
        return Path(vid)

    monkeypatch.setattr(adp, "_hf", _fake)


def test_adapter_builds_canonical_with_real_keypoints(tmp_path, fake_hf):
    ep, session = adp.adapt("test-pkg", tmp_path / "work")
    assert len(ep.frames) == 5
    # both hands present, 21 metric keypoints each
    f0 = ep.frames[0]
    assert set(f0.hands) == {Side.RIGHT, Side.LEFT}
    assert len(f0.hands[Side.RIGHT].keypoints_3d) == 21
    assert f0.hands[Side.RIGHT].wrist_pose is not None      # derived from keypoints
    assert ep.task == "apply hand cream"                     # dataset's own label
    assert (session / "session_meta.json").exists()          # video staged + meta


def test_adapter_carries_subtasks_from_segments(tmp_path, fake_hf):
    ep, _ = adp.adapt("test-pkg", tmp_path / "work")
    assert len(ep.subtasks) == 1
    assert ep.subtasks[0].instruction == "rub hands"


def test_adapter_records_provenance_source(tmp_path, fake_hf):
    ep, _ = adp.adapt("test-pkg", tmp_path / "work")
    assert "source" in ep.derivation_notes
    assert "not our WiLoR" in ep.derivation_notes["source"]  # honest: not our perception


def test_adapter_refuses_when_no_hands(tmp_path, monkeypatch):
    empty = tmp_path / "kp.jsonl"
    empty.write_text(json.dumps({"global_frame_index": 0, "hands": []}) + "\n")
    vid = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    vw.write(np.zeros((32, 32, 3), np.uint8))
    vw.release()
    monkeypatch.setattr(adp, "_hf",
                        lambda r, rel: empty if rel.endswith("keypoints.jsonl") else vid)
    with pytest.raises(ValueError, match="no frames with 21-joint hands"):
        adp.adapt("test-pkg", tmp_path / "work")
