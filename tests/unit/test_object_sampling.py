"""Object perception must share the same sparse source-frame clock as hands and depth."""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _video(path, n=20, width=64, height=48):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    for i in range(n):
        writer.write(np.full((height, width, 3), i, dtype=np.uint8))
    writer.release()


def test_object_run_keeps_original_evenly_sampled_frame_ids(tmp_path, monkeypatch):
    from actuate.perception.objects import objects as mod
    from actuate.perception.sampling import sampled_indices

    n, cap = 20, 4
    _video(tmp_path / "video.mp4", n=n)
    (tmp_path / "session_meta.json").write_text(json.dumps({
        "session_id": "s",
        "frame_count": n,
        "fps_nominal": 30.0,
        "video_width": 64,
        "video_height": 48,
    }))

    class Detector:
        def __init__(self, device):
            pass

        def detect(self, image, prompts):
            return [{"box": [8, 8, 24, 24], "label": "cup", "score": 0.9}]

    class Tracker:
        def __init__(self, device):
            pass

        def propagate(self, frames, box):
            mask = np.zeros((48, 64), dtype=bool)
            mask[8:24, 8:24] = True
            return {i: mask for i in range(len(frames))}

    monkeypatch.setattr(mod, "GroundingDinoDetector", Detector)
    monkeypatch.setattr(mod, "Sam2Tracker", Tracker)

    result = mod.run(tmp_path, prompts=["cup"], max_frames=cap, chunk=cap)
    assert sorted(result.frames) == sampled_indices(n, cap)
    assert result.n_frames == cap
