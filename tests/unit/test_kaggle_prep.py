"""The Kaggle runner's session normalization: raw dataset -> writable processed session.

The user's real dataset mounts read-only with filenames like `video (1).mp4` and no
`session_meta.json`. This pins the two behaviours that must hold on that exact shape: raw
filenames get normalised, and a valid `session_meta.json` is synthesised from the video (frame
count/fps/resolution) with no manual editing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_RUNNER = Path(__file__).resolve().parents[2] / "kaggle" / "run_perception.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("kaggle_run_perception", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_video(path: Path, n=6, w=64, h=48, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not vw.isOpened():
        pytest.skip("no mp4 writer codec available in this environment")
    for i in range(n):
        vw.write(np.full((h, w, 3), i * 10, dtype=np.uint8))
    vw.release()


def test_raw_kaggle_dataset_is_normalised_and_meta_synthesised(tmp_path):
    runner = _load_runner()

    raw = tmp_path / "session-001"
    raw.mkdir()
    _tiny_video(raw / "video (1).mp4", n=6, w=64, h=48, fps=30)
    for name in ["motion (1).json", "timestamps (1).json",
                 "camera_intrinsic (1).json", "metadata (2).json"]:
        (raw / name).write_text('{"dummy": true}')

    assert runner._looks_processed(raw) is False

    sess = runner.prepare_session(raw, work_root=tmp_path / "work")

    # filenames normalised
    for expected in ["compressed.mp4", "motion.json", "timestamps.json",
                     "camera_intrinsics.json", "metadata.json", "session_meta.json"]:
        assert (sess / expected).exists(), f"missing {expected}"

    # session_meta.json synthesised from the video
    meta = json.loads((sess / "session_meta.json").read_text())
    assert meta["session_id"] == "session_001"
    assert meta["frame_count"] >= 1              # decoded from the actual file
    assert meta["fps_nominal"] > 0
    assert meta["resolution"][0] > 0 and meta["resolution"][1] > 0

    # the normalised dir now reads as a processed session
    assert runner._looks_processed(sess) is True


def test_already_processed_session_is_used_in_place(tmp_path):
    runner = _load_runner()

    proc = tmp_path / "session_001"
    proc.mkdir()
    _tiny_video(proc / "compressed.mp4", n=4)
    (proc / "session_meta.json").write_text(
        json.dumps({"session_id": "session_001", "frame_count": 4, "fps_nominal": 30})
    )
    assert runner._looks_processed(proc) is True

    out = runner.prepare_session(proc, work_root=tmp_path / "work")
    assert out == proc.resolve()                 # writable + processed -> in place, no copy
