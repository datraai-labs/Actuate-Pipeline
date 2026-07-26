from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")


def _video(path, n=9, width=96, height=64):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    for i in range(n):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (10 + i * 5, 30), 5, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_limited_visual_sampling_keeps_full_frame_indexed_imu_poses(tmp_path):
    n = 9
    _video(tmp_path / "video.mp4", n=n)
    (tmp_path / "session_meta.json").write_text(
        json.dumps(
            {
                "frame_count": n,
                "fps_nominal": 30.0,
                "video_width": 96,
                "video_height": 64,
            }
        )
    )
    with h5py.File(tmp_path / "session.h5", "w") as h5:
        imu = h5.create_group("imu")
        imu.create_dataset("gyro", data=np.tile([0.0, 0.0, 0.2], (n, 1)))

    from actuate.config import Provenance
    from actuate.perception.slam.runner import run

    result = run(tmp_path, max_frames=3)

    assert len(result.poses) == n
    assert len(result.rotation_rad) == n
    assert result.provenance["camera_pose.rotation"] is Provenance.MEASURED_HUMAN
    # The last evenly sampled perception frame has a real, integrated pose at its own index.
    assert result.poses[-1].quaternion_wxyz != (1.0, 0.0, 0.0, 0.0)
