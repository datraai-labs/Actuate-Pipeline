"""Phase 6 Part D: auto rig-detect from real video geometry, layout detection, filename
normalization, and the local-consent default (delivery gate must still block)."""

from __future__ import annotations

import numpy as np
import pytest

from actuate.config import ConsentStatus, RigType
from actuate.sources.detect import (
    detect_layout,
    detect_rig,
    local_consent_default,
    normalize_filenames,
)

cv2 = pytest.importorskip("cv2")


def _write_video(path, w, h, n=5):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


# ---------------------------------------------------------------- rig from geometry
def test_widescreen_side_by_side_is_stereo(tmp_path):
    _write_video(tmp_path / "clip.mp4", 2560, 720)      # ~3.55:1 -> side-by-side stereo
    assert detect_layout(tmp_path) == "stereo_video"
    assert detect_rig(tmp_path) == RigType.STEREO.value


def test_standard_aspect_is_head_mounted(tmp_path):
    _write_video(tmp_path / "clip.mp4", 1920, 1080)     # 16:9 -> egocentric default
    assert detect_layout(tmp_path) == "single_video"
    assert detect_rig(tmp_path) == RigType.HEAD_MOUNTED.value


def test_multiple_videos_is_multi_camera_teleop(tmp_path):
    _write_video(tmp_path / "top.mp4", 640, 480)
    _write_video(tmp_path / "wrist.mp4", 640, 480)
    assert detect_layout(tmp_path) == "multi_camera"
    assert detect_rig(tmp_path) == RigType.TELEOP_ROBOT.value


def test_lerobot_dataset_detected_by_info_json(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    assert detect_layout(tmp_path) == "lerobot"


def test_h5_without_video_is_rlds_but_video_wins(tmp_path):
    (tmp_path / "data.h5").write_bytes(b"\x00")
    assert detect_layout(tmp_path) == "rlds"
    # a v1 session carries BOTH an .mp4 and a stray .h5 -> the video wins
    _write_video(tmp_path / "compressed.mp4", 1920, 1080)
    assert detect_layout(tmp_path) == "single_video"


# ---------------------------------------------------------------- filename normalization
def test_normalize_fixes_spaces_and_parens(tmp_path):
    (tmp_path / "My Video (1).mp4").write_bytes(b"x")
    renames = normalize_filenames(tmp_path)
    assert renames == [("My Video (1).mp4", "My_Video_1_.mp4")]
    assert (tmp_path / "My_Video_1_.mp4").exists()


def test_normalize_is_idempotent(tmp_path):
    (tmp_path / "clean_name.mp4").write_bytes(b"x")
    assert normalize_filenames(tmp_path) == []          # already safe -> no renames


def test_normalize_drops_non_ascii(tmp_path):
    (tmp_path / "vidéo.mp4").write_bytes(b"x")
    normalize_filenames(tmp_path)
    assert (tmp_path / "vido.mp4").exists()              # non-ascii dropped, not crashed


# ---------------------------------------------------------------- consent default
def test_local_consent_is_granted():
    assert local_consent_default() == ConsentStatus.GRANTED


def test_local_consent_still_blocks_delivery_via_pii(tmp_path):
    """The safety invariant: local consent=GRANTED does NOT make an episode deliverable,
    because pii_status stays PENDING -> the fail-closed delivery gate still blocks."""
    from actuate.config import PiiStatus
    from actuate.io.consent import ConsentViolation, check_deliverable

    # consent granted, pii still pending -> must raise
    with pytest.raises(ConsentViolation):
        check_deliverable("ep", ConsentStatus.GRANTED, PiiStatus.PENDING)


def test_redaction_variant_is_not_multi_camera(tmp_path):
    """compressed.mp4 + redacted_compressed.mp4 are ONE camera (a redaction variant), not
    two -- the real processed-session layout must not read as multi_camera."""
    _write_video(tmp_path / "compressed.mp4", 1920, 1080)
    _write_video(tmp_path / "redacted_compressed.mp4", 1920, 1080)
    assert detect_layout(tmp_path) == "single_video"
    assert detect_rig(tmp_path) == RigType.HEAD_MOUNTED.value
