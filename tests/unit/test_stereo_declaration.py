"""L0 declaration-vs-reality gate: a stereo eye must never ingest as monocular.

The bug this closes: a stereo capture's eyes are separate files (`take0003_L.mp4` /
`take0003_R.mp4`). Drop one into a session directory and it is 1920x1080 — pixel-identical
to a genuine monocular egocentric capture — so it satisfied the `head_mounted` declaration
and ingested cleanly. Its depth is then physically meaningless but looks plausible.

The synthetic tests below cover the provenance/naming logic and run anywhere. The
`real_data` tests exercise the actual Panoculon Trinet corpus and skip when it is absent;
point `ACTUATE_L0_FIXTURES` at a directory laid out as:

    stereo_pair/  take0003_L.mp4 + take0003_R.mp4   (real pair, declared stereo -> passes)
    lone_eye/     take0003_L.mp4                    (CONSTRUCTED: R deliberately withheld)
    mono/         video.mp4                         (real monocular, unmodified)

`lone_eye/` is a constructed fixture, not a found-in-the-wild artifact: it is the real left
eye of `stereo_pair/` with its companion withheld. The original mis-ingested session that
motivated this gate is not recoverable, so the failure mode is reproduced from real pixels
rather than asserted from a synthetic stand-in.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from actuate.config import RigType
from actuate.ingest import (
    RigDeclarationError,
    RigStreamError,
    classify_eye,
    find_companion_eye,
    find_lone_eye,
    validate_declared_rig,
    verify_stereo_pair,
)

cv2 = pytest.importorskip("cv2")


def _write_video(path: Path, w: int, h: int, n: int = 5) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


def _fixtures() -> Path | None:
    root = os.environ.get("ACTUATE_L0_FIXTURES")
    return Path(root) if root and Path(root).is_dir() else None


real_data = pytest.mark.real_data
needs_corpus = pytest.mark.skipif(
    _fixtures() is None,
    reason="set ACTUATE_L0_FIXTURES to the real Trinet corpus (see module docstring)",
)


# ------------------------------------------------------------------ provenance / naming
def test_classify_eye_recognizes_short_and_long_spellings():
    assert classify_eye(Path("take0003_L.mp4")).eye == "l"
    assert classify_eye(Path("take0003_R.mp4")).eye == "r"
    assert classify_eye(Path("cam_left.mp4")).eye == "left"
    assert classify_eye(Path("cam_right.mp4")).eye == "right"


def test_classify_eye_ignores_ordinary_names():
    """A monocular capture must not be read as an eye — this is the false-positive guard."""
    for name in ("video.mp4", "compressed.mp4", "redacted_compressed.mp4", "raw.mp4"):
        assert classify_eye(Path(name)) is None


def test_companion_found_when_pair_is_complete(tmp_path):
    _write_video(tmp_path / "take0003_L.mp4", 1920, 1080)
    _write_video(tmp_path / "take0003_R.mp4", 1920, 1080)
    eye = classify_eye(tmp_path / "take0003_L.mp4")
    assert find_companion_eye(eye, tmp_path).name == "take0003_R.mp4"
    assert find_lone_eye(tmp_path) is None


def test_lone_eye_detected_when_companion_missing(tmp_path):
    _write_video(tmp_path / "take0003_L.mp4", 1920, 1080)
    lone = find_lone_eye(tmp_path)
    assert lone is not None and lone.stem == "take0003" and lone.eye == "l"


def test_lone_eye_declared_head_mounted_is_refused(tmp_path):
    """The exact bug: one eye, 16:9, declared monocular. Geometry cannot catch it."""
    _write_video(tmp_path / "take0003_L.mp4", 1920, 1080)
    with pytest.raises(RigDeclarationError, match="one eye of a stereo pair"):
        validate_declared_rig(tmp_path, RigType.HEAD_MOUNTED)


def test_lone_eye_message_cites_orphaned_sidecars(tmp_path):
    _write_video(tmp_path / "take0003_L.mp4", 1920, 1080)
    (tmp_path / "take0003.imu").write_bytes(b"\x00")
    (tmp_path / "take0003_L.vts").write_bytes(b"\x00")
    with pytest.raises(RigDeclarationError, match="Orphaned stereo sidecars"):
        validate_declared_rig(tmp_path, RigType.HEAD_MOUNTED)


def test_genuine_monocular_still_passes(tmp_path):
    """Zero false positives: an ordinary 16:9 capture must sail through untouched."""
    _write_video(tmp_path / "video.mp4", 1920, 1080)
    assert validate_declared_rig(tmp_path, RigType.HEAD_MOUNTED) == "single_video"


def test_declared_camera_names_are_not_read_as_eyes(tmp_path):
    """`stereo_left`/`stereo_right` are the STEREO rig's DECLARED cameras, not a stray pair.

    Regression: the provenance gate's `_L`/`_left` match originally swallowed these, hiding
    the manifest's precise "declared camera 'stereo_right' has no video file" behind a
    vaguer lone-eye error. The rig registry is the authority on camera naming.
    """
    from actuate.ingest.stereo import declared_camera_stems

    assert "stereo_left" in declared_camera_stems()
    _write_video(tmp_path / "stereo_left.mp4", 1920, 1080)
    assert classify_eye(tmp_path / "stereo_left.mp4") is None
    assert find_lone_eye(tmp_path) is None


def test_companion_name_preserves_spelling_and_case(tmp_path):
    _write_video(tmp_path / "take0003_L.mp4", 1920, 1080)
    assert find_lone_eye(tmp_path).expected_companion_name == "take0003_R.mp4"
    (tmp_path / "take0003_L.mp4").unlink()
    _write_video(tmp_path / "cam_left.mp4", 1920, 1080)
    assert find_lone_eye(tmp_path).expected_companion_name == "cam_right.mp4"


# ------------------------------------------------------------------ ordering (Part A.2)
def test_declaration_gate_runs_before_rig_manifest_check(tmp_path):
    """Ordering is observable when BOTH gates would fail.

    A lone eye declared `umi_gripper` violates the declaration (one eye) AND the manifest
    (no aperture/IMU stream). The declaration error must win: the manifest describes what
    the DECLARED rig should produce, so checking it first validates a mis-declared session
    against the wrong contract. Before the ordering fix this raised RigStreamError.
    """
    from actuate.ingest import run as ingest_run

    _write_video(tmp_path / "take0003_L.mp4", 1920, 1080)
    (tmp_path / "session_meta.json").write_text('{"frame_count": 5, "fps_nominal": 30}')
    with pytest.raises(RigDeclarationError):
        ingest_run.run(RigType.UMI_GRIPPER, tmp_path)


def test_manifest_error_still_raised_when_declaration_is_clean(tmp_path):
    """The ordering fix must not swallow RigStreamError for a correctly-declared session."""
    from actuate.ingest import run as ingest_run

    _write_video(tmp_path / "video.mp4", 1920, 1080)
    (tmp_path / "session_meta.json").write_text('{"frame_count": 5, "fps_nominal": 30}')
    with pytest.raises(RigStreamError):
        ingest_run.run(RigType.UMI_GRIPPER, tmp_path)   # missing aperture + imu


# ------------------------------------------------------------------ real corpus
@real_data
@needs_corpus
def test_real_stereo_pair_corresponds():
    d = _fixtures() / "stereo_pair"
    ev = verify_stereo_pair(d / "take0003_L.mp4", d / "take0003_R.mp4")
    assert ev.corresponds, ev.reason
    assert ev.sign_stable and ev.rel_sd_dx < 0.40 and ev.max_abs_dy < 25.0
    assert abs(ev.mean_dx) > 8.0


@real_data
@needs_corpus
def test_real_stereo_pair_declared_stereo_passes():
    assert validate_declared_rig(_fixtures() / "stereo_pair", RigType.STEREO) == "multi_camera"


@real_data
@needs_corpus
def test_real_lone_eye_declared_head_mounted_is_refused():
    """The regression case, from real pixels. Replaces the unrecoverable f0ca2b74."""
    with pytest.raises(RigDeclarationError, match="one eye of a stereo pair"):
        validate_declared_rig(_fixtures() / "lone_eye", RigType.HEAD_MOUNTED)


@real_data
@needs_corpus
def test_real_monocular_declared_head_mounted_passes():
    """Zero false positives on the real monocular capture — as important as catching the bad one."""
    assert validate_declared_rig(_fixtures() / "mono", RigType.HEAD_MOUNTED) == "single_video"


@real_data
@needs_corpus
def test_same_view_twice_is_not_a_stereo_pair():
    """Negative control: a file against itself has zero baseline, so it is not a pair."""
    left = _fixtures() / "stereo_pair" / "take0003_L.mp4"
    ev = verify_stereo_pair(left, left)
    assert not ev.corresponds
