"""Content-addressed provenance — and proof it removes the failure classes.

Increment 1's three integrity failures were caught by ad-hoc checks bolted onto the
migration. Checks catch what you thought to check for. Content-addressing removes the
failure class instead, and these tests demonstrate the difference: each one shows the bug
being *structurally impossible to express*, not merely detected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from actuate.config import RigType
from actuate.ingest import (
    CaptureManifest,
    IntegrityError,
    build_manifest,
    check_legacy_claims,
    hash_file,
)
from actuate.schema import CanonicalEpisode

REPO = Path(__file__).resolve().parents[2]
REAL_VIDEO = REPO / "raw" / "session_001" / "raw.mp4"


@pytest.fixture
def video(tmp_path: Path) -> Path:
    p = tmp_path / "raw.mp4"
    p.write_bytes(b"\x00\x01fake video payload" * 500)
    return p


# --- identity is derived from the bytes, not assigned ------------------------------------


def test_capture_id_IS_the_content_hash(video: Path):
    m = build_manifest(video, RigType.HEAD_MOUNTED)
    assert m.capture_id == m.content_hash == hash_file(video)


def test_identical_bytes_produce_one_capture_regardless_of_filename(tmp_path: Path, video: Path):
    """The duplicate-upload bug, made impossible.

    The real corpus had one recording under four session ids. Content-addressed, they
    compute the same id and collapse to one capture — so there is no second record to
    disagree with the first about consent.
    """
    copy = tmp_path / "totally_different_name.mp4"
    copy.write_bytes(video.read_bytes())

    a = build_manifest(video, RigType.HEAD_MOUNTED)
    b = build_manifest(copy, RigType.HEAD_MOUNTED)

    assert a.capture_id == b.capture_id
    assert a.prefix == b.prefix, "identical bytes must land at the identical S3 key"


def test_one_flipped_byte_is_a_different_capture(tmp_path: Path, video: Path):
    data = bytearray(video.read_bytes())
    data[0] ^= 0xFF
    other = tmp_path / "other.mp4"
    other.write_bytes(bytes(data))

    assert build_manifest(video, RigType.HEAD_MOUNTED).capture_id != build_manifest(
        other, RigType.HEAD_MOUNTED
    ).capture_id


# --- metadata cannot drift from its payload ------------------------------------------------


def test_verify_catches_a_swapped_payload(tmp_path: Path, video: Path):
    """THE Increment-1 bug: a 1.49 MB video filed under metadata describing 181 MB.

    The frame counts were internally consistent, so a check on those alone waved it
    through. Here the manifest carries the hash of the bytes it describes, so swapping the
    payload is detected by construction — there is nothing to remember to check.
    """
    m = build_manifest(video, RigType.HEAD_MOUNTED)
    m.verify(video)  # fine

    video.write_bytes(b"a completely different recording")

    with pytest.raises(IntegrityError, match="does not describe this payload"):
        m.verify(video)


def test_verify_catches_a_size_mismatch_before_hashing(tmp_path: Path, video: Path):
    m = build_manifest(video, RigType.HEAD_MOUNTED)
    tampered = m.model_copy(update={"size_bytes": m.size_bytes + 1})
    with pytest.raises(IntegrityError, match="bytes, file is"):
        tampered.verify(video)


def test_legacy_claims_check_catches_both_real_corpus_bugs(video: Path):
    """Both failures from the real corpus, detected against the actual bytes."""
    m = build_manifest(video, RigType.HEAD_MOUNTED, frame_count=2850)

    # Bug 2: the metadata describes a 181.84 MB source; the bytes are ~10 KB.
    problems = check_legacy_claims(m, {"raw_size_mb": 181.84}, perception_frames=2850)
    assert any("different recordings" in p for p in problems)

    # Bug 1: metadata declares 1544 frames; 2850 per-frame records exist.
    problems = check_legacy_claims(
        m, {"frame_count": 1544, "raw_size_mb": m.size_bytes / 1e6}, perception_frames=2850
    )
    assert any("different recording" in p for p in problems)

    # Consistent metadata passes.
    assert (
        check_legacy_claims(
            m, {"frame_count": 2850, "raw_size_mb": m.size_bytes / 1e6}, perception_frames=2850
        )
        == []
    )


# --- derived artifacts are bound back to the capture ----------------------------------------


def test_an_episode_cannot_claim_a_hash_that_disagrees_with_its_capture(video: Path):
    """schema v2: capture_id IS the content hash. Letting them diverge would reintroduce
    exactly the ambiguity content-addressing exists to remove."""
    h = hash_file(video)
    other = "b" * 64

    CanonicalEpisode(
        episode_id="e", capture_id=h, source_content_hash=h, rig=RigType.HEAD_MOUNTED
    )

    with pytest.raises(ValidationError, match="disagrees with capture_id"):
        CanonicalEpisode(
            episode_id="e", capture_id=h, source_content_hash=other, rig=RigType.HEAD_MOUNTED
        )


def test_manifest_round_trips(video: Path):
    m = build_manifest(video, RigType.GLOVE, frame_count=100, duration_sec=3.3, fps=30.0)
    assert CaptureManifest.model_validate_json(m.model_dump_json()) == m


# --- against the real capture ----------------------------------------------------------------


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="real session_001 raw.mp4 not on disk")
def test_the_real_capture_hashes_and_its_legacy_metadata_is_consistent():
    """Hash the actual 190 MB recording and cross-check the v1 metadata against it."""
    meta = json.loads(
        (REPO / "processed" / "session_001" / "session_meta.json").read_text()
    )
    frames = len(
        json.loads((REPO / "processed" / "session_001" / "hand_pose_3d.json").read_text())
    )

    m = build_manifest(
        REAL_VIDEO,
        RigType.HEAD_MOUNTED,
        frame_count=meta["frame_count"],
        duration_sec=meta["duration_seconds"],
    )

    assert len(m.capture_id) == 64
    assert m.size_bytes > 100_000_000, "expected the ~190 MB recording"
    m.verify(REAL_VIDEO)

    # session_001 is the ONE session whose bytes and metadata actually agree.
    assert check_legacy_claims(m, meta, frames) == []
