"""Canonical store round-trip — Parquet + Zarr, BIT-EXACT (Master Spec §3 gate).

The last test in this file is the one that counts: it round-trips frames built from the
**real** session_001 capture on disk, not a synthetic fixture. Per Master Spec §0, a
component is done only when tested against real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from actuate.config import (
    Bucket,
    ConsentStatus,
    Finger,
    InteractionState,
    PiiStatus,
    Provenance,
    RigType,
    Side,
)
from actuate.io import LocalBackend, read_episode, write_episode
from actuate.schema import (
    SE3,
    CanonicalEpisode,
    CanonicalFrame,
    ContactReading,
    DepthRef,
    EpisodeMeta,
    HandState,
    ImageRef,
    MANOParams,
)

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "processed" / "session_001"

KP = tuple((i * 1e-3 + 1e-9, -i * 2e-3, 1.0 + i * 1e-7) for i in range(21))
MANO = MANOParams(
    betas=tuple(0.1 * i for i in range(10)),
    theta=tuple(0.01 * i for i in range(45)),  # schema v3: full 45 axis-angle
    global_orient=(0.1, 0.2, 0.3),
)


def _glove_frame(i: int) -> CanonicalFrame:
    return CanonicalFrame(
        t=i / 30.0,
        rig=RigType.GLOVE,
        episode_id="ep",
        frame_idx=i,
        images={"head": ImageRef(uri="s3://actuate-work-dev/v.mp4", frame_index=i)},
        depth={"head": DepthRef(uri="s3://x/d", frame_index=i, mean_uncertainty=0.123456789)},
        camera_pose=SE3(position_m=(0.1, 0.2, 0.3), quaternion_wxyz=(1, 0, 0, 0)),
        hands={
            Side.RIGHT: HandState(
                mano=MANO,
                keypoints_3d=KP,
                wrist_pose=SE3(position_m=(1e-8, 2.0, 3.0), quaternion_wxyz=(1, 0, 0, 0)),
            )
        },
        finger_joints_human={Side.RIGHT: (0.1, 0.2, 0.3, 0.4, 0.5)},
        contact={
            Side.RIGHT: {
                f: ContactReading(confidence=0.1 * k, source=Provenance.MEASURED_HUMAN)
                for k, f in enumerate(Finger)
            }
        },
        interaction_state=InteractionState.GRASPED_R,
        confidence={"hands": 0.987654321},
        provenance={
            "hands": Provenance.VISION_PRIMARY,
            "camera_pose": Provenance.VISION_PRIMARY,
            "depth": Provenance.VISION_PRIMARY,
            "contact": Provenance.MEASURED_HUMAN,
            "finger_joints_human": Provenance.MEASURED_HUMAN,
            "interaction_state": Provenance.MEASURED_HUMAN,
        },
    )


@pytest.fixture
def backend(tmp_path: Path) -> LocalBackend:
    return LocalBackend(tmp_path)


def test_round_trip_is_bit_exact(backend: LocalBackend):
    ep = CanonicalEpisode(
        episode_id="ep",
        capture_id="cap",
        rig=RigType.GLOVE,
        frames=tuple(_glove_frame(i) for i in range(5)),
        task="Pick up the red cube",
        consent=ConsentStatus.GRANTED,
        pii_status=PiiStatus.PASSED,
        episode_meta=EpisodeMeta(quality=4, speed=150, mistakes=("slipped once",)),
    )
    write_episode(backend, ep)
    assert read_episode(backend, "ep") == ep


def test_a_frame_with_no_hand_comes_back_with_no_hand(backend: LocalBackend):
    """NOT a zero-filled hand.

    Absence is encoded as NaN on disk, and the reader must restore `None`. If it restored
    a zeroed HandState instead, every frame where the hand left the view would look to a
    training loader like a hand at the origin.
    """
    ep = CanonicalEpisode(
        episode_id="ep",
        capture_id="cap",
        rig=RigType.GLOVE,
        frames=(
            _glove_frame(0),
            CanonicalFrame(
                t=1 / 30, rig=RigType.GLOVE, episode_id="ep", frame_idx=1,
                interaction_state=InteractionState.STATIC,
                provenance={"interaction_state": Provenance.VISION_FALLBACK},
            ),
        ),
    )
    write_episode(backend, ep)
    back = read_episode(backend, "ep")

    assert back.frames[1].hands == {}
    assert back.frames[1].contact is None
    assert back == ep


def test_canonical_artifacts_land_in_work_not_delivery(backend: LocalBackend):
    """Getting to the delivery bucket requires passing the consent gate. The canonical
    store has no business writing there, and does not."""
    ep = CanonicalEpisode(
        episode_id="ep", capture_id="cap", rig=RigType.GLOVE, frames=(_glove_frame(0),)
    )
    write_episode(backend, ep)

    assert list(backend.list(Bucket.WORK, "canonical/ep")), "nothing written to work/"
    assert not list(backend.list(Bucket.DELIVERY)), "canonical store wrote to delivery!"


# --- the one that counts ---------------------------------------------------------------


@pytest.mark.real_data
@pytest.mark.skipif(not REAL.exists(), reason="real session_001 not on disk")
def test_round_trip_of_real_captured_frames_is_bit_exact(backend: LocalBackend):
    """Build canonical frames from the REAL session_001 capture and round-trip them.

    This is not the L3 `canonical build` (that is Increment 2) — it lifts the real
    per-frame 3D hand keypoints straight out of the v1 outputs into the frozen schema and
    proves the schema and the store survive contact with actual captured floats, including
    the frames where MediaPipe found no hand.
    """
    landmarks = json.loads((REAL / "hand_pose_3d.json").read_text())
    assert len(landmarks) > 1000, "expected a real multi-thousand-frame session"

    frames = []
    for rec in landmarks[:500]:
        i = rec["frame_idx"]
        hands = {}
        if rec.get("hands_detected") and rec.get("landmarks_3d_m"):
            side = Side.LEFT if rec.get("dominant_hand") == "left" else Side.RIGHT
            hands[side] = HandState(
                keypoints_3d=tuple(tuple(p) for p in rec["landmarks_3d_m"])
            )
        frames.append(
            CanonicalFrame(
                t=i / 30.0,
                rig=RigType.HEAD_MOUNTED,
                episode_id="session_001_ep00",
                frame_idx=i,
                hands=hands,
                # A bare-hand head-mounted rig measures NOTHING. Everything here is a
                # monocular-depth inference, and the schema will not let us say otherwise.
                provenance=(
                    {"hands": Provenance.VISION_PRIMARY} if hands else {}
                ),
            )
        )

    ep = CanonicalEpisode(
        episode_id="session_001_ep00",
        capture_id="real-capture",
        rig=RigType.HEAD_MOUNTED,
        frames=tuple(frames),
        consent=ConsentStatus.PENDING,  # the real record says pending
        pii_status=PiiStatus.PENDING,
    )

    write_episode(backend, ep)
    back = read_episode(backend, "session_001_ep00")

    assert back == ep, "real captured keypoints did not survive the round-trip bit-exactly"
    assert any(f.hands for f in back.frames), "expected some frames with a detected hand"
    assert any(not f.hands for f in back.frames), (
        "expected some frames with NO hand — session_001 has ~5% non-detection, and those "
        "must come back empty rather than zeroed"
    )
    assert not back.is_deliverable, "session_001's consent is pending; it must not ship"
