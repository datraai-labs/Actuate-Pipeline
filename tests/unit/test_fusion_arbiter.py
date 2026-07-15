"""L2 fusion: the trust-ordered arbiter, the Schmitt de-flicker, and per-finger contact.

The load-bearing test here is `test_hardware_overrides_vision_AND_broken_priority_fails`: it
proves the arbiter obeys the trust ordering by showing that INVERTING the ordering flips the
outcome. A correctness test that cannot be made to fail proves nothing; this one is wired so the
broken variant genuinely fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from actuate.config import Finger, InteractionState, Provenance, Side
from actuate.fusion import (
    Candidate,
    HardwareSources,
    SchmittTrigger,
    TRUST_RANK,
    arbitrate,
    enforce_min_dwell,
    finger_curl,
    grasp_signal,
)
from actuate.fusion import run as fusion_run


# --------------------------------------------------------------------------------------
# The arbiter
# --------------------------------------------------------------------------------------


def test_trust_rank_matches_the_spec_ordering():
    order = [
        Provenance.MEASURED_ROBOTSPACE,
        Provenance.MEASURED_HUMAN,
        Provenance.GRIPPER_APERTURE,
        Provenance.VISION_PRIMARY,
        Provenance.VISION_FALLBACK,
        Provenance.APPROXIMATED,
    ]
    ranks = [TRUST_RANK[p] for p in order]
    assert ranks == sorted(ranks, reverse=True), "trust must decrease down the spec ordering"
    assert TRUST_RANK[Provenance.MEASURED_ROBOTSPACE] == max(TRUST_RANK.values())


def test_higher_tier_beats_higher_confidence_in_a_lower_tier():
    """A vision reading at confidence 1.0 must NOT beat a glove reading at confidence 0.3 --
    the tier dominates confidence. This is the whole point of the trust ordering."""
    glove = Candidate("grasp", 0.9, Provenance.MEASURED_HUMAN, confidence=0.3)
    vision = Candidate("grasp", 0.1, Provenance.VISION_PRIMARY, confidence=1.0)
    assert arbitrate([vision, glove]).provenance is Provenance.MEASURED_HUMAN


def test_confidence_breaks_ties_within_a_tier():
    a = Candidate("c", 1, Provenance.VISION_PRIMARY, confidence=0.4)
    b = Candidate("c", 2, Provenance.VISION_PRIMARY, confidence=0.8)
    assert arbitrate([a, b]).value == 2


def test_hardware_overrides_vision_AND_broken_priority_fails():
    """Synthetic glove (measured_human) vs vision (vision_primary) for the same channel.

    Correct arbiter: glove wins. Broken arbiter (inverted rank): vision wins -> the SAME
    assertion fails. Demonstrated both ways so the test can't pass vacuously.
    """
    glove = Candidate("finger_joints", [0.5, 0.6], Provenance.MEASURED_HUMAN, confidence=0.3)
    vision = Candidate("finger_joints", [0.1, 0.1], Provenance.VISION_PRIMARY, confidence=1.0)

    # Correct ordering -> hardware wins.
    assert arbitrate([vision, glove]).provenance is Provenance.MEASURED_HUMAN

    # Broken ordering (inverted trust) -> vision wins. Prove the arbiter depends on the rank.
    broken = {p: -r for p, r in TRUST_RANK.items()}
    assert arbitrate([vision, glove], rank=broken).provenance is Provenance.VISION_PRIMARY
    # ...and that broken result would FAIL the correctness assertion:
    with pytest.raises(AssertionError):
        assert arbitrate([vision, glove], rank=broken).provenance is Provenance.MEASURED_HUMAN


# --------------------------------------------------------------------------------------
# Schmitt de-flicker
# --------------------------------------------------------------------------------------


def test_schmitt_holds_through_a_noisy_dip():
    trig = SchmittTrigger(high=0.5, low=0.3)
    seq = [0.0, 0.6, 0.55, 0.35, 0.6]  # 0.35 dips below high but not below low
    out = [trig.update(v) for v in seq]
    assert out == [False, True, True, True, True], "a dip above `low` must not drop the state"


def test_schmitt_requires_low_to_release():
    trig = SchmittTrigger(high=0.5, low=0.3)
    for v in [0.6, 0.4, 0.31]:
        trig.update(v)
    assert trig.state is True          # 0.31 still above low
    assert trig.update(0.29) is False  # now it releases


def test_schmitt_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        SchmittTrigger(high=0.3, low=0.5)


def test_min_dwell_absorbs_single_frame_islands():
    s = ["A", "A", "B", "A", "A"]          # the lone "B" is a 1-frame flicker
    assert enforce_min_dwell(s, 3) == ["A"] * 5
    # a run that meets the dwell survives
    s2 = ["A", "A", "B", "B", "B", "A"]
    assert enforce_min_dwell(s2, 3) == s2


# --------------------------------------------------------------------------------------
# Geometry: grasp signal
# --------------------------------------------------------------------------------------


def _flat_hand() -> np.ndarray:
    """A straight-fingered hand: fingers extended along +y from the palm row at y=0."""
    kp = np.zeros((21, 3), dtype=np.float64)
    bases = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for col, (_, mcp) in enumerate(bases.items()):
        x = col * 0.02
        for seg in range(4):  # mcp, pip, dip, tip straight along y
            kp[mcp + seg] = [x, 0.03 * seg, 0.0]
    return kp


def _fist() -> np.ndarray:
    """A curled hand: each finger folds back so the tip returns near the mcp."""
    kp = np.zeros((21, 3), dtype=np.float64)
    bases = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for col, (_, mcp) in enumerate(bases.items()):
        x = col * 0.02
        # pip forward, dip up, tip curls back toward mcp
        kp[mcp] = [x, 0.0, 0.0]
        kp[mcp + 1] = [x, 0.03, 0.0]
        kp[mcp + 2] = [x, 0.03, 0.03]
        kp[mcp + 3] = [x, 0.005, 0.005]
    return kp


def test_grasp_signal_separates_open_from_closed():
    assert grasp_signal(_flat_hand()) < 0.15
    assert grasp_signal(_fist()) > 0.4
    # per-finger curl is in [0,1)
    for v in finger_curl(_fist()).values():
        assert 0.0 <= v < 1.0


# --------------------------------------------------------------------------------------
# fusion.run end to end on synthetic hands
# --------------------------------------------------------------------------------------


class _HF:
    def __init__(self, side, kp3d, kp2d, bbox):
        self.side, self.keypoints_3d, self.keypoints_2d, self.bbox = side, kp3d, kp2d, bbox


class _Hands:
    def __init__(self, frames):
        self.frames = frames


def test_run_vision_only_is_vision_fallback_and_contact_is_populated():
    # 20 frames, right hand, alternating open/closed but hysteresis should not flicker.
    kp2 = np.tile([500.0, 400.0], (21, 1))
    frames = {}
    for i in range(20):
        kp3 = _fist() if i >= 5 else _flat_hand()
        frames[i] = [_HF(Side.RIGHT, kp3, kp2, (480, 380, 520, 420))]
    rep = fusion_run(_Hands(frames), rig=__import__("actuate.config", fromlist=["RigType"]).RigType.HEAD_MOUNTED)

    assert rep.provenance["grasp"] is Provenance.VISION_FALLBACK
    assert rep.provenance["contact"] is Provenance.VISION_FALLBACK
    # contact populated, non-NaN, and capped low (vision cannot feel contact)
    for ff in rep.frames.values():
        for side, fingers in ff.contact.items():
            for finger, cp in fingers.items():
                assert np.isfinite(cp.confidence)
                assert 0.0 <= cp.confidence <= 0.40
                assert cp.source is Provenance.VISION_FALLBACK
    # no single-frame flickers
    assert rep.flicker_count() == 0
    # and the GRASPED branch actually fires on the closed-fist frames (>=5): the state machine
    # is not structurally stuck, it just stays open on real footage because the real hand does.
    assert InteractionState.GRASPED_R in rep.states


def test_run_with_glove_flips_finger_joint_provenance_to_measured_human():
    from actuate.config import RigType

    kp2 = np.tile([500.0, 400.0], (21, 1))
    frames = {i: [_HF(Side.RIGHT, _fist(), kp2, (480, 380, 520, 420))] for i in range(6)}
    hw = HardwareSources(
        provenance=Provenance.MEASURED_HUMAN,
        finger_joints={i: {Side.RIGHT: (0.5, 0.6, 0.7)} for i in range(6)},
        contact={i: {Side.RIGHT: {f: 0.9 for f in Finger}} for i in range(6)},
    )
    rep = fusion_run(_Hands(frames), rig=RigType.GLOVE, hardware=hw)
    # every frame's right-hand finger joints now come from the glove, not vision
    for ff in rep.frames.values():
        assert ff.finger_joints_provenance[Side.RIGHT] is Provenance.MEASURED_HUMAN
        # and the glove's high-confidence contact overrides the capped vision value
        assert ff.contact[Side.RIGHT][Finger.INDEX].source is Provenance.MEASURED_HUMAN
        assert ff.contact[Side.RIGHT][Finger.INDEX].confidence == pytest.approx(0.9)
