"""Regression tests for the two convention bugs that broke real-capture finger retargeting.

Both were silent: the code ran, produced in-limit joints, and was WRONG. Each test here is paired
with the broken variant it must reject -- a test that passes against the old behaviour proves
nothing (Master Spec §0).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from actuate.retarget.finger import human_fingertips
from actuate.retarget.finger.hand import ALLEGRO_MANO_TIPS

mano_calib = pytest.importorskip("actuate.retarget.finger.mano_calib")


def test_flexion_theta_bends_only_the_flexion_axis():
    """MANO's 45 = 15 joints x 3 axis-angle; a uniform fill is not a fist."""
    theta = mano_calib.flexion_theta(0.8)
    assert theta.shape == (45,)
    assert np.allclose(theta[mano_calib.FLEXION_AXIS::3], 0.8)
    # the other two axes per joint must stay at zero -- that is the whole point
    assert np.count_nonzero(theta) == 15


@pytest.mark.slow
def test_fist_is_more_curled_than_open_and_uniform_theta_is_not():
    """The canonical fist must actually close the hand.

    BROKEN VARIANT: filling all 45 components uniformly (the original bug). It leaves the fingers
    nearly extended, which is what inverted the calibration on real data.
    """
    pytest.importorskip("torch")
    cal = mano_calib.mano_canonical_fingertips()
    open_d = np.linalg.norm(cal["open"], axis=1).mean()
    fist_d = np.linalg.norm(cal["fist"], axis=1).mean()
    assert fist_d < open_d * 0.7, f"fist {fist_d:.3f} not meaningfully closed vs open {open_d:.3f}"

    # BROKEN VARIANT: the original uniform fill. Measured at 0.113 vs open 0.146 -- it must fail
    # the same bar, or this test would have passed against the bug.
    import torch

    mano = mano_calib._mano_layer()
    tip_v = [mano_calib.MANO_TIP_VERTICES[f] for f in ("index", "middle", "ring", "thumb")]
    r = mano(betas=torch.zeros(1, 10), hand_pose=torch.full((1, 45), 1.2),
             global_orient=torch.zeros(1, 3))
    j = r.joints.detach().numpy()[0]
    v = r.vertices.detach().numpy()[0]
    uniform_d = float(np.linalg.norm(np.array([v[i] for i in tip_v]) - j[0], axis=1).mean())
    assert not uniform_d < open_d * 0.7, "uniform-45 fill should NOT qualify as a fist"


@pytest.mark.slow
def test_flexion_magnitude_is_monotonic_within_the_canonical_range():
    """Past the calibrated magnitude the fingers over-rotate and tips travel back OUT.

    A larger "more closed" number silently meaning a less-closed hand is exactly the kind of
    non-monotonicity that makes a calibration constant unsafe to tune by eye.
    """
    torch = pytest.importorskip("torch")

    mano = mano_calib._mano_layer()
    tip_v = [mano_calib.MANO_TIP_VERTICES[f] for f in ("index", "middle", "ring", "thumb")]

    def mean_tip(mag: float) -> float:
        t = torch.tensor(mano_calib.flexion_theta(mag)).reshape(1, 45)
        r = mano(betas=torch.zeros(1, 10), hand_pose=t, global_orient=torch.zeros(1, 3))
        j = r.joints.detach().numpy()[0]
        v = r.vertices.detach().numpy()[0]
        return float(np.linalg.norm(np.array([v[i] for i in tip_v]) - j[0], axis=1).mean())

    mags = np.linspace(0.0, mano_calib.FIST_THETA, 5)
    d = [mean_tip(m) for m in mags]
    assert all(b < a for a, b in itertools.pairwise(d)), f"curl not monotonic over [0, FIST_THETA]: {d}"
    # and the documented reversal beyond it is real, not folklore
    assert mean_tip(1.6) > mean_tip(mano_calib.FIST_THETA)


def test_human_fingertips_derotates_with_global_orient():
    """Wrist-relative is not enough: WiLoR keypoints carry the hand's camera-frame rotation.

    BROKEN VARIANT: ignoring global_orient. A hand at a known pose, observed rotated, must yield
    the SAME canonical fingertips -- de-rotation makes the map orientation-invariant, and without
    it the same hand pose reads differently at every camera angle.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    kp = rng.normal(size=(21, 3)) * 0.05
    kp[0] = 0.0

    rot = Rotation.from_rotvec([0.4, -0.9, 0.3])
    kp_rot = rot.apply(kp)

    canonical = human_fingertips(kp, global_orient=np.zeros(3))
    derotated = human_fingertips(kp_rot, global_orient=rot.as_rotvec())
    np.testing.assert_allclose(canonical, derotated, atol=1e-9)

    # broken variant: no de-rotation -> the same pose reads differently
    naive = human_fingertips(kp_rot)
    assert not np.allclose(naive, canonical, atol=1e-3)


def test_human_fingertips_uses_mediapipe_keypoint_order():
    """WiLoR emits MediaPipe order (thumb 1-4, index 5-8, ...), verified against its own output."""
    assert ALLEGRO_MANO_TIPS == (8, 12, 16, 4)
    kp = np.zeros((21, 3))
    for i in range(21):
        kp[i] = [i, 0, 0]
    tips = human_fingertips(kp)
    np.testing.assert_allclose(tips[:, 0], [8, 12, 16, 4])
