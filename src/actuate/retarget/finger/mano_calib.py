"""Per-human calibration synthesised from the MANO model (Master Spec §L5).

GeoRT needs per-human C-space calibration -- canonical poses ("open your hand", "make a fist")
to pin the human->robot curl correspondence. **Our capture does not contain one**: it is 95 s of
flat-hand paperwork, and the fusion analysis measured finger curl never exceeding 0.15 (the hand
never closes). There is no fist in the footage to calibrate against.

Rather than fake it, this synthesises the calibration from the MANO model itself, using the
DEMONSTRATOR'S OWN shape parameters (`betas`, which WiLoR estimates per frame). Posing that
person's hand at canonical thetas gives their open/fist fingertips **in MANO's own convention** --
which is exactly the convention WiLoR's keypoints arrive in, so the fitted transform is valid for
them.

Why this matters concretely: calibrating on a *synthetic* hand and applying it to real WiLoR
keypoints measurably inverted the map -- a real flat hand retargeted to openness 0.086 m, BELOW
the fist reference (0.094 m), i.e. more curled than a fist. Convention mismatch, not tuning.

Only the wrist joint and four fingertip VERTICES are needed (Allegro has no pinky), so no
16->21 joint reindex is required.

LICENCE: MANO is MPI non-commercial. This path inherits the launch blocker already tracked in
STATUS.md -- internal research only.
"""

from __future__ import annotations

import numpy as np

#: Standard MANO fingertip vertex indices (the mesh has no tip JOINTS).
MANO_TIP_VERTICES = {"index": 317, "middle": 444, "ring": 556, "thumb": 745, "pinky": 673}

#: MANO's 45 = 15 joints x 3 axis-angle. Only ONE axis per joint is flexion (measured: axis 2).
#: Setting all 45 components uniformly does NOT make a fist -- it twists and splays the hand. At
#: 1.2 uniform the tips barely move (index 0.146 -> 0.113 m) while only the thumb folds; that pose
#: is what inverted the calibration. Bend the flexion axis alone.
FLEXION_AXIS = 2

#: Canonical flexion magnitudes: 0 == MANO's open pose.
#: FIST is 0.8, the END of the monotonic curl range -- measured, mean tip 0.136 -> 0.073 m. Past
#: ~0.8 the fingers over-rotate and the tips travel back AWAY from the palm (1.6 -> 0.092 m), so a
#: larger "more closed" magnitude silently means a less-closed hand.
OPEN_THETA = 0.0
FIST_THETA = 0.8


def flexion_theta(magnitude: float) -> np.ndarray:
    """(45,) MANO hand_pose bending only the flexion axis of each of the 15 joints."""
    theta = np.zeros(45, dtype=np.float32)
    theta[FLEXION_AXIS::3] = magnitude
    return theta


def _mano_layer():
    from pathlib import Path

    import wilor_mini

    from actuate.perception.hands.mano_compat import patch_smplx

    patch_smplx()
    import smplx

    model_path = Path(wilor_mini.__file__).parent / "pretrained_models"
    return smplx.MANO(model_path=str(model_path), use_pca=False, is_rhand=True,
                      flat_hand_mean=False)


def mano_canonical_fingertips(
    betas: np.ndarray | None = None,
    fingers: tuple[str, ...] = ("index", "middle", "ring", "thumb"),
    open_theta: float = OPEN_THETA,
    fist_theta: float = FIST_THETA,
) -> dict[str, np.ndarray]:
    """{"open": (n_fingers,3), "fist": (n_fingers,3)} wrist-relative, in MANO convention.

    `betas`: the demonstrator's MANO shape from WiLoR (mean over frames is fine). None -> the
    mean hand shape.
    """
    import torch

    mano = _mano_layer()
    b = torch.zeros(1, 10) if betas is None else torch.tensor(
        np.asarray(betas, dtype=np.float32).reshape(1, 10))

    out = {}
    for name, val in (("open", open_theta), ("fist", fist_theta)):
        res = mano(betas=b, hand_pose=torch.tensor(flexion_theta(val)).reshape(1, 45),
                   global_orient=torch.zeros(1, 3))
        joints = res.joints.detach().numpy()[0]
        verts = res.vertices.detach().numpy()[0]
        wrist = joints[0]
        out[name] = np.array([verts[MANO_TIP_VERTICES[f]] for f in fingers]) - wrist
    return out
