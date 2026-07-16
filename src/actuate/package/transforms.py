"""Export-time co-training image transforms (EgoMimic). Applied at export, never stored.

Two variants EgoMimic uses to make human data trainable alongside robot data:

- **masked_hand** -- black out the hand region, forcing the policy to use proprioception
  (the state vector) instead of imitating human hand pixels a robot will never see.
- **eef_overlay** -- draw the end-effector's motion direction on the frame, giving the
  policy the intent signal without the human-specific appearance.

Both need the hand projected into PIXELS, and the canonical schema deliberately carries
keypoints in the camera frame (metres), not pixels -- so the caller must supply intrinsics.
No intrinsics, no transform: refusing beats projecting with a guess, because a mask drawn
with wrong intrinsics blacks out the wrong region and SILENTLY teaches the policy on exactly
the pixels it was meant to hide.
"""

from __future__ import annotations

import numpy as np

#: Padding around the projected hand bbox, as a fraction of its size.
_MASK_MARGIN = 0.25

TRANSFORM_NAMES = ("masked_hand", "eef_overlay")


def project_points(pts_3d: np.ndarray, intrinsics: tuple[float, float, float, float],
                   src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray | None:
    """Camera-frame (N,3) metres -> (N,2) pixels at the EXPORT resolution.

    `intrinsics` (fx, fy, cx, cy) are at the SOURCE resolution; the projection is scaled to
    the destination. Points at or behind the camera plane make the projection undefined ->
    None, and the caller skips the transform for that frame rather than drawing garbage.
    """
    pts = np.asarray(pts_3d, dtype=np.float64)
    z = pts[:, 2]
    if np.any(z <= 1e-6):
        return None
    fx, fy, cx, cy = intrinsics
    u = fx * pts[:, 0] / z + cx
    v = fy * pts[:, 1] / z + cy
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    return np.stack([u * sx, v * sy], axis=1)


def masked_hand(img: np.ndarray, hand_px: np.ndarray | None) -> np.ndarray:
    """Black out the hand's padded bbox. No projected hand -> the frame passes unchanged
    (a hand out of view is already 'masked' by reality)."""
    if hand_px is None:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    x0, y0 = hand_px.min(axis=0)
    x1, y1 = hand_px.max(axis=0)
    mx, my = _MASK_MARGIN * (x1 - x0), _MASK_MARGIN * (y1 - y0)
    xa, xb = int(np.clip(x0 - mx, 0, w)), int(np.clip(x1 + mx, 0, w))
    ya, yb = int(np.clip(y0 - my, 0, h)), int(np.clip(y1 + my, 0, h))
    out[ya:yb, xa:xb] = 0
    return out


def eef_overlay(img: np.ndarray, wrist_px: np.ndarray | None,
                next_wrist_px: np.ndarray | None) -> np.ndarray:
    """Arrow from the wrist toward its next-frame position (the EEF motion direction)."""
    import cv2

    if wrist_px is None or next_wrist_px is None:
        return img
    out = img.copy()
    p0 = tuple(int(v) for v in np.asarray(wrist_px).reshape(2))
    p1 = tuple(int(v) for v in np.asarray(next_wrist_px).reshape(2))
    if p0 != p1:
        cv2.arrowedLine(out, p0, p1, color=(0, 255, 0), thickness=2, tipLength=0.3)
    return out
