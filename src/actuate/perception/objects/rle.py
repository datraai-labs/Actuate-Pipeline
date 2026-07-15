"""Binary-mask RLE -- portable, no pycocotools.

The canonical schema's MaskRef carries `rle: str`. pycocotools has no Windows wheel and
would be a heavy dependency for one small codec, so this is a self-contained column-major
run-length encoding (the same axis order COCO uses, so the strings are conceptually
compatible even though the framing bytes differ).

Format: ``"<H>x<W>:<r0>,<r1>,..."`` -- height, width, then run lengths over the
column-major-flattened boolean mask, starting from a run of False. Decoding is exact; a
round-trip is bit-identical. Kept trivially decodable on purpose: every consumer that reads a
mask (IoU checks, visualisation, the certificate) must be able to without a native extension.
"""

from __future__ import annotations

import numpy as np


def encode_rle(mask: np.ndarray) -> str:
    """Boolean (H, W) -> RLE string. Column-major runs starting from False."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    flat = m.flatten(order="F")
    # Run lengths via change points. Prepend an implicit False run of length 0 so the
    # sequence always starts with False (matches COCO's convention).
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(bounds)
    if flat.size and flat[0]:  # mask starts True -> emit a leading zero-length False run
        runs = np.concatenate(([0], runs))
    return f"{h}x{w}:" + ",".join(str(int(r)) for r in runs)


def decode_rle(s: str) -> np.ndarray:
    """RLE string -> boolean (H, W). Exact inverse of encode_rle."""
    shape, _, body = s.partition(":")
    hs, _, ws = shape.partition("x")
    h, w = int(hs), int(ws)
    runs = [int(x) for x in body.split(",")] if body else []
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    val = False
    for r in runs:
        if val:
            flat[pos : pos + r] = True
        pos += r
        val = not val
    return flat.reshape((h, w), order="F")


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks. 0.0 if both empty is treated as 1.0
    only when they are identical-empty; two empties that 'agree' still return 1.0 so a lost
    track does not masquerade as a perfect match -- callers gate on presence separately."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if a.sum() == b.sum() else 0.0
    return float(inter / union)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two xyxy boxes. Used for track association across detection samples."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
