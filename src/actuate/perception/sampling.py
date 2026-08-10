"""Shared frame sampling for the perception stages (hands + depth).

The bug this fixes: reading the FIRST `max_frames` frames means a short cap only sees the
first ~half-second, so a video whose hands enter later yields nothing. Sampling EVENLY
across the whole clip makes a small `max_frames` representative -- and hands and depth MUST
use the same indices, or their per-frame records won't align.
"""

from __future__ import annotations

import numpy as np


def sampled_indices(frame_count: int, max_frames: int | None) -> list[int]:
    """Frame indices to process: all of them, or `max_frames` spread evenly across the clip.

    `max_frames=None` or >= frame_count -> every frame (contiguous 0..n-1). Otherwise
    `max_frames` indices linearly spaced over [0, frame_count-1], so a 15-frame sample of a
    2850-frame video spans the whole recording rather than its first half-second.
    """
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []
    if not max_frames or max_frames >= frame_count:
        return list(range(frame_count))
    return sorted({round(x) for x in
                      np.linspace(0, frame_count - 1, int(max_frames))})
