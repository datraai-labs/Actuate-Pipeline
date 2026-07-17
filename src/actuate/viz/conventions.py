"""Pure, viewer-free constants for visualization -- topology, colours, coordinate frame.

Kept separate from the Rerun logging so they can be unit-tested without importing (or running)
a viewer, and so the same skeleton/colour choices are used everywhere. Nothing here imports
rerun.
"""

from __future__ import annotations

from actuate.config import InteractionState, Side

#: MediaPipe/MANO 21-keypoint kinematic chain, as (parent, child) index pairs. Five fingers,
#: each wrist(0) -> mcp -> pip -> dip -> tip. This is what draws the hand as a skeleton.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
)

#: Fingertip keypoint indices, per finger name -- where contact confidence is shown.
FINGERTIPS: dict[str, int] = {
    "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20,
}

#: Left/right hand colours (RGB). Right = warm, left = cool -- a fixed convention so a viewer
#: never has to guess which hand is which.
HAND_COLORS: dict[Side, tuple[int, int, int]] = {
    Side.RIGHT: (255, 140, 0),   # orange
    Side.LEFT: (0, 160, 255),    # blue
}

#: Interaction-state colours. The spec asks for green=static, blue=grasped, red=moving.
STATE_COLORS: dict[InteractionState, tuple[int, int, int]] = {
    InteractionState.STATIC: (40, 200, 40),        # green
    InteractionState.GRASPED_L: (60, 120, 255),    # blue
    InteractionState.GRASPED_R: (60, 120, 255),    # blue
    InteractionState.GRASPED_BOTH: (30, 80, 255),  # deeper blue
    InteractionState.MOVING: (230, 60, 60),        # red
}


#: Color per action verb, grouped by kind so the timeline reads at a glance:
#: grey = idle, blue = transit/reach, green = grasp/hold, amber = manipulate.
_VERB_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (120, 120, 120),
    "reach": (80, 160, 255), "transport": (80, 160, 255), "lift": (60, 200, 200),
    "lower": (60, 200, 200), "place": (60, 200, 200),
    "grasp": (60, 220, 90), "hold": (40, 180, 70), "release": (200, 220, 60),
    "align": (240, 170, 40), "stabilize": (240, 170, 40), "insert": (240, 140, 40),
    "remove": (240, 140, 40), "open": (240, 140, 40), "close": (240, 140, 40),
    "push": (230, 110, 60), "pull": (230, 110, 60), "rotate": (230, 110, 60),
    "wipe": (230, 110, 60), "pour": (230, 110, 60),
}


def verb_color(verb: str) -> tuple[int, int, int]:
    """Timeline color for an action verb (grey if unknown, never crashes viz)."""
    return _VERB_COLORS.get(verb, (150, 150, 150))


def contact_color(confidence: float) -> tuple[int, int, int]:
    """Bright = confident contact, dim = uncertain. confidence in [0, 1] (vision caps it low).

    Maps to a green ramp: a confident contact is bright green, an uncertain one is dim. Kept
    linear and clamped so a NaN-free confidence always yields a valid colour.
    """
    c = 0.0 if confidence != confidence else max(0.0, min(1.0, float(confidence)))
    v = int(40 + 215 * c)
    return (30, v, 30)


def skeleton_strips(positions) -> list:
    """Turn 21 keypoint positions into the list of 2-point line segments for LineStrips3D.

    positions: (21, 3) array-like. Returns [[p_parent, p_child], ...] for each HAND_EDGE,
    skipping any edge that touches a non-finite point (a dropped keypoint) rather than drawing
    a line to the origin.
    """
    import numpy as np

    p = np.asarray(positions, dtype=np.float64)
    strips = []
    for a, b in HAND_EDGES:
        if np.isfinite(p[a]).all() and np.isfinite(p[b]).all():
            strips.append([p[a].tolist(), p[b].tolist()])
    return strips
