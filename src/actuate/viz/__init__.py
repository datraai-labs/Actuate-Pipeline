"""Rerun visualization -- the "see your pipeline" capability. Master Spec §F.

`viz.log_episode(episode, hands=, depth=, objects=, fusion=, slam=, video_frames=)` logs every
present modality to the active Rerun recording on one scrubable `frame` timeline: video, depth
(image + 3D cloud), camera trajectory, 3D hand skeletons placed at metric depth, object boxes +
masks, interaction state, per-finger contact, and wrist-action arrows. The CLI wraps it as
`actuate viz <episode>`.

Rerun is imported lazily inside the logging functions so importing this package (and the
library as a whole) never requires the viewer to be installed.
"""

from __future__ import annotations

from actuate.viz.conventions import (
    HAND_COLORS,
    HAND_EDGES,
    STATE_COLORS,
    contact_color,
    skeleton_strips,
)
from actuate.viz.rerun_log import (
    log_episode,
    log_hand,
    log_objects,
    log_state,
)

__all__ = [
    "log_episode",
    "log_hand",
    "log_objects",
    "log_state",
    "HAND_EDGES",
    "HAND_COLORS",
    "STATE_COLORS",
    "contact_color",
    "skeleton_strips",
]
