"""
DatraAI Pipeline — Glove-Aware Grasp Threshold Profiles (v2 addendum §2)

Gloves add material thickness around the fingers/palm, so the SAME
physical grasp closes to a larger apparent normalized landmark distance
than a bare hand produces. Using bare-hand POWER_GRASP_DIST/
LATERAL_PINCH_DIST thresholds directly on gloved-hand footage would
under-detect grasps — gloved fingers never register as "closed enough".
This module resolves the effective thresholds for a session's declared
glove_type; scripts/05_primitives.py calls this once per session and
threads the result through utils/imu_source_router.py's detectors and
utils/confidence.py's matching confidence functions via the per-frame
imu_window dict (see 05_primitives.py's run()).
"""

from typing import Optional

import config as cfg


def resolve_grasp_thresholds(glove_type: Optional[str]) -> dict:
    """
    Resolve the effective power_grasp/lateral_pinch distance thresholds
    for a declared glove_type. An unrecognized or missing glove_type falls
    back to config.GLOVE_TYPE_DEFAULT's multiplier (1.0, i.e. bare-hand
    thresholds) rather than raising — a session with no glove_type field
    at all (older ingest, or a session_config.json that omits it) should
    behave exactly as it did before this section existed.
    """
    multiplier = cfg.GLOVE_THRESHOLD_MULTIPLIERS.get(
        glove_type, cfg.GLOVE_THRESHOLD_MULTIPLIERS[cfg.GLOVE_TYPE_DEFAULT]
    )
    return {
        "glove_type": glove_type if glove_type in cfg.GLOVE_THRESHOLD_MULTIPLIERS else cfg.GLOVE_TYPE_DEFAULT,
        "multiplier": multiplier,
        "power_grasp_dist": round(cfg.POWER_GRASP_DIST * multiplier, 6),
        "lateral_pinch_dist": round(cfg.LATERAL_PINCH_DIST * multiplier, 6),
    }
