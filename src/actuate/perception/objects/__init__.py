"""L1 objects -- open-vocab detection, video segmentation, tracking, and (stubbed) 6-DoF.

Grounding DINO (text-prompted boxes) + SAM2 (temporal mask propagation) + optional metric
position via Part C depth. FoundationPose 6-DoF is an interface stub (needs a mesh + a bigger
GPU). Master Spec §L1.
"""

from __future__ import annotations

from actuate.perception.objects.foundationpose import (
    FoundationPoseEstimator,
    FoundationPoseUnavailable,
    synthetic_interface_check,
)
from actuate.perception.objects.objects import (
    GroundingDinoDetector,
    ObjectFrame,
    ObjectResult,
    Sam2Tracker,
    run,
)
from actuate.perception.objects.rle import (
    bbox_iou,
    decode_rle,
    encode_rle,
    mask_iou,
)

__all__ = [
    "FoundationPoseEstimator",
    "FoundationPoseUnavailable",
    "GroundingDinoDetector",
    "ObjectFrame",
    "ObjectResult",
    "Sam2Tracker",
    "bbox_iou",
    "decode_rle",
    "encode_rle",
    "mask_iou",
    "run",
    "synthetic_interface_check",
]
