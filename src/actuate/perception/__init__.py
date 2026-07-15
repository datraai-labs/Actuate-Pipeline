"""L1 -- Perception & Metric Reconstruction.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L1

Built:
    slam/    ego-motion (gyro + vision). The Phase-3 blocker.

Not built (Phase 3 Parts B-D):
    hands/   WiLoR -> MANO
    depth/   UniDepthV2 (mono) / FoundationStereo (stereo)
    objects/ Grounding DINO + SAM2 + FoundationPose
"""
