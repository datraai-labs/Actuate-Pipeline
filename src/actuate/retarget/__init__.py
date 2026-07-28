"""L5 — Cross-Embodiment Retargeting (Master Spec §L5).

The largest net-new workstream. Arm (Vector-Neuron + flow-matching root-frame estimator,
MuJoCo-trained IK) is in `actuate.retarget.arm`. Finger (GeoRT), contact-consistency
reconciliation, and sim no-slip validation are the remaining branches (Parts E, F).

`actuate.retarget.arm` retargets a canonical wrist trajectory to a robot joint trajectory:
sim-validated (gates 1/2/4). Real-capture validation (gate 3) is deferred until depth is
trustworthy (Phase 3.5 gate). DexUMI exoskeleton rigs BYPASS the finger branch — already
robot-space.

`actuate.retarget.humanoid` is the separate full-body path from the GMR paper
(arXiv:2510.02252). It consumes BVH motion and targets humanoids; it is not a Franka
replacement and cannot consume an egocentric wrist trajectory by itself.
"""

from __future__ import annotations

from actuate.retarget import arm, humanoid

__all__ = ["arm", "humanoid"]
