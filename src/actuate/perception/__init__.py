"""L1 — Perception & Metric Reconstruction.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L1

WiLoR->MANO, UniDepthV2/FoundationStereo, Grounding DINO + SAM2, FoundationPose, SLAM.
All model swaps are BENCHMARK-BEFORE-LOCK (Master Spec §7) and need real GPU
validation before they count as done. GPU; behind the `perception` extra.

NOT IMPLEMENTED. Increment 1 builds the foundation only (schema, io, catalog, infra).
This package is a placeholder so the import-linter contract has the full layer graph to
check against, and so nothing is quietly built out of order.
"""
