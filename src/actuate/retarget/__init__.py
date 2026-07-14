"""L5 — Cross-Embodiment Retargeting.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L5

NET-NEW and the largest workstream. Arm (Vector-Neuron + flow-matching, MuJoCo-trained),
finger (GeoRT), contact-consistency reconciliation, sim no-slip validation.
DexUMI exoskeleton rigs BYPASS the finger branch — already robot-space.
Gate: reconciliation MUST fail on a deliberately contact-inconsistent trajectory.

NOT IMPLEMENTED. Increment 1 builds the foundation only (schema, io, catalog, infra).
This package is a placeholder so the import-linter contract has the full layer graph to
check against, and so nothing is quietly built out of order.
"""
