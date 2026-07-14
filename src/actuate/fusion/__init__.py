"""L2 — Sensor-Fused Interaction Refinement.

Spec: docs/architecture/MASTER_IMPLEMENTATION_SPEC.md §L2

NET-NEW. Trust-weighted arbiter:
  measured_robotspace > measured_human > gripper_aperture > vision_primary > vision_fallback
Outputs interaction_state and contact.<finger> — the hard prerequisite for the
dexterous branch. Gate: a broken-priority variant of the arbiter MUST fail its test.

NOT IMPLEMENTED. Increment 1 builds the foundation only (schema, io, catalog, infra).
This package is a placeholder so the import-linter contract has the full layer graph to
check against, and so nothing is quietly built out of order.
"""
