# Research basis for Actuate's trust contract

This is an implementation map, not a claim that citing a paper validates our model outputs.
Each product claim still needs a named artifact, calibration set, and real-data gate.

## Confidence, abstention, and unfamiliar data

- Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599):
  raw neural-network scores are not automatically calibrated probabilities. Actuate therefore
  preserves raw detector/depth signals for diagnostics but only certificate fields explicitly
  marked `_calibrated` may contribute to customer-facing perception confidence.
- Geifman and El-Yaniv,
  [SelectiveNet](https://arxiv.org/abs/1901.09192): a system should be able to abstain instead
  of manufacturing a prediction. Missing confidence, grasp, task, or eligibility is represented
  as unknown and gates export where the field is required.
- Ovadia et al.,
  [Can You Trust Your Model's Uncertainty?](https://arxiv.org/abs/1906.02530): confidence can
  deteriorate under dataset shift. Rig mismatch and unfamiliar-source checks happen before a
  quality number is trusted.

## Sequential robot datasets

- Mandlekar et al., [RLDS](https://arxiv.org/abs/2111.02767): episodes and steps preserve
  sequential semantics and metadata. Actuate refuses exports without trainable successors and
  records dropped frames and lineage rather than silently changing the episode.
- Open X-Embodiment Collaboration,
  [Open X-Embodiment](https://arxiv.org/abs/2310.08864): cross-embodiment scale makes consistent
  action/state contracts and embodiment metadata essential. Actuate exports robot action only
  after the exact embodiment's eligibility verdict passes.
- Khazatsky et al., [DROID](https://arxiv.org/abs/2403.12945): broad environments, tasks, and
  viewpoints matter for generalization. Actuate's source adapter preserves provenance and its
  certificate does not upgrade unfamiliar imports to high confidence by default.

## Data documentation and privacy

- Gebru et al., [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): shipped datasets
  need motivation, composition, collection, processing, and use constraints. Every run/export
  includes a README, schema, source identity, omissions, and code/dependency lineage.
- Nagar et al., [EgoBlur](https://arxiv.org/abs/2308.13093): egocentric video needs dedicated
  face/license-plate anonymization. Redaction is visible, explicit, and fail-closed for delivery;
  it is never implied merely because processing completed.

## Geometry and hand perception

- Piccinelli et al., [UniDepth](https://arxiv.org/abs/2403.18913) and
  [UniDepthV2](https://arxiv.org/abs/2502.20110): metric monocular depth is model-derived and its
  native confidence remains a model signal until calibrated on the active capture domain.
- Potamias et al., [WiLoR](https://arxiv.org/abs/2409.12259): the live hand path uses WiLoR,
  while absent detector confidence remains unknown rather than defaulting to 1.0.
- Grauman et al., [Ego-Exo4D](https://arxiv.org/abs/2311.18259): egocentric and exocentric views
  have materially different geometry and evidence. Actuate validates declared rig/layout and
  does not process a stereo composite as a single monocular image.

## Rules enforced in code

1. Never persist or propagate a claim without validating it at the point of use.
2. Never use a plausible numeric default for a signal that was not measured.
3. Never turn a raw model score into a probability without an identified calibration method.
4. Preserve source identity, frame coverage, code/model lineage, and every skip reason.
5. Consent, PII, task, and physics eligibility are export-time checks, not decorative metadata.
