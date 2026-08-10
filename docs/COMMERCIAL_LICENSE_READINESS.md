# Commercial license readiness

Audit date: 2026-08-10
Proposed use: license Actuate to a commercial robotics company to process egocentric video.

This is an engineering due-diligence record, not legal advice. Counsel must verify the
underlying agreements, ownership chain, model-weight terms, patents, privacy obligations,
and the intended deployment before a commercial offer is signed.

## Decision

**Not commercially clearable as currently assembled.** The Actuate-authored orchestration,
schema, certification, dashboard, and adapters can be the basis of a proprietary license if
their ownership chain is documented. The default perception path cannot currently be offered
for commercial use because it actively uses multiple non-commercial model stacks.

A research evaluation of the architecture is technically possible, but a commercial company's
internal product R&D should not be assumed to satisfy a non-commercial-purpose restriction.

## Rights inventory

| Component actually used | Published terms found | Commercial status |
|---|---|---|
| Actuate code in this repository | No top-level `LICENSE`, copyright notice, contributor agreement, or third-party notice file | Ownership and authority to license must be documented before diligence |
| WiLoR models | [CC-BY-NC-ND-4.0](https://github.com/rolpotamias/WiLoR/blob/main/license.txt); its README also identifies MANO and Ultralytics dependencies | **Blocked** without separate commercial rights from the relevant rights holder(s) |
| MANO model/assets | [Max Planck non-commercial research grant](https://mano.is.tue.mpg.de/license.html); the page provides a commercial-licensing contact | **Blocked** without a signed commercial grant |
| UniDepth software and published weights | [CC-BY-NC-4.0](https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE) | **Blocked** without separate commercial rights or replacement |
| Ultralytics used by WiLoR's detector | AGPL-3.0 package; [Ultralytics says proprietary/commercial use requires Enterprise terms unless the complete project is released compatibly](https://www.ultralytics.com/license) | **Blocked for a closed proprietary offer** without Enterprise terms or replacement |
| HaMeR fallback code | [MIT](https://github.com/geopavlakos/hamer/blob/main/LICENSE.md), but it still requires MANO and separately downloaded checkpoints/front-end assets | Code is permissive; the complete runtime is **not cleared** by the code license alone |
| Grounding DINO | [Apache-2.0](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) | Permissive, subject to notices and model-card/weight verification |
| SAM 2 | [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) | Permissive, subject to notices and applicable-use terms |
| LeRobot | [Apache-2.0](https://github.com/huggingface/lerobot/blob/main/LICENSE) | Permissive, subject to notices |
| GMR reference implementation | [MIT](https://github.com/YanjieZe/GMR/blob/master/LICENSE) | Permissive code; robot assets and input motion retain their own terms |

Every Hugging Face/S3/customer import also retains its source dataset and recording rights.
The adapter's ability to read an object is not permission to train on, redistribute, or
sublicense it.

## Technical evidence

The repository-level acceptance gates pass:

- The current local unit suite passes (`312 passed, 4 skipped`); the GitHub tests, contracts,
  and infrastructure jobs are green; frozen schema v5 and both import-boundary contracts pass.
- Dashboard API: 11 tests passed. The Alembic database is at migration head
  `0003_subject_consent_events`.
- Dashboard: ESLint and the Next.js production build passed; all 18 routes were generated.
- A consent-cleared 59-second, 1,770-frame egocentric video was evaluated on an NVIDIA A100.
  Source, perception, and canonical coverage were all 1,770/1,770 with no configured frame
  cap. Privacy scanned all 1,770 frames, action labeling completed, and perception confidence
  remained unknown. LeRobot and RLDS each exported and natively reloaded 619 valid human-space
  records; a real CUDA ACT optimizer step completed with finite loss and non-zero gradients.
- Franka validation was correctly ineligible (89.7536% IK convergence, 11 collision frames,
  246 excessive temporal jumps), so no robot trajectory was persisted or exported.

See [GPU_VALIDATION_2026-08-10.md](GPU_VALIDATION_2026-08-10.md) for exact timings, the
cache/recovery chain, evidence hashes, and limitations. This acceptance proves the technical
workflow and honesty gates; it does **not** clear commercial model rights or establish
customer-grade accuracy.

## GPU decision

The human-space processing and packaging path is now NVIDIA-GPU acceptance tested on commit
`92499388c26d7fd361ed5de93817bfb3bca27236` using an A100 PCIe 40 GB. The exact harness and
dependency integrity gate are reproducible, and every final bundle checksum was verified.

This changes the technical answer, not the licensing answer: the run used research/non-
commercial model assets, and its Franka candidate was physically ineligible and withheld.
The assembled runtime therefore remains unsuitable for a commercial Figure license.

## Work required before a commercial offer

1. Establish Actuate ownership: company/author copyright, employee and contractor invention
   assignments, contributor provenance, and a deliberate proprietary or dual-license policy.
2. Replace the WiLoR/MANO/UniDepth/Ultralytics chain with commercially permissive code and
   weights, or obtain signed commercial agreements from every relevant rights holder.
3. Generate an SBOM and ship complete attribution/notice material. Add CI policy that blocks
   unknown, copyleft-incompatible, and non-commercial runtime/model artifacts.
4. Record model name, weight digest, version, source, license class, and commercial-clearance
   evidence in every run manifest. Do not infer rights from a package's code license.
5. Convert the proven A100 harness into a locked, reproducible container and repeat it cold on
   the replacement/commercially cleared model chain. Add concurrent admission-control and
   multi-run soak tests; the current evidence is one consent-cleared source plus recovery runs.
6. Benchmark on licensed ground truth: rig classification, stereo rejection, hand pose,
   metric depth/calibration, object recall, grasp signal, task accuracy, multi-episode
   segmentation, redaction recall, and retarget physics. The uncertainty and abstention rules
   follow [calibration](https://arxiv.org/abs/1706.04599) and
   [selective prediction](https://arxiv.org/abs/1901.09192), but citations are not validation.
7. Put source-footage ownership, per-subject consent, privacy/biometric processing, deletion,
   security, warranty, IP indemnity, and permitted training/redistribution terms into the
   commercial contract and data-processing agreement.

## Safe deal shape today

The defensible near-term offer is a paid evaluation of the **Actuate-authored, model-agnostic
control plane and canonical/certification interfaces**, using customer-supplied or separately
commercially cleared perception outputs. Do not include the current WiLoR/MANO/UniDepth
runtime or any artifacts generated by that runtime in a commercial delivery until counsel has
the corresponding written grants.
