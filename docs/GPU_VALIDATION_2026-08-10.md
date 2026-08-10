# NVIDIA A100 end-to-end validation — 2026-08-10

This record covers the consent-cleared acceptance run on an NVIDIA A100 PCIe 40 GB. It is
technical evidence, not an accuracy benchmark, a privacy guarantee, or commercial model-rights
clearance. Raw capture, redacted video, dense depth, canonical hand data, and model weights are
deliberately excluded from Git.

## Result

The end-to-end human-space data path passed on pipeline commit
`92499388c26d7fd361ed5de93817bfb3bca27236` with `git_dirty=false`:

| Gate | Measured result |
|---|---|
| Hardware | NVIDIA A100-PCIE-40GB; CUDA 12.4; Torch 2.5.0 |
| Dependency integrity | `pip check`: no broken requirements |
| Source | 59.0 s, 30 fps, 1280×720, 1,770 frames |
| Perception coverage | 1,770 / 1,770, full-source mode, no configured cap |
| Canonical coverage | frame IDs 0–1,769; 1,770 / 1,770 retained |
| Privacy pass | 1,770 frames scanned; 653 detected face regions blurred; status `passed` |
| Action annotation | 4 intervals over both hands; 0 flagged; no sparse-sampling skip |
| Certificate | quality 3/5; perception confidence `null` (uncalibrated) |
| Franka validation | ineligible; 89.7536% IK, 11 collision frames, 246 temporal jumps |
| Delivery behavior | failed robot trajectory withheld; human-space data only |
| LeRobot v3 | 619 frames, one episode, all 619 loaded by LeRobot |
| ACT training gate | one real CUDA optimizer step; finite loss 82.7736; grad norm 1337.2826 |
| RLDS | 619 steps, one episode, all 619 loaded by TFDS |

The 619 exported training records are fewer than the 1,770 canonical frames because the
human-state/action exporter requires usable measured successors; this is explicit filtering,
not a hidden source-processing cap. No robot action space was included because the physics
verdict was false.

Perception confidence being `null` is the intended result. WiLoR detector output and
UniDepth's relative confidence raster are useful internal signals, but neither has been
calibrated as a probability on this capture domain. Publishing `1.0` would be a fabricated
measurement. See [RESEARCH_BASIS.md](RESEARCH_BASIS.md).

## Execution chain and discovered defects

Acceptance used content-addressed perception caches after the first complete A100 inference
pass. This is stated explicitly so the evidence is not misread as one cold, uninterrupted run:

1. The cold A100 pass ran WiLoR, UniDepth, Grounding DINO, and SAM2 over all 1,770 frames.
   Perception took about 14 minutes 30 seconds. OpenCV 5 then exposed a missing Haar-cascade
   runtime asset; OpenCV is now constrained to the supported 4.x line and redaction fails with
   a clear preflight error if the classifier asset is unavailable.
2. The recovery run proved full perception coverage but exposed that canonical assembly kept
   only 771 hand-detection frames. The canonical clock now uses all measured dense-perception
   frames and aligns fusion by source-frame ID.
3. The corrected pipeline run reused the unchanged, content-addressed 1,770-frame perception
   artifacts, scanned/redacted all frames, rebuilt 1,770 canonical frames, labeled actions,
   validated Franka motion, certified, packaged, and visualized in 1,192.26 seconds.
4. Export validation first exposed a TorchCodec/system-FFmpeg dependency and then a mixed
   system/venv Hugging Face stack. The final harness uses an isolated LeRobot environment,
   the supported PyAV backend, and a mandatory clean `pip check`. LeRobot/ACT and RLDS/TFDS
   then passed.

This recovery chain tests cache integrity and crash recovery as well as the normal stages. It
does not replace a future cold-image benchmark on commercially cleared models.

## Evidence custody

The ignored local evidence tree contains the exact executed scripts, managed-run logs,
manifests, schemas, reports, native export outputs, and checksums. The primary essential bundle
has SHA-256:

`54dcc5f2ba7187e01ef2b4844b94477e2501136c250ce7337b488308d4ff04ce`

The supplemental exact-script/log bundle has SHA-256:

`0fddddfb1a830fada7cb5d31ddb2388133111360b8f506e8468cf5a2de04cae4`

The reproducible, media-free harness is maintained separately in the private
`yug-space/jarvislabs-actuate-demo` repository. The JarvisLabs machine was paused after the
downloads and independently reported `Paused`, with zero active runtime cost.

## What this proves—and what it does not

It proves that the current egocentric human-space pipeline can process the complete source
clock, preserve uncertainty, redact detected face regions, fail closed on unsafe robot motion,
and produce LeRobot/RLDS datasets that their native loaders consume.

It does **not** prove:

- calibrated hand/depth accuracy, metric-depth scale, or redaction recall;
- a commercially deployable robot trajectory—the measured Franka candidate failed physics;
- permission to use the current perception models commercially; or
- production throughput, concurrency, or reliability across a representative customer corpus.

Those remain explicit release gates rather than being inferred from one successful run.
