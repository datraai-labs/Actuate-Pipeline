# Corpus B V1 implementation notes

## Scope

Build the approved OSS-first semantic search and adaptive VLM evidence path around immutable Corpus B sources. No model output may alter or certify rights, source media, timestamps, IMU, synchronization, calibration, acceptance, or price.

## Plan

- Add provider-neutral semantic artifact contracts.
- Add an adaptive evidence state machine with replayable triggers and stop reasons.
- Add rebuildable cross-video index interfaces backed by the existing catalog.
- Keep Cosmos Curator and GPU models behind optional adapters.
- Present exact prompts and review rubric before any VLM execution.
- Add an optional Transformers adapter that constrains generation to the approved schema.
- Record decoding parameters, backend version, and schema hash with each model call.
- Re-run both pinned observers on the cached A30 and reject any uncited or forbidden claim.

## Verification

- Unit tests for schemas and state transitions.
- Import-boundary checks.
- A broken-version test for each correctness invariant.
- No GPU/model call before prompt approval and an accessible GPU endpoint.
- Baseline `2026-08-14`: `.venv/bin/pytest -q` reports 685 passed, 15 failed, 40 skipped. Failures predate this work: missing local `session_001` API artifacts, absent optional model/media dependencies, frozen-schema drift, and a source-loader dependency-order failure.
- Prompt proposal: both embedded JSON schemas parse and the proposal remains isolated under `.context`.
- Repository diff check: no semantic schema, prompt, adapter, CLI, migration, or inference code is wired before the exact contract is approved.
- All eight approved prompt blocks were copied verbatim into the implementation and compared byte-for-byte with the reviewed proposal.
- JarvisLabs CLI authentication uses `JL_API_KEY`; the key will be read from macOS Keychain and never stored in the repository or logged.
- Model revisions locked for the first smoke test: Qwen3-VL-4B-Instruct `ebb281ec70b05090aa6165b016eac8ec08e71b17`; SmolVLM2-2.2B-Instruct `482adb537c021c86670beed01cd58990d01e72e4`.
- Focused semantic verification: 36 tests pass; Ruff, `git diff --check`, and both import-linter contracts pass.
- Full-suite comparison after the additions: 721 passed, 15 failed, 40 skipped. The same 15 baseline failures remain; semantic work introduced no new failure.
- JarvisLabs A30 smoke: both pinned observers loaded and produced non-empty responses. Qwen took 31.106 seconds and SmolVLM2 took 19.197 seconds after model load.
- Both smoke responses were correctly rejected by `VLMRecord`: Qwen returned separate `visible_fact` / `inference` / `unknown` arrays instead of `claims` / `review_flags`; SmolVLM2 returned plain text. The visual content was plausible, but neither output is pipeline-valid.
- The JarvisLabs instance was paused after every attempt. The successful smoke result is isolated under ignored `.context/corpus-b-v1/gpu-smoke/results.json`.
- Constrained-provider verification: 43 focused semantic tests pass. The new tests reject both prior malformed response shapes, invented frame citations, forbidden authority claims, missing required fields, and inference references that do not resolve to visible facts.
- Full-suite comparison after the provider addition: 728 passed, 15 failed, 40 skipped. The same 15 baseline failures remain. Ruff, `git diff --check`, and both import-linter contracts pass.
- Cached A30 constrained smoke: direct Transformers plus LM Format Enforcer 0.11.2 exceeded the five-minute cap before Qwen produced its first record. The run was stopped, no constrained result was accepted, SmolVLM2 was not reached, and the GPU was paused.
- Status: the adapter is unit-only. The approved runtime acceptance gate of two valid, cited `VLMRecord` outputs has not passed, so this decision is not SHIPPED.

## Deviations

- This Mac has no NVIDIA GPU. JarvisLabs A30 access is now verified through a dedicated SSH key; inference runs remotely and the instance is paused between attempts.
- An initial semantic contract scaffold was removed after review because it would have frozen the still-unapproved claim schema and imported the AWS catalog into core language code.
- Cosmos Curator will run as a pinned external sidecar. Its windows and scores remain work artifacts, never canonical episode truth or delivery decisions.
- Plain Transformers generation did not obey the approved JSON envelope. Constrained decoding was approved as a follow-up architecture change.
- Direct per-token LM Format Enforcer decoding on Qwen3-VL is too slow under the current five-minute prototype cap. Changing the decoding engine, schema shape, or call structure requires a new architecture decision.
