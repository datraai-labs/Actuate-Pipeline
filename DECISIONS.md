## 2026-08-14 JarvisLabs access and private checkpoint
Status: APPROVED
Decision: Register a dedicated local SSH public key for GPU execution and publish only the Corpus B semantic V1 work to the current private GitHub branch with a draft PR against master.
Why: Complete a real GPU smoke test while keeping credentials out of the repository and preserving an isolated, reviewable checkpoint.
Rejected: Storing secrets in the repo; pushing directly to master; including unrelated workspace files.

## 2026-08-14 Corpus B VLM evidence behavior
Status: APPROVED
Decision: Use visible_fact/inference/unknown claims, two blind observers per prototype window, a provisional 1,000 ms boundary-disagreement trigger, Qwen/Qwen3-VL-4B-Instruct plus HuggingFaceTB/SmolVLM2-2.2B-Instruct, and keep matching inferences model_proposed until human review.
Why: Maximize prototype recall and expose disagreement without allowing model agreement to become factual certification.
Rejected: Self-reported model confidence as truth; one-observer prototype; model-only verification of inferred task or completion.

## 2026-08-14 Corpus B semantic search and VLM stack
Status: APPROVED
Decision: Use selected Cosmos Curator stages with Qwen3-VL embedding/reranking/VLM models, Grounding DINO + SAM 2, and Datra-owned Parquet/Postgres/pgvector artifacts; use GPU compute when needed.
Why: Reuse scalable video curation while retaining commercial control, immutable evidence, adaptive diagnosis, and provider-neutral outputs.
Rejected: VSS as a production dependency; fixed VLM sampling; external APIs receiving unredacted footage; LangGraph as the runtime; automatic motion/aesthetic deletion.

## 2026-08-14 Corpus B raw-delivery V1
Status: APPROVED
Decision: Build one immutable Corpus B artifact graph with deterministic evidence, adaptive VLM intelligence, and independent verification/packaging loops; promote sessions into video-only, indexed, or validated video+IMU releases.
Why: Ship a comparable processed corpus quickly while retaining raw evidence, enabling rich diagnosis, and keeping VLMs outside rights, sensor, synchronization, calibration, and final-acceptance authority.
Rejected: Three independent pipelines per commercial tier; fixed VLM sampling frequency; overwriting raw media, timestamps, or sensor values.
