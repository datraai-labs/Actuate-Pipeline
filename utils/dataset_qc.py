"""
DatraAI Pipeline — pure logic for Step 11b: Dataset-level QC (v2 addendum §11)

Kept separate from scripts/11b_dataset_qc.py's orchestration (manifest/
video/JSON I/O) so every decision rule here — near-duplicate detection,
task-imbalance ratio, diversity summary, stratified split — is unit-
testable against synthetic multi-session fixtures without needing a real
delivered batch on disk. See config.py's "DATASET-LEVEL QC" section for
why: with only one real session in this repo today, these functions can
only be verified for correctness, not validated against real dataset-scale
data.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def phash_frame(gray_frame: np.ndarray, hash_size: int = 8) -> int:
    """
    Average-hash a single grayscale frame down to hash_size x hash_size,
    returning a (hash_size**2)-bit integer. Coarse and fast — flags
    accidental duplicate uploads / re-recordings, not a fine-grained
    similarity metric (that would need an embedding model, an unjustified
    GPU dependency for a check this coarse).
    """
    small = cv2.resize(gray_frame.astype(np.float32), (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_similarity(hash_a: int, hash_b: int, bits: int = 64) -> float:
    """1.0 = identical hash, 0.0 = every bit differs."""
    xor = hash_a ^ hash_b
    differing = bin(xor).count("1")
    return 1.0 - (differing / bits)


def dedup_sessions(
    session_hashes: List[Tuple[str, int]],
    threshold: float,
    bits: int = 64,
) -> dict:
    """
    Flag near-duplicate sessions by pairwise perceptual-hash similarity.
    Processes session_hashes in the given order and keeps the FIRST
    occurrence of each near-duplicate cluster — later sessions similar
    enough to an earlier one (similarity >= threshold) are flagged removed,
    not the other way around, so re-running with the same session order
    is deterministic.

    Returns {"kept": [...], "removed": [...],
             "duplicate_of": {removed_id: kept_id},
             "similarities": {removed_id: similarity}}.
    """
    kept: List[str] = []
    kept_hashes: List[Tuple[str, int]] = []
    removed: List[str] = []
    duplicate_of: Dict[str, str] = {}
    similarities: Dict[str, float] = {}

    for session_id, h in session_hashes:
        match_id = None
        best_similarity = 0.0
        for existing_id, existing_hash in kept_hashes:
            sim = hamming_similarity(h, existing_hash, bits=bits)
            if sim >= threshold and sim > best_similarity:
                match_id = existing_id
                best_similarity = sim
        if match_id is not None:
            removed.append(session_id)
            duplicate_of[session_id] = match_id
            similarities[session_id] = round(best_similarity, 4)
        else:
            kept.append(session_id)
            kept_hashes.append((session_id, h))

    return {
        "kept": kept,
        "removed": removed,
        "duplicate_of": duplicate_of,
        "similarities": similarities,
    }


def compute_task_imbalance_ratio(task_distribution: Dict[str, int]) -> Optional[float]:
    """
    max(count) / min(count) across tasks with at least one episode.
    None (not 1.0, not 0.0) when there are fewer than 2 distinct
    represented tasks — the ratio isn't just "balanced" in that case, it's
    genuinely undefined, and returning a numeric placeholder would let a
    downstream reader mistake "undefined" for "perfectly balanced".
    """
    nonzero_counts = [c for c in task_distribution.values() if c > 0]
    if len(nonzero_counts) < 2:
        return None
    return round(max(nonzero_counts) / min(nonzero_counts), 4)


def compute_diversity_summary(
    object_class_counts: Dict[str, int],
    worker_ids: List[Optional[str]],
) -> dict:
    """
    Object-class diversity (from real class_label output — v2 §3) plus
    worker diversity (from session_meta.json's worker_id — v2 §2/§5, only
    meaningful once multiple workers' sessions exist in a batch). None
    worker_ids (no declared worker for that session) are excluded from the
    count rather than counted as one more "unknown worker" bucket.
    """
    known_worker_ids = sorted({w for w in worker_ids if w})
    return {
        "unique_object_classes": sorted(object_class_counts.keys()),
        "object_class_counts": dict(object_class_counts),
        "object_class_diversity_count": len(object_class_counts),
        "unique_worker_ids": known_worker_ids,
        "worker_diversity_count": len(known_worker_ids),
    }


def stratified_split(
    episodes: List[dict],
    split_ratios: Dict[str, float],
    stratify_key: str,
) -> Dict[str, List[str]]:
    """
    Deterministic episode-level train/val/test split, proportional within
    each stratify_key group (so a rare task's episodes are distributed
    across splits instead of all landing in one split by chance).

    Deterministic by construction (sorted episode_id order + cumulative
    ratio boundaries), not hash- or RNG-based — for QC purposes exact
    proportion adherence per group matters more than pseudo-random
    assignment, and determinism makes this directly unit-testable.

    Raises ValueError if split_ratios don't sum to 1.0 (within float
    tolerance) — a typo here should fail loudly, not silently renormalize.
    """
    ratio_sum = sum(split_ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {ratio_sum}")

    split_names = list(split_ratios.keys())
    result: Dict[str, List[str]] = {name: [] for name in split_names}

    groups: Dict[str, List[dict]] = {}
    for ep in episodes:
        key = ep.get(stratify_key, "unknown")
        groups.setdefault(key, []).append(ep)

    for key, group_episodes in groups.items():
        ordered = sorted(group_episodes, key=lambda e: e["episode_id"])
        n = len(ordered)

        cumulative = 0.0
        boundaries = []
        for name in split_names:
            cumulative += split_ratios[name]
            boundaries.append(round(cumulative * n))
        boundaries[-1] = n  # last split absorbs any rounding remainder

        start = 0
        for name, end in zip(split_names, boundaries):
            for ep in ordered[start:end]:
                result[name].append(ep["episode_id"])
            start = end

    return result
