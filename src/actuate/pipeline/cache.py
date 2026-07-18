"""Perception stage cache -- library-layer home of the sha256-keyed `.actuate_cache/` spine.

Moved out of `cli.viz` (Phase 6 refactor): the orchestration lives in the LIBRARY now so
both the CLI and the SDK can drive it without inverting the import-linter contract (cli
wraps sdk wraps pipeline, never the reverse). Behaviour is unchanged from the CLI version.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from pathlib import Path

CACHE_DIR = ".actuate_cache"


def stage_cached(session: Path, stage: str, key: str, *, use_cache: bool, force: bool,
                 run_fn: Callable, warn: Callable[[str], None] | None = None):
    """Run a perception stage, or load a matching cached result.

    Returns (result, source) where source is 'cache' or 'ran'. The cache key hashes the
    inputs that change the output (frame count, prompts), so a changed input misses the
    cache and the stage re-runs. `force` re-runs even on a hit; without `use_cache` the cache
    is neither read nor written. `warn` is an optional reporter for a best-effort write miss.
    """
    if not use_cache:
        return run_fn(), "ran"

    cache_dir = Path(session) / CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    keyhash = hashlib.sha256(f"{stage}|{key}".encode()).hexdigest()[:12]
    path = cache_dir / f"{stage}_{keyhash}.pkl"

    if path.exists() and not force:
        try:
            with path.open("rb") as fh:
                return pickle.load(fh), "cache"
        except Exception:
            pass  # corrupt/old-format cache -> fall through and re-run

    result = run_fn()
    try:
        with path.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # caching is best-effort; never fail over it
        if warn:
            warn(f"{stage}: could not write cache: {exc}")
    return result, "ran"
