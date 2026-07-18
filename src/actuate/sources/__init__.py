"""Source resolvers -- turn a source spec (local / hf:// / s3:// / http(s):// / openx://)
into a local session directory the pipeline can read (Phase 6 Parts A + F).

Only local resolution is wired in the first cut; the remote resolvers land with the source
step. `resolve(source, work_root)` is the single entry point the SDK calls.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve"]


def resolve(source: str, work_root: Path) -> Path:
    """Dispatch a source spec to its resolver. Remote schemes are added in the source step."""
    raise NotImplementedError(
        f"remote source resolvers (for {source!r}) are not wired yet; use a local path.")
