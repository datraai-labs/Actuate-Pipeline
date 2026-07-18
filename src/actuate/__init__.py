"""Actuate — multimodal capture to certified, retargeted, VLA-training-ready robot data.

Library-shaped core (Implementation Spec §1.1). Architecture: docs/architecture/.

High-level SDK (Phase 6):

    import actuate
    actuate.login()                                 # one-time, local by default
    run = actuate.process("./my_video.mp4", task="pick up cup")
    run.export("lerobot_v3", path="./dataset/")
"""

from __future__ import annotations

from actuate.schema import SCHEMA_VERSION

__version__ = "2.0.0-dev"


def __getattr__(name: str):
    """Lazy SDK surface -- keeps `import actuate` cheap (no torch/rerun until you call it)."""
    if name in ("process", "process_and_export", "login", "ProcessingRun", "ExportResult"):
        from actuate import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'actuate' has no attribute {name!r}")


__all__ = [
    "SCHEMA_VERSION",
    "__version__",
    "login",
    "process",
    "process_and_export",
    "ProcessingRun",
    "ExportResult",
]
