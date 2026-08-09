"""Actuate — multimodal capture to certified, retargeted, VLA-training-ready robot data.

Library-shaped core (Implementation Spec §1.1). Architecture: docs/architecture/.

High-level SDK (Phase 6):

    import actuate
    actuate.login()                                 # one-time, local by default
    run = actuate.process("./my_video.mp4", task="pick up cup")
    run.export("lerobot_v3", path="./dataset/")
"""

from __future__ import annotations

import os as _os

# Force `transformers` onto its PyTorch backend for the whole process. Left alone it also
# imports TensorFlow when TF is installed, and TF here needs protobuf>=6.31 while perception
# is pinned to protobuf<6 (wandb/mediapipe) -> a hard VersionError. WiLoR imports transformers
# during the hands stage, BEFORE the objects module loads, so the guard must be set here (at
# first `import actuate`) to beat every transformers import. Our models are all pure PyTorch.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

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
    "ExportResult",
    "ProcessingRun",
    "__version__",
    "login",
    "process",
    "process_and_export",
]
