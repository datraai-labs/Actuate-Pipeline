"""Keep customer output readable while preserving a diagnostic verbose mode."""
from __future__ import annotations

import logging
import os

_NOISY_LOGGERS = (
    "PIL",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "transformers",
    "accelerate",
    "ultralytics",
    "matplotlib",
    "tensorflow",
    "WiLorHandPose3dEstimationPipeline",
)


def configure_customer_logging(*, verbose: bool) -> None:
    """Make Actuate's stage reporter the default console surface."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["YOLO_VERBOSE"] = "true" if verbose else "false"
    os.environ["TQDM_DISABLE"] = "0" if verbose else "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if verbose else "2"
    os.environ["ACTUATE_VERBOSE"] = "1" if verbose else "0"
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, force=verbose)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)

    try:
        from transformers.utils import logging as transformers_logging

        if verbose:
            transformers_logging.set_verbosity_info()
            transformers_logging.enable_progress_bar()
        else:
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
    except ImportError:
        pass
