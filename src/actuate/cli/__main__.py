"""Module entry point used by isolated export interpreters.

Keeping this tiny lets callers run ``python -m actuate.cli`` without depending on the
console-script shim from whichever environment launched the processing pipeline.
"""

from actuate.cli import app


if __name__ == "__main__":
    app()
