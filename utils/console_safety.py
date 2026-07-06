"""
DatraAI Pipeline — Console Safety
Prevents UnicodeEncodeError crashes when printing symbols like "✓"/"⚠" on a
legacy-codepage console (observed on Windows outside a UTF-8 terminal).

run_pipeline.py installs this globally when it's the entry point, so any
step module it dynamically loads is already protected. Scripts that can
also be invoked standalone (`python scripts/some_stage.py --session ...`,
bypassing run_pipeline.py) should call install() in their own
`if __name__ == "__main__":` guard — see scripts/03b_privacy_redact.py for
the pattern.
"""

import sys


class SafeStreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", "ascii") or "ascii"
            safe_data = data.encode(encoding, errors="replace").decode(encoding)
            self.stream.write(safe_data)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def install() -> None:
    """Wrap sys.stdout/sys.stderr so console-unsafe characters degrade instead of crashing."""
    if sys.stdout and not isinstance(sys.stdout, SafeStreamWrapper):
        sys.stdout = SafeStreamWrapper(sys.stdout)
    if sys.stderr and not isinstance(sys.stderr, SafeStreamWrapper):
        sys.stderr = SafeStreamWrapper(sys.stderr)
