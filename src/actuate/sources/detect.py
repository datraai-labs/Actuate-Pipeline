"""Auto-detection + smart defaults so messy real-world data runs without manual prep
(Phase 6 Part D).

Four detectors, each with a safe, stated fallback -- an autodetector that guesses wrong
silently is worse than one that defaults conservatively and says so:

- `detect_rig(session)`     -- rig from video geometry / directory layout.
- `detect_layout(session)`  -- single video / stereo / multi-cam / lerobot / rlds.
- `normalize_filenames(dir)`-- spaces, parens, unicode -> pipeline-safe names (the Kaggle bug).
- `auto_task(session, ...)` -- one VLM call on a keyframe to name the task, or None.

Consent is not inferred here. Local/self-hosted execution says where computation happens; it
does not prove that every recorded subject granted permission.
"""

from __future__ import annotations

from pathlib import Path

from actuate.config import ConsentStatus
from actuate.ingest.layout import detect_layout, detect_rig
#: filename characters the perception path trips on -> replaced with '_'.
_UNSAFE = ' ()[]{}&,;'


def normalize_filenames(directory: Path) -> list[tuple[str, str]]:
    """Rename files with spaces/parens/unicode to pipeline-safe names. Returns the renames.

    The perception stages and OpenCV paths choke on `My Video (1).mp4`; this is the Kaggle
    filename bug, generalised. Idempotent -- already-safe names are left alone.
    """
    directory = Path(directory)
    renames: list[tuple[str, str]] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        safe = p.name
        for ch in _UNSAFE:
            safe = safe.replace(ch, "_")
        safe = safe.encode("ascii", "ignore").decode() or p.name  # drop non-ascii
        while "__" in safe:
            safe = safe.replace("__", "_")
        if safe != p.name and not (directory / safe).exists():
            p.rename(directory / safe)
            renames.append((p.name, safe))
    return renames


def local_consent_default() -> ConsentStatus:
    """Compatibility helper: local processing defaults to PENDING, never auto-granted."""
    return ConsentStatus.PENDING


# NOTE: task auto-detection needs the VLM (actuate.language), which sits ABOVE this layer in
# the import graph -- so `auto_task` lives in actuate.sdk, not here. sources.detect stays
# pure geometry/filesystem: rig, layout, filenames, consent.
