"""Auto-detection + smart defaults so messy real-world data runs without manual prep
(Phase 6 Part D).

Four detectors, each with a safe, stated fallback -- an autodetector that guesses wrong
silently is worse than one that defaults conservatively and says so:

- `detect_rig(session)`     -- rig from video geometry / directory layout.
- `detect_layout(session)`  -- single video / stereo / multi-cam / lerobot / rlds.
- `normalize_filenames(dir)`-- spaces, parens, unicode -> pipeline-safe names (the Kaggle bug).
- `auto_task(session, ...)` -- one VLM call on a keyframe to name the task, or None.

Consent: `local_consent_default()` returns GRANTED for local/self-hosted processing (you are
processing your OWN data). The fail-closed consent gate at the DELIVERY boundary is untouched
-- this only affects what a locally-built canonical starts with, never what may ship.
"""

from __future__ import annotations

from pathlib import Path

from actuate.config import ConsentStatus, RigType

#: width/height ratio above this = side-by-side stereo (two ~square-ish views side by side).
_STEREO_ASPECT = 1.9
#: filename characters the perception path trips on -> replaced with '_'.
_UNSAFE = ' ()[]{}&,;'


def _probe_video(video: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h


def detect_layout(session: Path) -> str:
    """single_video | stereo_video | multi_camera | lerobot | rlds | unknown.

    A present VIDEO wins: a v1 processed session carries both an .mp4 and a stray
    `session.h5`, and it is a video session -- so .h5/.hdf5 -> rlds only fires when there is
    NO video to process. LeRobot's own meta/info.json is checked first (it is unambiguous).
    """
    session = Path(session)
    if (session / "meta" / "info.json").exists():
        return "lerobot"
    videos = sorted(p for p in session.glob("*.mp4"))
    # collapse redaction/derivative variants -- `compressed.mp4` and
    # `redacted_compressed.mp4` are one camera, not two. Count DISTINCT base names.
    bases = {v.name.replace("redacted_", "").replace("_redacted", "") for v in videos}
    if len(bases) > 1:
        return "multi_camera"
    if videos:
        # prefer a non-redacted variant to probe geometry
        probe = next((v for v in videos if "redacted" not in v.name), videos[0])
        w, h = _probe_video(probe)
        if h and w / h >= _STEREO_ASPECT:
            return "stereo_video"
        return "single_video"
    if any(session.glob("*.h5")) or any(session.glob("*.hdf5")):
        return "rlds"                            # no video -> a robomimic/RLDS h5 dataset
    return "unknown"


def detect_rig(session: Path) -> str:
    """Infer the rig from the session. Falls back to head_mounted (the common egocentric
    case) rather than guessing an exotic rig -- a wrong rig changes downstream assumptions,
    so the default is the safe, most-common one."""
    layout = detect_layout(session)
    if layout == "stereo_video":
        return RigType.STEREO.value
    if layout == "multi_camera":
        return RigType.TELEOP_ROBOT.value        # multiple synced cameras = a teleop rig
    # lerobot / rlds / single_video / unknown -> head_mounted egocentric default
    return RigType.HEAD_MOUNTED.value


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
    """Local/self-hosted processing: your own data -> GRANTED. The DELIVERY gate stays
    fail-closed regardless, so this never lets un-consented data ship -- it only spares a
    developer from a PENDING block on data they own."""
    return ConsentStatus.GRANTED


# NOTE: task auto-detection needs the VLM (actuate.language), which sits ABOVE this layer in
# the import graph -- so `auto_task` lives in actuate.sdk, not here. sources.detect stays
# pure geometry/filesystem: rig, layout, filenames, consent.
