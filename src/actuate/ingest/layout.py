"""Pixel/layout facts used to validate rig declarations at the ingestion boundary."""
from __future__ import annotations

from pathlib import Path

from actuate.config import RigType

_STEREO_ASPECT = 1.9


def _probe_video(video: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


def detect_layout(session: Path) -> str:
    """Return a conservative file/pixel layout classification."""
    session = Path(session)
    if (session / "meta" / "info.json").exists():
        return "lerobot"
    videos = sorted(session.glob("*.mp4"))
    bases = {v.name.replace("redacted_", "").replace("_redacted", "") for v in videos}
    if len(bases) > 1:
        return "multi_camera"
    if videos:
        probe = next((v for v in videos if "redacted" not in v.name), videos[0])
        width, height = _probe_video(probe)
        if height and width / height >= _STEREO_ASPECT:
            return "stereo_video"
        return "single_video"
    if any(session.glob("*.h5")) or any(session.glob("*.hdf5")):
        return "rlds"
    return "unknown"


def detect_rig(session: Path) -> str:
    """Infer only from observable layout; exotic/unsupported rigs are never guessed."""
    layout = detect_layout(session)
    if layout == "stereo_video":
        return RigType.STEREO.value
    if layout == "multi_camera":
        return RigType.TELEOP_ROBOT.value
    return RigType.HEAD_MOUNTED.value
