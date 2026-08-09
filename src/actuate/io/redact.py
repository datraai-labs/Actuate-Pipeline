"""PII redaction: blur faces (and any injected regions) out of a capture video, so the
consent/PII gate can move an episode from pii_status=PENDING to PASSED (Master Spec §L4).

WHY THIS IS THE MISSING PIECE
-----------------------------
`io.consent` fail-closes delivery on `pii_status is PASSED`, and nothing in the pipeline ever
produced a PASSED: every locally-processed episode sat at PENDING, so `is_deliverable` was
always False. That is correct as a default (absence is not permission) but it meant NO episode
could ever ship. This module is the step that legitimately earns PASSED: it runs an actual
redaction pass over the video and writes `redacted_compressed.mp4` -- the exact filename the
rest of the pipeline (perception, canonical `video` pointer) already prefers.

HONEST LIMIT -- recall-bounded, not a guarantee
-----------------------------------------------
The default detector is OpenCV's Haar cascade (frontal faces). It is CPU-only and dependency-
free, but its RECALL is not 1.0: a face it misses is a face it does not blur, and PASSED would
then over-claim. So:

  * PASSED means "a redaction pass completed", NOT "provably zero PII remains".
  * The detector is injectable. A production deployment should pass a stronger one (e.g. a
    DNN face+person+screen detector on the GPU box); the gate logic is identical.
  * `redact_video` reports how many regions it blurred so a reviewer can sanity-check.

This keeps the boundary honest: the mechanism to reach PASSED now exists and is auditable,
and the quality of the guarantee is exactly the quality of the detector you give it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from actuate.config import PiiStatus

#: (x, y, w, h) pixel box.
Region = tuple[int, int, int, int]
#: A detector maps one BGR frame to the regions to blur.
Detector = Callable[["object"], list[Region]]


@dataclass
class RedactionReport:
    """What a redaction pass actually did -- auditable, not just a boolean."""

    method: str
    frames_scanned: int
    regions_blurred: int
    output: Path

    @property
    def status(self) -> PiiStatus:
        """A completed pass earns PASSED. Recall-bounded (see module docstring): this asserts
        the pass ran and any DETECTED PII was blurred, not that none was missed."""
        return PiiStatus.PASSED


def haar_face_detector() -> Detector:
    """OpenCV frontal-face Haar cascade -> face boxes. CPU-only, no extra dependency."""
    import cv2

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(frame_bgr) -> list[Region]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(24, 24))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    return detect


def _blur_region(frame, region: Region, kernel: int) -> None:
    """Gaussian-blur one ROI in place, clamped to the frame (a box off the edge must not
    silently blur nothing or throw)."""
    import cv2

    h_img, w_img = frame.shape[:2]
    x, y, w, h = region
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    k = kernel | 1  # GaussianBlur needs an odd kernel
    frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)


def _downscaled(frame, detect: Detector, detect_width: int) -> list[Region]:
    """Detect on a width-limited copy, then scale boxes back to full resolution.

    Haar cost is ~O(pixels); at 1080p that is ~0.3 s/frame (15 min for a 2850-frame clip).
    Detecting at ~640 px wide and blurring at full res cuts that ~9x with no loss of blur
    quality -- a face is just as findable small, and we blur the full-size box regardless.
    """
    import cv2

    h, w = frame.shape[:2]
    if detect_width <= 0 or w <= detect_width:
        return detect(frame)
    s = w / detect_width
    small = cv2.resize(frame, (detect_width, int(round(h / s))))
    return [(int(x * s), int(y * s), int(bw * s), int(bh * s))
            for (x, y, bw, bh) in detect(small)]


def redact_video(src: Path, dst: Path, *, detector: Detector | None = None,
                 kernel: int = 51, detect_width: int = 0) -> RedactionReport:
    """Blur every detected region in `src`, writing the redacted video to `dst`.

    Returns a `RedactionReport` whose `.status` is PASSED once the pass completes. `detector`
    defaults to the Haar face detector; inject a stronger one for a stronger guarantee.

    `detect_width` downscales the frame for detection only (blur is always full-res). It is a
    direct **speed-vs-recall** trade: on real 1080p egocentric footage, detecting at 640 px
    ran ~3x faster but recovered far FEWER faces (small/partial faces vanish when shrunk).
    Default 0 (no downscale) keeps recall; raise it only when you have measured that your
    detector still finds the faces at the smaller size. For a PII guarantee, the right lever
    is a stronger `detector`, not a smaller frame.
    """
    import cv2

    src, dst = Path(src), Path(dst)
    detect = detector or haar_face_detector()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video for redaction: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, blurred = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for region in _downscaled(frame, detect, detect_width):
            _blur_region(frame, region, kernel)
            blurred += 1
        writer.write(frame)
        frames += 1
    cap.release()
    writer.release()
    return RedactionReport(method=("custom" if detector else "haar_face"),
                           frames_scanned=frames, regions_blurred=blurred, output=dst)


def redact_session(session_dir: Path, *, detector: Detector | None = None) -> RedactionReport:
    """Redact a session's capture video -> `redacted_compressed.mp4` in the same dir.

    Picks the raw `compressed.mp4` (or the single video present) as the source, never a video
    already named `redacted_*`. The output filename is the one perception + canonical prefer,
    so downstream stages consume the redacted stream automatically.
    """
    session_dir = Path(session_dir)
    src = session_dir / "compressed.mp4"
    if not src.exists():
        cands = [v for v in sorted(session_dir.glob("*.mp4"))
                 if not v.name.startswith("redacted")]
        if not cands:
            raise FileNotFoundError(f"no source video to redact in {session_dir}")
        src = cands[0]
    report = redact_video(src, session_dir / "redacted_compressed.mp4", detector=detector)
    (session_dir / "privacy_report.json").write_text(
        json.dumps({
            "status": report.status.value,
            "method": report.method,
            "frames_scanned": report.frames_scanned,
            "regions_blurred": report.regions_blurred,
            "scope": "face regions detected by the configured detector",
            "limitation": (
                "Recall-bounded: PASSED means the redaction pass completed and detected "
                "regions were blurred; it is not proof that zero PII remains."
            ),
        }, indent=2),
        encoding="utf-8",
    )
    return report
