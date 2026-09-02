"""Stereo-eye provenance and correspondence checks for the L0 declaration gate.

Closes the declaration-vs-reality gap: a stereo capture whose eyes are separate files can
have ONE eye dropped into a session directory and ingest cleanly as `head_mounted`. The
resulting depth is physically meaningless but looks plausible, which is the whole bug class
`validate_declared_rig` exists to stop.

### Why the aspect-ratio gate cannot catch this

The gate in `layout.py` keys on width/height >= 1.9 (a side-by-side composite). Measured on
the real Panoculon Trinet corpus:

    take0003_L.mp4  1920x1080  aspect 1.7778   <- one eye of a stereo pair
    video.mp4       1920x1080  aspect 1.7778   <- genuine monocular egocentric

One eye of a stereo rig *is* a monocular image. No single-file pixel test can separate the
two, so the discriminator has to be provenance (naming + sidecars), not geometry. The
aspect gate is still correct for true side-by-side sources and is deliberately retained --
this corpus simply does not exercise it.

### Measured thresholds -- NOT guesses

Derived 2026-08-13 from `stereo_sample1` (take0003_L/R.mp4, 1920x1080, 3160 frames),
ORB(4000) + BFMatcher(NORM_HAMMING, crossCheck), Hamming distance < 40, 11 frames sampled
every 300 from frame 60. Per frame: median (dx, dy) over matched keypoints.

    condition                          rel_sd(dx)   sign-stable   max|dy| px
    ---------------------------------  ----------   -----------   ----------
    TRUE PAIR    L[i] vs R[i]               0.088       yes             8.6
    NEGATIVE A   L[i] vs L[i+45]            1.527       no            249.4
    NEGATIVE B   L[i] vs R[i+90]           13.813       no            426.2

A real rectified pair holds a tight, sign-stable horizontal baseline (mean dx -91.2 px,
sd 8.0) with near-zero vertical disparity. Anything else wanders in both.

Signals deliberately NOT used, because the same run showed they do not separate the classes:

  * **NCC** -- true pair 0.495-0.627 vs negative control 0.100 AND 0.527. The 0.527 sits
    inside the true-pair band, so NCC in isolation is ambiguous (this reproduces the
    earlier audit's 0.32-0.48 finding on different footage).
  * **ORB inlier ratio** -- true pair 0.63-0.75 vs negative 0.74-0.79. The negative scores
    HIGHER; a threshold here would invert the verdict.

Thresholds below sit ~3-4x above the observed true-pair values and ~4-10x below the
nearest negative, so the margin absorbs footage variation without approaching either class.
Re-derive them against a second stereo rig before treating them as general.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Video containers an eye file may use.
_VIDEO_EXTS = (".mp4", ".mov", ".avi")

#: Filename suffixes that mark one eye of a stereo pair, and the eye they pair with.
_EYE_PAIRS = {"l": "r", "r": "l", "left": "right", "right": "left"}

#: `<stem>_<eye>` where <eye> is one of _EYE_PAIRS. Case-insensitive.
_EYE_RE = re.compile(r"^(?P<stem>.+)_(?P<eye>l|r|left|right)$", re.IGNORECASE)

#: Sidecars the stereo rig writes alongside its eyes. Corroborating evidence only -- the
#: eye suffix is the decisive signal, these just sharpen the error message.
_STEREO_SIDECAR_EXTS = (".vts", ".tel", ".imu")

# --- correspondence thresholds (see module docstring for the measurement they come from) ---

#: Max |median dy| for a rectified pair. True pair measured 8.6 px; nearest negative 249.4.
_MAX_EPIPOLAR_DY_PX = 25.0

#: Max sd(dx)/|mean(dx)| across sampled frames. True pair measured 0.088; nearest negative
#: 1.527. A real baseline is near-constant; a non-pair's apparent shift is scene motion.
_MAX_DISPARITY_REL_SD = 0.40

#: A pair with a near-zero baseline is not a stereo pair -- it is the same view twice.
_MIN_ABS_DISPARITY_PX = 8.0

#: Matches needed before a frame's median disparity means anything.
_MIN_GOOD_MATCHES = 12

#: Frames that must yield a usable measurement before a verdict is issued.
_MIN_USABLE_SAMPLES = 4


class StereoPairError(ValueError):
    """Two files declared as a stereo pair do not actually correspond."""


@dataclass(frozen=True)
class EyeFile:
    """One video identified as an eye of a stereo pair."""

    path: Path
    stem: str          # shared prefix, e.g. "take0003"
    eye: str           # normalized lowercase: "l" | "r" | "left" | "right"
    raw_eye: str       # suffix exactly as it appears on disk, e.g. "L"

    @property
    def expected_companion(self) -> str:
        return _EYE_PAIRS[self.eye]

    @property
    def expected_companion_name(self) -> str:
        """The companion's filename, preserving this file's spelling and case.

        `take0003_L.mp4` -> `take0003_R.mp4`; `cam_left.mp4` -> `cam_right.mp4`. Naming the
        exact missing file makes the error actionable instead of merely descriptive.
        """
        want = self.expected_companion
        if self.raw_eye.isupper():
            want = want.upper()
        elif self.raw_eye[0].isupper():
            want = want.capitalize()
        return f"{self.stem}_{want}{self.path.suffix}"


@dataclass(frozen=True)
class PairEvidence:
    """Outcome of comparing two videos for genuine stereo correspondence."""

    corresponds: bool
    reason: str
    n_samples: int = 0
    mean_dx: float = 0.0
    rel_sd_dx: float = 0.0
    max_abs_dy: float = 0.0
    sign_stable: bool = False

    def summary(self) -> str:
        return (
            f"{self.n_samples} frames: mean dx {self.mean_dx:+.1f}px, "
            f"rel_sd {self.rel_sd_dx:.3f}, max|dy| {self.max_abs_dy:.1f}px, "
            f"sign_stable={self.sign_stable}"
        )


def declared_camera_stems() -> frozenset[str]:
    """Every camera name any registered rig declares, lowercased.

    The rig registry is the authority on camera naming (config/rigs.py), and some of those
    names legitimately end in an eye-like suffix -- `stereo_left` / `stereo_right` are the
    STEREO rig's declared cameras, not an orphaned pair. Those files belong to
    `verify_rig_streams`, which reports a missing camera precisely; the provenance gate must
    stand aside rather than pre-empt it with a vaguer error.
    """
    from actuate.config.rigs import RIG_REGISTRY

    return frozenset(
        cam.lower() for spec in RIG_REGISTRY.values() for cam in spec.cameras
    )


def classify_eye(path: Path) -> EyeFile | None:
    """Return eye identity if `path`'s name marks it as one eye, else None.

    A name that IS a declared camera of some rig is never an eye -- see
    `declared_camera_stems`.
    """
    stem = Path(path).stem
    if stem.lower() in declared_camera_stems():
        return None
    m = _EYE_RE.match(stem)
    if not m:
        return None
    return EyeFile(path=Path(path), stem=m.group("stem"), eye=m.group("eye").lower(),
                   raw_eye=m.group("eye"))


def find_companion_eye(eye: EyeFile, session: Path) -> Path | None:
    """Locate the opposite eye of `eye` in `session`, or None if it is absent.

    Matches on the shared stem so `take0003_L.mp4` finds `take0003_R.mp4` regardless of
    container. Both the short (`_L`) and long (`_left`) spellings are accepted, since a
    pair written by different tooling can legitimately mix them.
    """
    want = eye.expected_companion
    spellings = {want}
    # "l" pairs with "r", but the companion may be spelled "right" -- accept both forms.
    spellings |= {k for k, v in _EYE_PAIRS.items() if k.startswith(want[0]) and k != eye.eye}
    for candidate in sorted(Path(session).iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in _VIDEO_EXTS:
            continue
        other = classify_eye(candidate)
        if other and other.stem == eye.stem and other.eye in spellings:
            return candidate
    return None


def orphaned_stereo_sidecars(stem: str, session: Path) -> list[str]:
    """Stereo-rig sidecars sharing `stem` — corroborates a dropped eye in the message."""
    found: list[str] = []
    for p in sorted(Path(session).iterdir()):
        if p.is_file() and p.suffix.lower() in _STEREO_SIDECAR_EXTS and p.stem.startswith(stem):
            found.append(p.name)
    return found


def find_lone_eye(session: Path) -> EyeFile | None:
    """Return an eye file whose companion is missing, or None.

    Pure filesystem/naming inspection — deliberately runs before any pixel analysis so a
    dropped eye is caught without decoding a single frame.
    """
    session = Path(session)
    if not session.is_dir():
        return None
    for p in sorted(session.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _VIDEO_EXTS:
            continue
        eye = classify_eye(p)
        if eye and find_companion_eye(eye, session) is None:
            return eye
    return None


def _sample_disparities(left: Path, right: Path, samples: int) -> list[tuple[float, float]]:
    """Median (dx, dy) per sampled frame for frames that yield enough matches."""
    import cv2
    import numpy as np

    capL, capR = cv2.VideoCapture(str(left)), cv2.VideoCapture(str(right))
    try:
        n = min(int(capL.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                int(capR.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if n <= 0:
            return []
        orb = cv2.ORB_create(4000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Skip the first/last 2% -- capture start/stop often has motion blur or a black lead-in.
        lo, hi = int(n * 0.02), int(n * 0.98)
        step = max(1, (hi - lo) // max(1, samples))
        out: list[tuple[float, float]] = []
        for idx in range(lo, hi, step):
            if len(out) >= samples:
                break
            frames = []
            for cap in (capL, capR):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                frames.append(frame if ok else None)
            if any(f is None for f in frames):
                continue
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            ka, da = orb.detectAndCompute(grays[0], None)
            kb, db = orb.detectAndCompute(grays[1], None)
            if da is None or db is None:
                continue
            good = [m for m in matcher.match(da, db) if m.distance < 40]
            if len(good) < _MIN_GOOD_MATCHES:
                continue
            delta = (np.float32([ka[m.queryIdx].pt for m in good])
                     - np.float32([kb[m.trainIdx].pt for m in good]))
            out.append((float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))))
        return out
    finally:
        capL.release()
        capR.release()


def verify_stereo_pair(left: Path, right: Path, samples: int = 8) -> PairEvidence:
    """Decide whether two videos are a genuine rectified stereo pair.

    Keys on horizontal-disparity consistency, which is the only signal measured to separate
    the classes (see module docstring). Returns evidence rather than raising, so the caller
    decides whether a non-correspondence is fatal or a review route.
    """
    import numpy as np

    pairs = _sample_disparities(Path(left), Path(right), samples)
    if len(pairs) < _MIN_USABLE_SAMPLES:
        # Too little texture to judge. An honest "cannot verify" -- never a silent pass.
        return PairEvidence(
            corresponds=False,
            reason=(f"only {len(pairs)} of {samples} sampled frames yielded >= "
                    f"{_MIN_GOOD_MATCHES} matches; correspondence UNVERIFIABLE, not verified"),
            n_samples=len(pairs),
        )

    dx = np.array([p[0] for p in pairs])
    dy = np.array([p[1] for p in pairs])
    mean_dx = float(dx.mean())
    rel_sd = float(dx.std() / max(abs(mean_dx), 1e-6))
    max_dy = float(np.abs(dy).max())
    sign_stable = bool(np.all(np.sign(dx) == np.sign(dx[0])))
    ev = dict(n_samples=len(pairs), mean_dx=mean_dx, rel_sd_dx=rel_sd,
              max_abs_dy=max_dy, sign_stable=sign_stable)

    if not sign_stable:
        return PairEvidence(False, "horizontal disparity changes sign across frames — "
                            "scene motion, not a fixed stereo baseline", **ev)
    if abs(mean_dx) < _MIN_ABS_DISPARITY_PX:
        return PairEvidence(False, f"baseline {abs(mean_dx):.1f}px < {_MIN_ABS_DISPARITY_PX}px "
                            "— the two files are the same viewpoint, not a stereo pair", **ev)
    if max_dy > _MAX_EPIPOLAR_DY_PX:
        return PairEvidence(False, f"vertical disparity {max_dy:.1f}px > "
                            f"{_MAX_EPIPOLAR_DY_PX}px — not epipolar-aligned", **ev)
    if rel_sd > _MAX_DISPARITY_REL_SD:
        return PairEvidence(False, f"disparity rel_sd {rel_sd:.3f} > {_MAX_DISPARITY_REL_SD} "
                            "— baseline not constant", **ev)
    return PairEvidence(True, "consistent rectified baseline", **ev)
