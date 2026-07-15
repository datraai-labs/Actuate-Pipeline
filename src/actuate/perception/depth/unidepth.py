"""UniDepthV2 -- metric depth + estimated intrinsics + per-pixel uncertainty (Master Spec §L1).

This is the load-bearing model of Phase 3. Everything else was blocked on it:

  * WiLoR gives an excellent hand, but places it by solving a weak-perspective camera
    against a breathing bounding box -- 20.7 mm/frame of depth jitter against 0.1 mm
    laterally. It infers depth from apparent size; it never measures it.
  * SLAM's "independent" vision rotation is decomposed from an essential matrix, which
    needs the camera intrinsics. We had none, so we guessed 82 deg HFOV -> fx=1104.
  * v1's whole metric-3D path lifted 2D landmarks by a monocular depth model with 28 mm/frame
    of jitter, producing a wrist trajectory that is literally white noise.

UniDepthV2 supplies all three missing pieces from the image alone: metric depth (not
relative), the camera intrinsics, and a per-pixel confidence that feeds certification.

### The intrinsics were the hidden bug

On the real capture UniDepthV2 estimates **fx~=660** (median over frames; a single frame
can read ~550) for a 1920-wide frame -- a ~111 deg horizontal FOV, i.e. a wide-angle
action-cam lens, which is exactly what a head-mounted egocentric rig uses. **Our assumed
82 deg HFOV (fx=1104) was ~1.7x too large.** A wrong focal length scales every essential-
matrix rotation and every back-projection: with fx=1104 the vision rotation ran 24% larger
than the gyro; with the real fx it matches to 1.00. That is why nothing agreed with anything.

### VRAM (measured, RTX 2050, 4.29 GB)

    ViT-L (vitl14): 1.46 GB weights (fp16), 3.64 GB peak, 0.65 GB headroom  -- FITS, tight
    ViT-S (vits14): 0.14 GB weights                                          -- fallback

fp32 does not fit. Models are run sequentially, never co-resident, so each gets the card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from actuate.config import Provenance, RigType

_VITL = "lpiccinelli/unidepth-v2-vitl14"
_VITS = "lpiccinelli/unidepth-v2-vits14"


@dataclass
class DepthFrame:
    depth_m: np.ndarray          # (H, W) metric depth, metres
    #: (H, W) per-pixel confidence. NOT normalised to [0, 1] -- on the real capture it
    #: ranges ~0.5 to ~98. Treat it as a relative weight (higher = more reliable), not a
    #: probability; sample_depth() uses it only for weighting, never as an absolute cutoff.
    confidence: np.ndarray
    intrinsics: np.ndarray       # (3, 3) ESTIMATED from the image


@dataclass
class DepthResult:
    frames: dict[int, DepthFrame] = field(default_factory=dict)
    #: One K for the whole session: intrinsics are a property of the LENS, not the frame.
    #: Per-frame estimates are averaged, which is both more accurate and more honest than
    #: pretending the focal length changes 30 times a second.
    intrinsics: np.ndarray | None = None
    model: str = ""
    provenance: dict[str, Provenance] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def focal_px(self) -> float:
        return float((self.intrinsics[0, 0] + self.intrinsics[1, 1]) / 2)

    def hfov_deg(self, width: int) -> float:
        return float(np.degrees(2 * np.arctan((width / 2) / self.intrinsics[0, 0])))


class UniDepthEstimator:
    """Loads once. fp16 is what makes ViT-L fit; falls back to ViT-S rather than OOM."""

    def __init__(self, variant: str = "auto", device: str = "cuda") -> None:
        import torch

        from unidepth.models import UniDepthV2

        self._torch = torch
        self._device = device

        order = [_VITL, _VITS] if variant == "auto" else [
            {"vitl": _VITL, "vits": _VITS}[variant]
        ]
        last: Exception | None = None
        for name in order:
            try:
                torch.cuda.empty_cache()
                self.model = UniDepthV2.from_pretrained(name).to(device).eval().half()
                self.name = name
                return
            except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover -- hardware path
                torch.cuda.empty_cache()
                last = exc
        raise RuntimeError(
            f"no UniDepthV2 variant fits on this GPU. Last error: {last}. "
            "Run on a T4/A100, or reduce the input resolution."
        ) from last

    def predict(self, rgb: np.ndarray) -> DepthFrame:
        torch = self._torch
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(self._device).half()
        with torch.no_grad():
            out = self.model.infer(t)
        return DepthFrame(
            depth_m=out["depth"].squeeze().float().cpu().numpy(),
            confidence=out["confidence"].squeeze().float().cpu().numpy(),
            intrinsics=out["intrinsics"].squeeze().float().cpu().numpy(),
        )


def sample_depth(
    depth: np.ndarray,
    confidence: np.ndarray,
    xy: np.ndarray,
    patch: int = 7,
    min_conf: float = 0.0,
) -> tuple[float, float]:
    """Metric depth at a 2D keypoint. Robust, not a single pixel lookup.

    ### Why a patch and not `depth[y, x]`

    v1 read depth at the single MediaPipe wrist pixel. That pixel jitters ~9 px/frame, and a
    9-pixel walk across a depth map that has an EDGE at the hand boundary swings the depth
    enormously -- regardless of how good the depth model is. Roughly half of v1's 28 mm/frame
    "depth noise" is a sampling bug, not a model failure.

    So: take a confidence-weighted MEDIAN over a small patch. The median is what kills the
    edge problem -- a mean would still be dragged by background pixels that are metres away.

    Returns (depth_m, confidence).
    """
    h, w = depth.shape
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    r = patch // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan"), 0.0

    d = depth[y0:y1, x0:x1].ravel()
    c = confidence[y0:y1, x0:x1].ravel()
    m = np.isfinite(d) & (d > 0) & (c >= min_conf)
    if not m.any():
        return float("nan"), 0.0

    # Weighted median: sort by depth, walk the cumulative confidence to the halfway point.
    dv, cv = d[m], c[m]
    order = np.argsort(dv)
    dv, cv = dv[order], cv[order]
    cum = np.cumsum(cv)
    if cum[-1] <= 0:
        return float(np.median(dv)), 0.0
    k = int(np.searchsorted(cum, cum[-1] / 2.0))
    return float(dv[min(k, len(dv) - 1)]), float(cv.mean())


def solve_root_depth(
    depth: np.ndarray,
    confidence: np.ndarray,
    keypoints_2d: np.ndarray,   # (21, 2) image pixels
    keypoints_3d: np.ndarray,   # (21, 3) root-relative metric, wrist = index 0
    patch: int = 5,
) -> float:
    """One root depth per frame from the WHOLE hand, not one wrist pixel.

    ### Why fit the hand cloud

    Sampling depth at the single wrist pixel inherits UniDepthV2's full per-frame noise
    (~13 mm at 0.58 m, measured against static background points that physically cannot move).
    That noise is LARGER than the real per-frame hand motion (1.7-10 mm), so the wrist z is
    below the signal floor.

    But WiLoR's hand is metrically self-consistent: it gives each of 21 keypoints a
    root-relative depth offset `dz_i` we trust. So each keypoint i independently PREDICTS the
    wrist depth: `Z_wrist_i = D(kp_i) - dz_i`, where `D(kp_i)` is the depth map read at that
    keypoint. Robust-averaging those 21 predictions (confidence-weighted median) averages down
    the depth-map noise using the one thing WiLoR is good at -- hand shape.

    This is the fit half of the "fit hand cloud + smooth" root-depth path. It measurably
    improves the trajectory (per-frame coherence chance -> cos +0.75 after smoothing) but does
    NOT fully reconstruct it at the action-chunk horizon: the monocular depth noise floor is
    the binding constraint. See STATUS.md, Phase 3 Part C. Returns NaN if no keypoint resolves.
    """
    dz = keypoints_3d[:, 2] - keypoints_3d[0, 2]     # root-relative depth per keypoint (m)
    preds, weights = [], []
    for j in range(len(keypoints_2d)):
        if not np.isfinite(keypoints_2d[j, 0]):
            continue
        d, c = sample_depth(depth, confidence, keypoints_2d[j], patch=patch)
        if np.isfinite(d):
            preds.append(d - dz[j])
            weights.append(c)
    if len(preds) < 3:
        return float("nan")

    v = np.asarray(preds)
    w = np.asarray(weights)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.median(v))
    k = int(np.searchsorted(cum, cum[-1] / 2.0))
    return float(v[min(k, len(v) - 1)])


def smooth_root_depth(z: np.ndarray, window: int = 7) -> np.ndarray:
    """Temporal low-pass on the per-frame root depth. The smooth half of the fit+smooth path.

    Hand dynamics are smooth over a ~0.25 s window; the depth-map noise is not. A moving
    average over `window` frames trades noise for a little lag. NaN gaps are linearly
    interpolated for filtering, then restored as NaN where the input was NaN.

    This is not free: the kernel manufactures short-range autocorrelation, so per-frame
    velocity-coherence flatters itself at short horizons. Judge the result at the action-chunk
    horizon, not at k=1.
    """
    z = np.asarray(z, dtype=np.float64)
    idx = np.where(np.isfinite(z))[0]
    if len(idx) < window:
        return z.copy()
    filled = np.interp(np.arange(len(z)), idx, z[idx])
    # Normalised moving average: divide the box-filtered signal by the box-filtered ones, so
    # the window SHRINKS at the array boundaries instead of averaging against zero-padding.
    # A plain convolve(..., "same") zero-pads and makes the first/last ~window/2 frames read
    # artificially shallow -- which are exactly the frames the wrist placement needs.
    ker = np.ones(window)
    num = np.convolve(filled, ker, mode="same")
    den = np.convolve(np.ones_like(filled), ker, mode="same")
    smoothed = num / den
    out = z.copy()
    out[idx] = smoothed[idx]
    return out


def backproject(xy: np.ndarray, depth_m: float, K: np.ndarray) -> np.ndarray:
    """2D pixel + metric depth + REAL intrinsics -> metric 3D in the camera frame.

    This is the step that was impossible before: it needs a focal length, and ours was a
    guess that turned out to be 2x wrong.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (float(xy[0]) - cx) * depth_m / fx
    y = (float(xy[1]) - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=np.float64)


def run(
    session_dir: Path,
    model: str = "auto",
    rig: RigType = RigType.HEAD_MOUNTED,
    frames: list[int] | None = None,
    max_frames: int | None = None,
) -> DepthResult:
    """`perception.depth.run(store, model="auto")` -- Master Spec §L1.

    `auto` picks by rig: monocular rigs get UniDepthV2; a stereo rig would get
    FoundationStereo, which is an interface stub (we have no stereo capture to validate it
    against, and shipping an unvalidated stereo path would be exactly the kind of claim this
    project refuses to make).
    """
    import json

    import cv2

    if rig is RigType.STEREO and model in ("auto", "foundation_stereo"):
        raise NotImplementedError(
            "FoundationStereo (the spec's stereo depth model) is not implemented. We have no "
            "stereo capture to validate it against, and an unvalidated depth path is worse "
            "than an absent one. Interface reserved; see Master Spec §L1."
        )

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    n = int(meta["frame_count"])
    if max_frames:
        n = min(n, max_frames)
    want = set(frames) if frames is not None else set(range(n))

    video = session_dir / "redacted_compressed.mp4"
    if not video.exists():
        video = session_dir / "compressed.mp4"

    variant = {"auto": "auto", "unidepth_v2": "auto", "vitl": "vitl", "vits": "vits"}[model]
    est = UniDepthEstimator(variant=variant)

    res = DepthResult(model=est.name)
    Ks = []
    cap = cv2.VideoCapture(str(video))
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        if i not in want:
            continue
        df = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res.frames[i] = df
        Ks.append(df.intrinsics)
    cap.release()

    if Ks:
        # Intrinsics belong to the LENS, not the frame. Averaging is both more accurate and
        # more honest than pretending the focal length changes 30 times a second.
        res.intrinsics = np.median(np.stack(Ks), axis=0)

    res.provenance = {
        # A metric depth model IS the primary source on a monocular rig. It is not a
        # hardware measurement -- there is no depth sensor here -- but it is not a heuristic
        # either.
        "depth": Provenance.VISION_PRIMARY,
        "depth.uncertainty": Provenance.VISION_PRIMARY,
        # Estimated from the image, not read from a calibration file. Better than our guess
        # by a factor of two, and still an estimate.
        "camera.intrinsics": Provenance.VISION_PRIMARY,
    }
    res.notes = {
        "intrinsics": (
            "ESTIMATED from the image by UniDepthV2, not read from a calibration file. On "
            "the real capture the per-frame median is fx~660 for a 1920-wide frame = ~111 "
            "deg HFOV, a wide-angle egocentric lens. Our previous ASSUMPTION of 82 deg HFOV "
            "(fx=1104) was ~1.7x too large, which scaled every essential-matrix rotation "
            "(24% too large vs the gyro; ~1.00 with the real fx) and every back-projection."
        ),
        "benchmark_gap": (
            "Master Spec §7 item 1 marks the depth model BENCHMARK-BEFORE-LOCK and notes "
            "EgoInfinity uses MoGe-2 + FLOW3R + GeoCalib, NOT UniDepthV2. That benchmark has "
            "not been run. UniDepthV2 is the spec's stated default and is what is validated "
            "here; it is not proven to be the best choice."
        ),
    }
    return res
