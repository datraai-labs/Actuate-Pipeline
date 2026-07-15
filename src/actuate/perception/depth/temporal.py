"""Temporal / video depth -- the A/B challenger to single-image UniDepthV2 (Master Spec §L1).

Part C's finding: UniDepthV2 is a SINGLE-IMAGE model with no temporal consistency, so a static
point's depth wobbles ~2.1% frame-to-frame -- larger than the real hand motion, which is why
the wrist trajectory is below the noise floor. A model that reasons over the VIDEO should read a
static point consistently and beat that floor.

Two backends, one interface (`DepthResult`, so it drops into everything downstream):

1. **video_depth_anything** -- wraps Video-Depth-Anything (a real temporally-consistent video
   depth model). VRAM-heavy; meant for Kaggle T4, not the 4 GB dev card. The headline A/B.

2. **flow_filter** -- post-processes ANY base DepthResult (e.g. UniDepth) with an
   optical-flow-warped temporal EMA. Runs anywhere (no extra model), and is a cheap way to see
   how much of the wobble is removable by temporal filtering alone vs needs a real video model.

Score both against UniDepth with `perception.depth.consistency.compare` -- lower static-point
wobble wins. The metric is the decision.

**Honesty:** the video_depth_anything wrapper is written against VDA's documented API and is
validated ON KAGGLE, not on this 4 GB box (same discipline as the FoundationPose stub -- reserve
the seam, don't fake the run). The flow_filter backend IS validated here (unit-tested on
synthetic depth). Neither estimates camera intrinsics; K is borrowed from a UniDepth pass or
passed in, since intrinsics do not affect the temporal-consistency comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from actuate.config import Provenance
from actuate.perception.depth.unidepth import DepthFrame, DepthResult

_VDA_ENCODERS = ("vits", "vitl")


class VideoDepthEstimator:
    """Video-Depth-Anything wrapper. Loads once, infers over a whole clip (that is the point --
    it reasons temporally). Not runnable on 4 GB; targets Kaggle T4.

    The API here follows the Video-Depth-Anything reference
    (`video_depth_anything.video_depth.VideoDepthAnything.infer_video_depth`). Verify against the
    installed version -- if the signature drifted, this is the ONE place to fix it.
    """

    def __init__(self, encoder: str = "vits", metric: bool = True, device: str = "cuda") -> None:
        import torch

        if encoder not in _VDA_ENCODERS:
            raise ValueError(f"encoder must be one of {_VDA_ENCODERS}")
        try:
            from video_depth_anything.video_depth import VideoDepthAnything
        except ImportError as exc:  # pragma: no cover -- Kaggle path
            raise RuntimeError(
                "Video-Depth-Anything is not installed. On Kaggle:\n"
                "  pip install -q git+https://github.com/DepthAnything/Video-Depth-Anything\n"
                "and download the checkpoint (metric variant for metric depth). See "
                "kaggle/README.md."
            ) from exc

        cfgs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        self._torch = torch
        self.device = device
        self.metric = metric
        self.model = VideoDepthAnything(**cfgs[encoder])
        # Checkpoint path is environment-specific; the Kaggle scaffold sets VDA_CKPT.
        import os

        ckpt = os.environ.get("VDA_CKPT")
        if not ckpt:
            raise RuntimeError("set VDA_CKPT to the Video-Depth-Anything checkpoint path")
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.to(device).eval()

    def infer(self, frames_rgb: np.ndarray, fps: float = 30.0, input_size: int = 518):
        """frames_rgb: (T, H, W, 3) uint8 RGB. Returns (T, H, W) depth (metric if metric=True)."""
        depths, _ = self.model.infer_video_depth(
            frames_rgb, fps, input_size=input_size, device=self.device
        )
        return np.asarray(depths, dtype=np.float32)


def _flow_filter(base: DepthResult, video_frames: list, alpha: float = 0.5) -> DepthResult:
    """Optical-flow-warped temporal EMA of a base DepthResult. Reduces per-frame noise while
    following real motion (the flow accounts for what moved). Locally consistent, not just a
    global-scale fix -- which is what Part C's H3 showed a global fix could not do.

        d_smooth[t] = alpha * d[t] + (1 - alpha) * warp(d_smooth[t-1], flow[t-1 -> t])

    Only the warp region blends; newly-revealed pixels keep the raw depth. Runs anywhere.
    """
    import cv2

    ids = sorted(base.frames)
    n = min(len(ids), len(video_frames))
    ids = ids[:n]
    gray = [cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY) for i in ids]

    out = DepthResult(intrinsics=base.intrinsics, model=f"{base.model}+flow_filter")
    prev_smooth = None
    for t in range(n):
        df = base.frames[ids[t]]
        d = df.depth_m.astype(np.float32).copy()
        if prev_smooth is not None:
            flow = cv2.calcOpticalFlowFarneback(
                gray[t - 1], gray[t], None, 0.5, 3, 21, 3, 5, 1.2, 0
            )
            h, w = d.shape
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            map_x = xs + flow[..., 0]
            map_y = ys + flow[..., 1]
            warped = cv2.remap(prev_smooth, map_x, map_y, cv2.INTER_LINEAR,
                               borderValue=np.nan)
            valid = np.isfinite(warped) & (warped > 0)
            d[valid] = alpha * d[valid] + (1 - alpha) * warped[valid]
        prev_smooth = d.copy()
        out.frames[ids[t]] = DepthFrame(depth_m=d, confidence=df.confidence,
                                        intrinsics=df.intrinsics)
    out.provenance = dict(base.provenance)
    out.notes = dict(base.notes)
    out.notes["temporal"] = (
        f"Flow-warped temporal EMA (alpha={alpha}) over {base.model}. Reduces per-frame depth "
        "noise using optical flow to follow motion; a post-process, not a video-depth model."
    )
    return out


def run(
    session_dir: Path,
    backend: str = "video_depth_anything",
    *,
    base: DepthResult | None = None,
    intrinsics: np.ndarray | None = None,
    encoder: str = "vits",
    max_frames: int | None = None,
    alpha: float = 0.5,
) -> DepthResult:
    """Temporal depth as a `DepthResult`, interface-compatible with `perception.depth.run`.

    backend="video_depth_anything": run VDA over the clip (Kaggle). K is borrowed from
    `intrinsics` or `base.intrinsics`, else approximated -- it does not affect the temporal A/B.
    backend="flow_filter": temporally filter `base` (required) -- runs anywhere.
    """
    import json

    import cv2

    if backend not in ("flow_filter", "video_depth_anything"):
        raise ValueError(f"unknown temporal backend {backend!r}")
    if backend == "flow_filter" and base is None:
        raise ValueError("flow_filter needs a base DepthResult (e.g. UniDepth) to filter")

    session_dir = Path(session_dir)
    video = session_dir / "redacted_compressed.mp4"
    if not video.exists():
        video = session_dir / "compressed.mp4"

    meta = json.loads((session_dir / "session_meta.json").read_text())
    n = int(meta.get("frame_count", 0))
    if max_frames:
        n = min(n, max_frames)
    frames_bgr = []
    cap = cv2.VideoCapture(str(video))
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames_bgr.append(f)
    cap.release()
    n = len(frames_bgr)

    if backend == "flow_filter":
        return _flow_filter(base, frames_bgr, alpha=alpha)

    # --- Video-Depth-Anything (Kaggle) ---
    from actuate.perception.depth.unidepth import approximate_intrinsics

    K = intrinsics if intrinsics is not None else (
        base.intrinsics if base is not None else None
    )
    if K is None and frames_bgr:
        h, w = frames_bgr[0].shape[:2]
        K = approximate_intrinsics(w, h)

    est = VideoDepthEstimator(encoder=encoder)
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
    fps = float(meta.get("fps_nominal", 30.0))
    depths = est.infer(rgb, fps=fps)

    res = DepthResult(intrinsics=K, model=f"video_depth_anything_{encoder}")
    for i in range(len(depths)):
        # VDA gives no per-pixel confidence; use uniform 1.0 (honest: no uncertainty channel).
        res.frames[i] = DepthFrame(
            depth_m=depths[i], confidence=np.ones_like(depths[i], dtype=np.float32),
            intrinsics=K,
        )
    res.provenance = {"depth": Provenance.VISION_PRIMARY,
                      "camera.intrinsics": Provenance.VISION_PRIMARY}
    res.notes = {
        "model": (
            "Video-Depth-Anything (temporally consistent video depth). Intrinsics NOT estimated "
            "by this model -- borrowed/approximated. Confidence is uniform (no uncertainty "
            "channel). Compare against UniDepth with perception.depth.consistency.compare."
        )
    }
    return res
