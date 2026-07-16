"""MoGe-2 -- per-frame metric geometry + focal estimation (Master Spec §L1 / §7 benchmark).

MoGe-2 (Microsoft, CVPR'25) recovers a metric-scale 3D point map AND the camera focal length
from a single image. It is what EgoInfinity uses for metric scale + intrinsics. It is NOT a
temporal model -- it is per-frame like UniDepth -- so it does not by itself fix temporal jitter.
Its role in the benchmark is a clean question: **does a better metric/focal ANCHOR alone help?**
Right now our focal is UniDepth's estimate (fx~660); MoGe-2 gives an independent one, and its
metric point map may be lower-noise.

Two uses:
  * a benchmark row (does the anchor alone lower wrist jitter?), and
  * the metric anchor for `temporal.anchor_scale` (a better scale reference than UniDepth).

Interface-compatible: returns the same `DepthResult` (depth_m + confidence + intrinsics).

**Honesty:** validated on Kaggle (needs the model + a GPU); the wrapper follows microsoft/MoGe's
documented `MoGeModel.infer` API. If the installed version's output keys differ, THIS is the one
place to fix. Not run on the 4 GB dev box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from actuate.config import Provenance
from actuate.perception.depth.unidepth import DepthFrame, DepthResult

_MOGE2_VITL = "Ruicheng/moge-2-vitl"


class MoGe2Estimator:
    """Loads once. Per-image metric point map + intrinsics."""

    def __init__(self, model_id: str = _MOGE2_VITL, device: str = "cuda") -> None:
        import torch

        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:  # pragma: no cover -- Kaggle path
            raise RuntimeError(
                "MoGe-2 is not installed. On Kaggle:\n"
                "  pip install git+https://github.com/microsoft/MoGe.git\n"
                "then it pulls weights from Hugging Face (Ruicheng/moge-2-vitl)."
            ) from exc

        self._torch = torch
        self.device = device
        self.model = MoGeModel.from_pretrained(model_id).to(device).eval()

    def predict(self, rgb: np.ndarray) -> DepthFrame:
        """rgb: (H, W, 3) uint8. Returns a DepthFrame with METRIC depth + estimated focal K."""
        torch = self._torch
        h, w = rgb.shape[:2]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model.infer(t)

        depth = out["depth"].float().cpu().numpy()          # (H, W) metric metres
        mask = out.get("mask")
        conf = (mask.float().cpu().numpy() if mask is not None
                else np.ones_like(depth, dtype=np.float32))

        # MoGe intrinsics are normalised (image mapped to [0,1]); denormalise to pixels.
        Kn = out["intrinsics"].float().cpu().numpy()
        K = np.array([[Kn[0, 0] * w, 0, Kn[0, 2] * w],
                      [0, Kn[1, 1] * h, Kn[1, 2] * h],
                      [0, 0, 1]], dtype=np.float64)
        return DepthFrame(depth_m=depth.astype(np.float32),
                          confidence=conf.astype(np.float32), intrinsics=K)


def run(session_dir: Path, max_frames: int | None = None, device: str = "cuda") -> DepthResult:
    """`perception.depth.run(store, model='moge2')` -- per-frame metric depth + focal."""
    import json

    import cv2

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    n = int(meta.get("frame_count", 0))
    if max_frames:
        n = min(n, max_frames)
    video = session_dir / "redacted_compressed.mp4"
    if not video.exists():
        video = session_dir / "compressed.mp4"

    est = MoGe2Estimator(device=device)
    res = DepthResult(model="moge2_vitl")
    Ks = []
    cap = cv2.VideoCapture(str(video))
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        df = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res.frames[i] = df
        Ks.append(df.intrinsics)
    cap.release()
    if Ks:
        res.intrinsics = np.median(np.stack(Ks), axis=0)
    res.provenance = {"depth": Provenance.VISION_PRIMARY,
                      "camera.intrinsics": Provenance.VISION_PRIMARY}
    res.notes = {
        "model": (
            "MoGe-2 (per-frame metric geometry + focal). NOT temporal -- benchmarks whether a "
            "better metric/focal anchor alone lowers wrist jitter, and serves as the scale "
            "anchor for temporal.anchor_scale. Intrinsics estimated per frame (median taken)."
        )
    }
    return res
