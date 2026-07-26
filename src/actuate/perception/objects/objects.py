"""Object detection + segmentation + tracking (+ optional 6-DoF) -- Master Spec §L1.

`perception.objects.run(store, prompts) -> ObjectResult`. The pipeline, per chunk of frames:

  1. **Grounding DINO** (open-vocab, text-prompted) proposes boxes on the chunk's first frame
     from the task prompts -- one detection pass per chunk, not per frame.
  2. **SAM2** seeds each box into a video session and PROPAGATES the mask across the chunk.
     Temporal propagation is what makes the masks consistent frame-to-frame instead of a
     detector re-firing independently and flickering.
  3. **IoU association** across chunk boundaries keeps a stable track_id per object.
  4. **Depth back-projection** (optional, Part C): the mask centroid + metric depth + real
     intrinsics give a metric 3D POSITION. Full 6-DoF needs FoundationPose, which needs a
     mesh and a bigger GPU -- see foundationpose.py; that path is an interface stub.

The GDINO + SAM2 call shapes here are the ones v1 verified on this exact GPU (RTX 2050), not
guessed. Per-object SAM2 sessions are used (the multi-object seeding order in this transformers
version is fragile); correctness over encoder-sharing efficiency.

Both models are the TINY variants and run sequentially, never co-resident:
    grounding-dino-tiny  ~0.7 GB      sam2-hiera-tiny  ~0.2 GB
so they fit in 4 GB alongside nothing else. UniDepth (for back-projection) runs in its own
pass, also never co-resident.
"""

from __future__ import annotations

import os

# Grounding DINO + SAM2 are pure PyTorch. Left to itself, `transformers` also imports
# TensorFlow/Flax if they're installed -- and TF here demands protobuf>=6.31 while perception
# is pinned to protobuf<6 (wandb/mediapipe), so that import HARD-CRASHES the objects stage.
# Tell transformers to use the torch backend only. Must be set before transformers is first
# imported (its lazy import lives in GroundingDinoDetector.__init__ below), hence module top.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from actuate.config import Provenance
from actuate.perception.objects.rle import bbox_iou, encode_rle

_GDINO_ID = "IDEA-Research/grounding-dino-tiny"
_SAM2_ID = "facebook/sam2-hiera-tiny"

#: A re-detected box within this IoU of an existing track's last box keeps that track_id.
_TRACK_MATCH_IOU = 0.3
#: Grounding DINO thresholds. Box/text thresholds from v1's validated settings.
_BOX_THRESHOLD = 0.30
_TEXT_THRESHOLD = 0.25


@dataclass
class ObjectFrame:
    """One tracked object, one frame."""

    track_id: int
    label: str
    score: float
    bbox: tuple[float, float, float, float]  # xyxy pixels
    mask_rle: str
    #: Metric 3D position in the camera frame, from mask-centroid depth back-projection.
    #: None when no depth was supplied. This is a POSITION, not a 6-DoF pose.
    position_cam: tuple[float, float, float] | None = None
    #: Full 6-DoF, only if FoundationPose ran (it does not here -- see foundationpose.py).
    pose_T_obj_cam: tuple[tuple[float, ...], ...] | None = None


@dataclass
class ObjectResult:
    frames: dict[int, list[ObjectFrame]] = field(default_factory=dict)
    #: track_id -> label, for the tracks that survived across >1 frame.
    tracks: dict[int, str] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    n_frames: int = 0


class GroundingDinoDetector:
    """Open-vocab box detector. Loads once."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(_GDINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_ID).to(
            self.device
        )
        self.model.eval()

    def detect(self, image_pil, prompts: list[str]) -> list[dict]:
        """One forward pass. Returns [{box: [x1,y1,x2,y2] px, label, score}, ...]."""
        torch = self._torch
        # Grounding DINO wants lower-cased phrases; this transformers processor takes them as
        # a per-image list of phrase strings.
        inputs = self.processor(images=image_pil, text=[[p.strip().lower() for p in prompts]],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=_BOX_THRESHOLD,
            text_threshold=_TEXT_THRESHOLD,
            target_sizes=[image_pil.size[::-1]],
        )[0]
        return [
            {"box": [float(v) for v in box], "label": str(label), "score": float(score)}
            for box, score, label in zip(
                res["boxes"].tolist(), res["scores"].tolist(), res["text_labels"]
            )
        ]


class Sam2Tracker:
    """SAM2 video mask propagation. Loads once."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = Sam2VideoProcessor.from_pretrained(_SAM2_ID)
        self.model = Sam2VideoModel.from_pretrained(_SAM2_ID).to(self.device)
        self.model.eval()

    def propagate(self, frames_pil: list, box: list[float]) -> dict[int, np.ndarray]:
        """Seed one box at chunk-frame 0, propagate across the chunk.

        One object per session -- the verified-working pattern (v1 found multi-object seeding
        in this transformers version fragile). Returns {local_frame_idx -> boolean mask}.
        """
        torch = self._torch
        import torch.nn.functional as F

        w, h = frames_pil[0].size
        out_masks: dict[int, np.ndarray] = {}
        session = self.processor.init_video_session(
            video=frames_pil, inference_device=self.device, dtype=torch.float32
        )
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1, input_boxes=[[box]]
        )
        for out in self.model.propagate_in_video_iterator(session, start_frame_idx=0):
            resized = F.interpolate(
                out.pred_masks.float(), size=(h, w), mode="bilinear", align_corners=False
            )
            out_masks[int(out.frame_idx)] = (resized[0, 0] > 0.0).cpu().numpy()
        return out_masks


def _centroid_position(
    mask: np.ndarray, depth_m: np.ndarray, confidence: np.ndarray, K: np.ndarray
) -> tuple[float, float, float] | None:
    """Metric 3D position of the masked object: robust depth over the mask, back-projected.

    Uses the mask's pixel centroid for (x, y) and a confidence-weighted median of the depth
    map OVER THE MASK for z -- averaging over the whole mask is far more stable than one pixel
    (the same lesson as the hand root-depth fit in Part C). Returns None if the mask is empty
    or no valid depth falls inside it.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cx_px, cy_px = float(xs.mean()), float(ys.mean())

    d = depth_m[ys, xs]
    c = confidence[ys, xs] if confidence is not None else np.ones_like(d)
    ok = np.isfinite(d) & (d > 0)
    if not ok.any():
        return None
    d, c = d[ok], c[ok]
    order = np.argsort(d)
    d, c = d[order], c[order]
    cum = np.cumsum(c)
    z = float(d[int(np.searchsorted(cum, cum[-1] / 2.0))]) if cum[-1] > 0 else float(np.median(d))

    fx, fy, kx, ky = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return ((cx_px - kx) * z / fx, (cy_px - ky) * z / fy, z)


def _prompts_from(prompts: list[str] | str | None, task: str | None) -> list[str]:
    """Resolve the text prompt list from an explicit list, a task string, or a default."""
    if isinstance(prompts, str):
        prompts = [p for p in prompts.replace(",", " ").split() if len(p) > 2]
    if prompts:
        return prompts
    if task:
        # crude noun-ish extraction: keep words > 3 chars, drop obvious verbs/stopwords
        stop = {"the", "with", "using", "hand", "right", "left", "perform", "grasp", "and"}
        words = [w.strip(".,").lower() for w in task.split()]
        picked = [w for w in words if len(w) > 3 and w not in stop]
        if picked:
            return picked
    # Real capture is a desk/paperwork scene.
    return ["stapler", "paper", "document", "box"]


def run(
    session_dir: Path,
    prompts: list[str] | str | None = None,
    task: str | None = None,
    device: str = "cuda",
    chunk: int = 30,
    max_frames: int | None = None,
    depth=None,          # optional DepthResult from perception.depth.run, for 3D position
    intrinsics: np.ndarray | None = None,
) -> ObjectResult:
    """`perception.objects.run(store, prompts)` -- Master Spec §L1.

    Detects with Grounding DINO once per `chunk` frames, propagates masks with SAM2 within the
    chunk, associates tracks by box IoU across chunks. If `depth` + `intrinsics` are supplied,
    each object also gets a metric 3D position via mask-centroid back-projection (Part C).
    """
    import json

    import cv2
    from PIL import Image

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    n = int(meta["frame_count"])
    if max_frames:
        n = min(n, max_frames)
    from actuate.ingest.run import _session_video

    video = _session_video(session_dir)

    prompt_list = _prompts_from(prompts, task)

    # Read the frames we will process (chunked). Kept as PIL for the models.
    cap = cv2.VideoCapture(str(video))
    frames_bgr: list = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames_bgr.append(f)
    cap.release()
    n = len(frames_bgr)
    frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

    result = ObjectResult(n_frames=n)
    result.provenance = {
        "objects.bbox": Provenance.VISION_PRIMARY,
        "objects.mask": Provenance.VISION_PRIMARY,
        # position is depth-derived, so it inherits depth's weaker standing (Part C).
        "objects.position": Provenance.VISION_FALLBACK,
        "objects.pose_6dof": Provenance.APPROXIMATED,  # not produced; reserved
    }
    result.notes = {
        "prompts": ", ".join(prompt_list),
        "pose_6dof": (
            "6-DoF ORIENTATION is not produced: FoundationPose needs a CAD mesh (absent for "
            "paperwork/stapler) and >8 GB VRAM. Only metric POSITION is emitted, via mask + "
            "depth back-projection. See perception.objects.foundationpose."
        ),
    }

    detector = GroundingDinoDetector(device=device)
    # active tracks: track_id -> {"box": last box, "label": str}
    active: dict[int, dict] = {}
    next_tid = 1

    # Detect per chunk, then hand boxes to SAM2. Detector and tracker are loaded one at a time
    # only if we could not hold both -- both tiny models fit, so keep both resident within run.
    tracker = Sam2Tracker(device=device)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_pil = frames_pil[start:end]
        dets = detector.detect(chunk_pil[0], prompt_list)
        if not dets:
            continue

        # Associate each detection to an existing track (box IoU) or open a new one. Each
        # existing track may be claimed by AT MOST ONE detection per chunk -- otherwise two
        # boxes over the same object both inherit its track_id and SAM2 propagates both,
        # doubling the masks under one id. Highest-scoring detections get first pick.
        seeded: list[dict] = []
        claimed: set[int] = set()
        for det in sorted(dets, key=lambda d: d["score"], reverse=True):
            box = tuple(det["box"])
            tid = None
            best = _TRACK_MATCH_IOU
            for t, info in active.items():
                if t in claimed:
                    continue
                iou = bbox_iou(box, info["box"])
                if iou >= best:
                    best, tid = iou, t
            if tid is None:
                tid = next_tid
                next_tid += 1
            claimed.add(tid)
            active[tid] = {"box": box, "label": det["label"]}
            seeded.append({"track_id": tid, **det})

        # SAM2 propagate each seeded object across the chunk.
        for s in seeded:
            masks = tracker.propagate(chunk_pil, s["box"])
            for local_idx, mask in masks.items():
                gidx = start + local_idx
                pos = None
                if depth is not None and intrinsics is not None and gidx in depth.frames:
                    df = depth.frames[gidx]
                    pos = _centroid_position(mask, df.depth_m, df.confidence, intrinsics)
                result.frames.setdefault(gidx, []).append(
                    ObjectFrame(
                        track_id=s["track_id"],
                        label=s["label"],
                        score=s["score"],
                        bbox=tuple(s["box"]),
                        mask_rle=encode_rle(mask),
                        position_cam=pos,
                    )
                )

    # A track counts as "tracked" only if it appears in more than one frame.
    seen: dict[int, int] = {}
    for objs in result.frames.values():
        for o in objs:
            seen[o.track_id] = seen.get(o.track_id, 0) + 1
    result.tracks = {
        tid: active[tid]["label"] for tid, c in seen.items() if c > 1 and tid in active
    }
    return result
