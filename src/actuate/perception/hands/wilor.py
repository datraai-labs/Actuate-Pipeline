"""WiLoR -> MANO hand pose (Master Spec §L1).

Replaces MediaPipe's 2D-ish landmarks with a real 3D hand reconstruction: MANO shape (beta),
pose (theta), 21 metric 3D keypoints, and a 778-vertex mesh. MANO is the embodiment-agnostic
intermediate the whole retargeting path (§L5) is built on, and MediaPipe cannot produce it.

--------------------------------------------------------------------------------------
LICENCE -- INTERNAL RESEARCH ONLY
--------------------------------------------------------------------------------------
WiLoR's published models are **CC-BY-NC-ND-4.0**. MANO is a Max Planck body model whose
standard grant is **non-commercial**. Actuate cannot use this path for a commercial customer
without separate rights from both relevant licensors; WiLoR also depends on Ultralytics,
whose closed-source commercial terms must be cleared separately. Tracked in
docs/COMMERCIAL_LICENSE_READINESS.md.

--------------------------------------------------------------------------------------
WHAT THIS MODEL IS GOOD AT, AND WHAT IT IS NOT -- measured on the real capture
--------------------------------------------------------------------------------------
Per-frame jitter of the estimated wrist position:

    x:  0.1 mm      y:  0.2 mm      z (DEPTH):  20.7 mm

The hand is excellent laterally and in articulation (2D keypoints move 3.0 px/frame,
coherently). **Depth is the one broken axis**, and the cause is structural, not a bug:
WiLoR solves a weak-perspective camera against the hand's bounding box, so

    depth  ~  focal_virtual / (scale * bbox_width)

and the detector re-runs every frame with no tracking, so the box breathes ~6.8% and the
depth breathes with it (corr = -0.31). It is not measuring depth; it is inferring it from
apparent size.

It also reports `scaled_focal_length = 37500 px` for a 1920x1080 frame. That is not a lens
-- it is HaMeR's *virtual* focal (5000 x img/256). Root translation is metric **in that
virtual camera**; rescaling by a real focal recovers a plausible ~1.0 m hand depth. So the
absolute placement of the hand depends entirely on a focal length we do not have.

=> This module emits the hand's ARTICULATION and its 2D keypoints as trustworthy, and marks
   the root depth as VISION_FALLBACK. Part C (UniDepthV2: metric depth + estimated
   intrinsics) supplies what is actually missing. B and C are not independent.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from actuate.config import Provenance, Side
from actuate.perception.sampling import sampled_indices

#: HaMeR/WiLoR convention: virtual focal 5000 for a 256 px crop, scaled to the image.
#: Root translation is metric in THIS camera, not in the real one.
_VIRTUAL_FOCAL_BASE = 5000.0
_VIRTUAL_CROP = 256.0


@dataclass
class HandFrame:
    """One hand, one frame."""

    side: Side
    #: MANO shape. Same person -> should be near-constant across frames.
    betas: np.ndarray               # (10,)
    #: MANO pose, axis-angle per joint. Schema v3 stores this FULL 45 (flattened) directly;
    #: to_theta_pca() is only for a consumer that wants the compressed 15-PCA form.
    hand_pose: np.ndarray           # (15, 3)
    global_orient: np.ndarray       # (3,)
    #: 21 keypoints, root-relative, metric MANO space. TRUSTWORTHY.
    keypoints_3d: np.ndarray        # (21, 3)
    #: 21 keypoints in image pixels. TRUSTWORTHY (3 px/frame, coherent).
    keypoints_2d: np.ndarray        # (21, 2)
    #: Root translation in WiLoR's VIRTUAL camera. Depth here is inferred from apparent
    #: size, not measured -- see the module docstring.
    root_translation_virtual: np.ndarray  # (3,)
    virtual_focal: float
    bbox: np.ndarray                # (4,) xyxy
    # Detector score in [0,1]. None means the backend did not expose a score; absence must
    # never be upgraded to certainty by a downstream default.
    detection_confidence: float | None

    def root_translation(self, real_focal_px: float) -> np.ndarray:
        """Rescale the root out of the virtual camera into a real one.

        Requires the TRUE focal length. Without it there is no way to place the hand in
        metric space, which is exactly the gap Part C closes.
        """
        return self.root_translation_virtual * (real_focal_px / self.virtual_focal)


@dataclass
class HandResult:
    frames: dict[int, list[HandFrame]] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    n_scanned: int = 0
    n_with_hands: int = 0


class WiLoREstimator:
    """Loads once, runs many frames. fp16 is what makes this fit in 4 GB."""

    def __init__(self, device: str = "cuda", dtype: str = "float16") -> None:
        import torch

        from actuate.perception.hands.mano_compat import patch_smplx

        patch_smplx()
        wilor_logger = logging.getLogger("WiLorHandPose3dEstimationPipeline")
        quiet = os.getenv("ACTUATE_VERBOSE", "0") != "1"
        wilor_logger.setLevel(logging.ERROR if quiet else logging.INFO)
        wilor_logger.disabled = quiet

        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        td = torch.float16 if dtype == "float16" else torch.float32
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")

        # Ultralytics 8.1 checkpoints contain a serialized PoseModel. Starting with
        # PyTorch 2.6, torch.load defaults to weights_only=True and rejects that trusted
        # legacy checkpoint before WiLoR can initialize. Restrict the compatibility
        # override to this constructor: these files are downloaded by the pinned WiLoR
        # package, while every unrelated torch.load call keeps PyTorch's safe default.
        original_torch_load = torch.load

        def _load_trusted_wilor_checkpoint(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = _load_trusted_wilor_checkpoint
            # Measured on an RTX 2050 (4.29 GB): 1.50 GB weights, 2.44 GB peak,
            # 1.85 GB spare. fp32 would not fit.
            self._pipe = WiLorHandPose3dEstimationPipeline(
                device=torch.device(device), dtype=td
            )
            # The third-party constructor resets its logger. Re-apply the customer-facing
            # setting after initialization so per-frame INFO/WARNING output stays behind
            # ``--verbose``.
            wilor_logger.setLevel(logging.ERROR if quiet else logging.INFO)
            wilor_logger.disabled = quiet
        finally:
            torch.load = original_torch_load

    def predict(self, frame_bgr: np.ndarray) -> list[HandFrame]:
        out = []
        for h in self._pipe.predict(frame_bgr):
            wp = h["wilor_preds"]
            out.append(
                HandFrame(
                    side=Side.RIGHT if h.get("is_right") else Side.LEFT,
                    betas=np.asarray(wp["betas"][0], dtype=np.float64),
                    hand_pose=np.asarray(wp["hand_pose"][0], dtype=np.float64),
                    global_orient=np.asarray(wp["global_orient"][0][0], dtype=np.float64),
                    keypoints_3d=np.asarray(wp["pred_keypoints_3d"][0], dtype=np.float64),
                    keypoints_2d=np.asarray(wp["pred_keypoints_2d"][0], dtype=np.float64),
                    root_translation_virtual=np.asarray(
                        wp["pred_cam_t_full"][0], dtype=np.float64
                    ),
                    virtual_focal=float(wp["scaled_focal_length"]),
                    bbox=np.asarray(h["hand_bbox"], dtype=np.float64),
                    detection_confidence=(
                        float(h["hand_processed_conf"])
                        if h.get("hand_processed_conf") is not None
                        else None
                    ),
                )
            )
        return out


def mediapipe_hand_presence(video: Path, n_frames: int) -> np.ndarray:
    """Two-pass activity scan (Master Spec §L1 5.4): a cheap CPU pre-filter.

    MediaPipe is fast and runs on CPU. It tells us which frames contain a hand, so WiLoR --
    which costs ~220 ms/frame on this GPU -- never runs on dead time (setup, idle stretches,
    the demonstrator looking away). Compute scales with manipulation content, not recording
    duration.

    MediaPipe is kept for exactly this and nothing else. Its 3D output is what Phase 3 is
    replacing.
    """
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )
    present = np.zeros(n_frames, dtype=bool)
    cap = cv2.VideoCapture(str(video))
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        present[i] = res.multi_hand_landmarks is not None
    cap.release()
    hands.close()
    return present


def estimator_for(model: str, device: str):
    """Build a hand-pose estimator exposing `.predict(frame_bgr) -> list[HandFrame]`.

    A single seam so the primary and the fallback are constructed the same way and the
    frame-scan loop stays estimator-agnostic (and so tests can inject a stub).
    """
    if model == "wilor":
        # WiLoR's grid sampler has no CPU fp16 kernel. CUDA keeps fp16 for memory;
        # local CPU execution must use fp32.
        return WiLoREstimator(
            device=device,
            dtype="float16" if device == "cuda" else "float32",
        )
    if model == "hamer":
        from actuate.perception.hands.hamer import HaMeREstimator

        return HaMeREstimator(device=device)
    raise NotImplementedError(
        f"hand model {model!r} is not implemented. Supported: 'wilor' (primary) and "
        "'hamer' (the detection fallback)."
    )


def _scan(video: Path, indices: list[int], present: np.ndarray, estimator,
          result: HandResult) -> HandResult:
    """Run one estimator over the sampled frames, filling `result` in place.

    Shared by the primary and fallback passes so sampling/seek/prefilter behave identically
    for both -- the only thing that changes is which model does `.predict`.
    """
    n = len(indices)
    contiguous = indices == list(range(n))
    cap = cv2.VideoCapture(str(video))
    for i in indices:
        if not contiguous:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)   # seek when sampling sparsely
        ok, frame = cap.read()
        if not ok:
            if contiguous:
                break                             # sequential EOF -> done
            continue                              # a bad seek skips one frame, not the run
        if not present[i]:
            continue
        hands = estimator.predict(frame)
        if hands:
            result.frames[i] = hands
            result.n_with_hands += 1
    cap.release()
    return result


def run(
    session_dir: Path,
    model: str = "wilor",
    device: str = "auto",
    max_frames: int | None = None,
    prefilter: bool = True,
    fallback: str | None = "hamer",
) -> HandResult:
    """`perception.hands.run(store, model="wilor")` -- Master Spec §L1.

    ### The HaMeR fallback (`fallback="hamer"`, default)

    WiLoR's detector (a YOLO hand detector) has a real detection envelope: on several
    egocentric datasets it returns **0 hands** on footage that plainly contains them. HaMeR
    reaches the hands through a **different detector** (ViTDet person detection -> ViTPose
    -> hand crops), so it recovers hands WiLoR's detector misses. When the primary pass finds
    hands in **zero** scanned frames and a `fallback` is configured, we re-scan with it.

    This does NOT change the licence position: HaMeR is also MANO-based, so its output is
    equally non-deliverable without an MPI commercial licence. The fallback buys *detection
    coverage*, not deliverability -- those were always two separate blockers, and this closes
    only the first. Provenance/notes record which model actually produced each frame.
    """
    import json

    if device == "auto":
        try:
            import torch
        except ImportError:
            # Keep orchestration and dependency-isolation tests usable from the light
            # install. A real estimator will still raise its actionable perception-extra
            # error when constructed below.
            device = "cpu"
        else:
            # MPS currently reaches a non-contiguous-view failure inside WiLoR. CPU fp32 is
            # slower but is the verified portable path on Apple silicon.
            device = "cuda" if torch.cuda.is_available() else "cpu"

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    frame_count = int(meta["frame_count"])
    # sample EVENLY across the clip, not the first max_frames -- otherwise a short cap only
    # sees the opening seconds and misses hands that enter later (shared with depth).
    indices = sampled_indices(frame_count, max_frames)
    n = len(indices)

    from actuate.ingest.run import _session_video

    video = _session_video(session_dir)

    present = (
        mediapipe_hand_presence(video, frame_count) if prefilter
        else np.ones(frame_count, dtype=bool)
    )

    est = estimator_for(model, device)
    result = HandResult(n_scanned=n)
    result.provenance = {
        # A metric hand-mesh model IS the primary source available on a bare-hand rig.
        "hands.mano": Provenance.VISION_PRIMARY,
        "hands.keypoints_3d": Provenance.VISION_PRIMARY,
        # ...but the ROOT DEPTH is inferred from apparent size, not measured.
        "hands.root_depth": Provenance.VISION_FALLBACK,
    }
    result.notes = {
        "licence": (
            "WiLoR models are CC-BY-NC-ND-4.0; MANO's standard grant is non-commercial; "
            "the WiLoR detector also uses Ultralytics. INTERNAL RESEARCH ONLY unless "
            "separate commercial rights for the complete chain have been signed."
        ),
        "root_depth": (
            "WiLoR solves a weak-perspective camera against the hand bbox, so depth ~ "
            "focal/(scale*bbox). The detector re-runs per frame with no tracking, the box "
            "breathes ~6.8%, and depth breathes with it: 20.7 mm/frame of jitter versus "
            "0.1 mm laterally. It infers depth from apparent size; it does not measure it. "
            "Part C (UniDepthV2) supplies real metric depth and real intrinsics."
        ),
        "prefilter": (
            "MediaPipe (CPU) gates which frames WiLoR runs on, so GPU cost tracks "
            "manipulation content rather than recording duration (Master Spec §L1 5.4)."
        ),
    }

    result.notes["hand_model"] = model
    _scan(video, indices, present, est, result)

    # WiLoR's detector found nothing -- fall through to HaMeR's (different) detector before
    # declaring the clip hand-free. Only worth it when the primary is empty: a partial WiLoR
    # result is already trustworthy and mixing two models' roots per clip would be worse.
    if result.n_with_hands == 0 and fallback and fallback != model:
        try:
            fb_est = estimator_for(fallback, device)
        except (NotImplementedError, ImportError, RuntimeError) as exc:
            result.notes["fallback"] = (
                f"{model} found 0 hands; {fallback} fallback unavailable ({exc}). "
                f"Install it on the GPU box to recover hands {model}'s detector misses.")
            return result
        _scan(video, indices, present, fb_est, result)
        result.notes["hand_model"] = fallback if result.n_with_hands else model
        result.notes["fallback"] = (
            f"{model} detected 0 hands; re-scanned with {fallback} -> "
            f"{result.n_with_hands} frames with hands. Same MANO non-commercial licence.")
    return result


#: MANO's PCA basis, loaded once. Schema v3 stores the full 45 axis-angle values WiLoR emits;
#: this basis is only for consumers that want to project down to a compressed form themselves.
_MANO_PCA: tuple[np.ndarray, np.ndarray] | None = None


def _mano_pca_basis() -> tuple[np.ndarray, np.ndarray]:
    global _MANO_PCA
    if _MANO_PCA is None:
        import wilor_mini

        from actuate.perception.hands.mano_compat import load_mano_pkl

        pkl = (
            Path(wilor_mini.__file__).parent
            / "pretrained_models"
            / "MANO_RIGHT.pkl"
        )
        m = load_mano_pkl(pkl)
        _MANO_PCA = (
            np.asarray(m["hands_components"], dtype=np.float64),  # (45, 45)
            np.asarray(m["hands_mean"], dtype=np.float64),        # (45,)
        )
    return _MANO_PCA


def to_theta_pca(hand_pose: np.ndarray, n: int = 15) -> np.ndarray:
    """Project WiLoR's 45 axis-angle values onto MANO's first `n` PCA components.

    Schema v3 stores the FULL 45 axis-angle, so this is NOT the storage format -- it is a
    convenience for a downstream consumer that deliberately wants the compressed form.

    ### Why least-squares, not a transpose

    MANO's `hands_components` are NOT orthonormal (measured: ||C C^T - I|| = 0.999), so the
    matrix transpose is the WRONG inverse -- an earlier version used it and reported a 45->45
    round-trip error of 1.66 rad (a full-rank projection must be lossless) and an inflated
    per-joint loss. The correct projection onto the span of the top-n components is a
    least-squares solve. Verified: at n=45 the round-trip is lossless (residual ~1e-13).

    It IS lossy for n < 45: on the real capture the top-15 subspace loses **median 10.3 deg,
    p90 17.3 deg, p99 22.5 deg** per joint. That is why the schema carries the full 45 (v3);
    the discarded tail is low-variance, so a consumer may still accept it for a smaller vector.
    """
    comps, mean = _mano_pca_basis()
    theta = np.asarray(hand_pose, dtype=np.float64).reshape(-1)  # (45,)
    coeffs, *_ = np.linalg.lstsq(comps[:n].T, theta - mean, rcond=None)
    return coeffs                                                # (n,)


def from_theta_pca(theta_pca: np.ndarray) -> np.ndarray:
    """Reconstruct 45 axis-angle from PCA coefficients -- MANO's own convention.

    `theta = mean + coeffs @ components[:n]`. Exact inverse of the least-squares `to_theta_pca`
    (round-trip lossless at n=45; lossy below, by design).
    """
    comps, mean = _mano_pca_basis()
    n = len(theta_pca)
    return mean + np.asarray(theta_pca, dtype=np.float64) @ comps[:n]
