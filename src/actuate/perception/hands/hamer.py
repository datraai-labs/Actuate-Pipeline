"""HaMeR -> MANO hand pose: the detection fallback for WiLoR (Master Spec §L1).

WHY THIS EXISTS
---------------
WiLoR reaches hands through a YOLO hand detector that, on real egocentric footage, returns
**0 detections** on clips that plainly contain hands (observed on multiple datasets). HaMeR
reaches them through a *different* front-end -- detectron2 ViTDet person detection -> ViTPose
whole-body keypoints -> hand crops -> the HaMeR transformer -- which recovers hands that
detector misses. `perception.hands.run(..., fallback="hamer")` invokes this only when WiLoR's
primary pass finds nothing, so we pay HaMeR's heavier cost only when it can actually help.

LICENCE -- changed front-end, same MANO blocker
-----------------------------------------------
HaMeR's repository code is MIT, but it emits MANO and requires separately downloaded model
assets. The standard MANO grant is non-commercial, so HaMeR does NOT by itself clear this
path for a commercial delivery. Detection coverage and deliverability remain separate.

VALIDATION STATUS -- honest
---------------------------
This wrapper is written against HaMeR's published demo API but is **validated on the GPU box
(Lightning/Linux), not in this repo's CI**: HaMeR needs detectron2 + ViTPose + ~2 GB of
weights that will not install on the 4 GB Windows dev machine. `perception.hands.run` treats
an unavailable/failed HaMeR as "fallback unavailable" and returns WiLoR's (empty) result with
a note, rather than crashing -- so a mis-set-up HaMeR degrades, it does not break the pipeline.
Output is mapped into the SAME `HandFrame` as WiLoR (HaMeR is WiLoR's parent model), with one
real difference handled below: HaMeR emits pose as rotation MATRICES, not axis-angle.
"""

from __future__ import annotations

import numpy as np

from actuate.config import Side
from actuate.perception.hands.wilor import HandFrame

#: ViTPose whole-body (133-kpt COCO-WholeBody) hand keypoint spans and the wrist/threshold
#: used by HaMeR's own demo to turn body keypoints into hand boxes.
_LEFT_HAND = slice(-42, -21)
_RIGHT_HAND = slice(-21, None)
_KP_CONF = 0.5


def _mat_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    """(...,3,3) rotation matrices -> (...,3) axis-angle. HaMeR emits matrices; WiLoR's
    HandFrame carries axis-angle (MANO's native theta), so we convert to keep them uniform."""
    import cv2

    rot = np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)
    aa = np.stack([cv2.Rodrigues(r)[0].reshape(3) for r in rot], axis=0)
    return aa


class HaMeREstimator:
    """Loads HaMeR + its detectron2/ViTPose front-end once, runs many frames.

    Mirrors `WiLoREstimator`: `.predict(frame_bgr) -> list[HandFrame]`. Heavier than WiLoR
    (a person detector + a pose model + the hand transformer), which is why it is a fallback,
    not the primary.
    """

    def __init__(self, device: str = "cuda") -> None:
        try:
            import torch
            from hamer.configs import CACHE_DIR_HAMER  # noqa: F401
            from hamer.models import DEFAULT_CHECKPOINT, load_hamer
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
            from vitpose_model import ViTPoseModel
        except ImportError as exc:  # pragma: no cover - GPU-box dependency
            raise ImportError(
                "HaMeR fallback needs the 'hamer' + 'vitpose_model' + 'detectron2' packages "
                "and HaMeR's weights. Install on the GPU box (see HaMeR's README); it will "
                "not build on the 4 GB Windows dev machine. WiLoR remains the primary.") from exc

        from actuate.perception.hands.mano_compat import patch_smplx

        patch_smplx()
        self._torch = torch
        self._device = torch.device(device)
        self._model, self._model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self._model = self._model.to(self._device).eval()

        # detectron2 ViTDet person detector (HaMeR demo's default front-end)
        from detectron2.config import LazyConfig
        from hamer.utils.utils_detectron2 import cascade_mask_rcnn_vitdet_h  # noqa: F401
        import hamer

        cfg_path = (
            __import__("pathlib").Path(hamer.__file__).parent
            / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py")
        det_cfg = LazyConfig.load(str(cfg_path))
        det_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl")
        for i in range(3):
            det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self._detector = DefaultPredictor_Lazy(det_cfg)
        self._vitpose = ViTPoseModel(self._device)

    def _hand_boxes(self, img_rgb: np.ndarray):
        """Person detect -> ViTPose -> (bboxes xyxy, is_right[]) for confident hands."""
        det = self._detector(img_rgb[:, :, ::-1])  # detectron2 wants BGR
        inst = det["instances"]
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)  # class 0 = person
        boxes = inst.pred_boxes.tensor[keep].cpu().numpy()
        scores = inst.scores[keep].cpu().numpy()
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        vitposes = self._vitpose.predict_pose(
            img_rgb, [np.concatenate([boxes, scores[:, None]], axis=1)])
        bboxes, is_right = [], []
        for vp in vitposes[0]:
            kps = vp["keypoints"]
            for hand_kp, right in ((kps[_LEFT_HAND], 0), (kps[_RIGHT_HAND], 1)):
                valid = hand_kp[:, 2] > _KP_CONF
                if valid.sum() < 3:
                    continue
                xy = hand_kp[valid, :2]
                bboxes.append([xy[:, 0].min(), xy[:, 1].min(),
                               xy[:, 0].max(), xy[:, 1].max()])
                is_right.append(right)
        return np.asarray(bboxes, dtype=np.float32), np.asarray(is_right)

    def predict(self, frame_bgr: np.ndarray) -> list[HandFrame]:
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full

        img_rgb = frame_bgr[:, :, ::-1].copy()
        boxes, right = self._hand_boxes(img_rgb)
        if len(boxes) == 0:
            return []

        ds = ViTDetDataset(self._model_cfg, frame_bgr, boxes, right, rescale_factor=2.0)
        loader = self._torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        img_h, img_w = frame_bgr.shape[:2]
        scaled_focal = (self._model_cfg.EXTRA.FOCAL_LENGTH
                        / self._model_cfg.MODEL.IMAGE_SIZE * max(img_h, img_w))

        out_frames: list[HandFrame] = []
        for batch in loader:
            batch = recursive_to(batch, self._device)
            with self._torch.no_grad():
                out = self._model(batch)
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            cam_t_full = cam_crop_to_full(
                out["pred_cam"], box_center, box_size, img_size,
                scaled_focal).cpu().numpy()

            mano = out["pred_mano_params"]
            betas = mano["betas"].cpu().numpy()
            # HaMeR predicts a right hand; left hands come from an x-flipped crop, so their
            # keypoints/root must be flipped back into image space (as HaMeR's demo does).
            hand_pose = mano["hand_pose"].cpu().numpy()          # (B,15,3,3)
            global_orient = mano["global_orient"].cpu().numpy()  # (B,1,3,3)
            kp3d = out["pred_keypoints_3d"].cpu().numpy()        # (B,21,3)
            kp2d = out["pred_keypoints_2d"].cpu().numpy()        # (B,21,2) in crop
            is_r = batch["right"].cpu().numpy().astype(bool)

            for j in range(betas.shape[0]):
                flip = 1.0 if is_r[j] else -1.0
                k3 = kp3d[j].copy(); k3[:, 0] *= flip
                out_frames.append(HandFrame(
                    side=Side.RIGHT if is_r[j] else Side.LEFT,
                    betas=betas[j].astype(np.float64),
                    hand_pose=_mat_to_axis_angle(hand_pose[j]),        # (15,3) axis-angle
                    global_orient=_mat_to_axis_angle(global_orient[j])[0],
                    keypoints_3d=k3.astype(np.float64),
                    keypoints_2d=kp2d[j].astype(np.float64),
                    root_translation_virtual=cam_t_full[j].astype(np.float64),
                    virtual_focal=float(scaled_focal),
                    bbox=boxes[len(out_frames) % len(boxes)].astype(np.float64),
                    # HaMeR's wrapper currently exposes no calibrated hand-level score.
                    # Unknown is materially different from a perfect detection.
                    detection_confidence=None,
                ))
        return out_frames
