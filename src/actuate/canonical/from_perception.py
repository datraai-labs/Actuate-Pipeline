"""Build a CanonicalEpisode from Phase 3 perception outputs -- the L1/L2 -> L3 wiring.

`build_episode` (build.py) reads v1's MediaPipe JSON and has no MANO. This builds a v3 episode
from the actual Phase 3 perception results (WiLoR hands with the full 45 MANO, UniDepth, SLAM,
fusion), so the LeRobot exporter emits the 53-dim state the schema-v3 bump added.

### What is and isn't carried, and why

- **hands.mano (45 axis-angle) + keypoints_3d + wrist_pose** -- carried. The wrist is placed at
  METRIC depth (Part C's `solve_root_depth` + smoothing), so the action is a real metric delta,
  not the bbox pseudo-depth WiLoR infers from apparent size. Stamped vision_primary.
- **camera_pose** -- from SLAM. Rotation is the trustworthy gyro-derived part; translation is
  not metric, so the pose is stamped no higher than its weakest component.
- **interaction_state** -- from the L2 fusion state machine. vision_fallback on a bare-hand rig.
- **contact / finger_joints_*** -- NOT carried on a bare-hand rig. The schema forbids it: this
  rig measures no contact, and vision-inferred "contact" is not a contact reading (§frame.py
  validators). Fusion's soft contact confidence is a visualization signal, not a schema channel.

So the canonical episode is honest about provenance while still carrying the full MANO pose the
retarget (§L5) needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from actuate.config import (
    ConsentStatus,
    ControlMode,
    FingerActionRepr,
    PiiStatus,
    Provenance,
    RigType,
    Side,
)
from actuate.schema import (
    SE3,
    CanonicalEpisode,
    CanonicalFrame,
    DepthRef,
    HandState,
    ImageRef,
    MANOParams,
    MaskRef,
    ObjectState,
)


def _write_depth_artifact(
    depth, frame_ids: list[int], artifact_dir: Path, cam: str, intrinsics: np.ndarray
):
    """Persist dense metric depth that would otherwise disappear with the process cache.

    ``DepthRef.frame_index`` indexes the first dimension of this NPZ chunk. The original
    source-frame IDs are stored alongside it so the mapping remains explicit and lossless.
    UniDepth's second raster is named ``confidence`` rather than being mislabeled as a
    calibrated uncertainty probability.
    """
    selected = [(i, depth.frames.get(i)) for i in frame_ids]
    selected = [(i, f) for i, f in selected if f is not None]
    if not selected:
        return {}, None
    shapes = {tuple(np.asarray(f.depth_m).shape) for _, f in selected}
    if len(shapes) != 1:
        raise ValueError(f"depth frames do not share one raster shape: {sorted(shapes)}")

    rel = Path("artifacts") / "depth" / f"{cam}.npz"
    dst = Path(artifact_dir) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        frame_indices=np.asarray([i for i, _ in selected], dtype=np.int64),
        depth_m=np.stack([np.asarray(f.depth_m, dtype=np.float32) for _, f in selected]),
        confidence=np.stack(
            [np.asarray(f.confidence, dtype=np.float32) for _, f in selected]
        ),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        model_intrinsics=np.asarray(depth.intrinsics, dtype=np.float64),
        model=np.asarray(str(getattr(depth, "model", ""))),
    )
    refs = {
        source_i: DepthRef(uri=rel.as_posix(), frame_index=chunk_i)
        for chunk_i, (source_i, _) in enumerate(selected)
    }
    return refs, rel.as_posix()


def _write_object_artifact(objects, artifact_dir: Path) -> str | None:
    """Persist labels, detector scores, boxes, and metric centroids not present in schema.

    The canonical schema carries each track's mask but intentionally has no position-only
    object pose (an SE3 would falsely imply a measured orientation). This sidecar preserves
    the useful partial result without weakening that invariant.
    """
    if objects is None or not getattr(objects, "frames", None):
        return None
    rel = Path("artifacts") / "perception" / "objects.json"
    dst = Path(artifact_dir) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coordinate_frame": "camera",
        "position_units": "metres",
        "orientation_available": False,
        "tracks": {str(k): v for k, v in getattr(objects, "tracks", {}).items()},
        "frames": {
            str(i): [
                {
                    "track_id": int(o.track_id),
                    "label": str(o.label),
                    "score": float(o.score),
                    "bbox_xyxy_px": [float(v) for v in o.bbox],
                    "position_cam_m": (
                        None if o.position_cam is None
                        else [float(v) for v in o.position_cam]
                    ),
                }
                for o in frame_objects
            ]
            for i, frame_objects in sorted(objects.frames.items())
        },
        "provenance": {
            k: (v.value if hasattr(v, "value") else str(v))
            for k, v in getattr(objects, "provenance", {}).items()
        },
        "notes": dict(getattr(objects, "notes", {})),
    }
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rel.as_posix()


def _wrist_pose_from(hand, root_cam: np.ndarray) -> SE3:
    """SE3 wrist pose: metric position (root_cam) + orientation from WiLoR global_orient."""
    go = np.asarray(hand.global_orient, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(go))
    if angle < 1e-12:
        q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q_xyzw = np.r_[go / angle * np.sin(angle / 2.0), np.cos(angle / 2.0)]
    return SE3(
        position_m=(float(root_cam[0]), float(root_cam[1]), float(root_cam[2])),
        quaternion_wxyz=(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])),
    )


def build_from_perception(
    session_dir: Path,
    capture_hash: str,
    *,
    hands,                         # HandResult (WiLoR)
    depth=None,                    # DepthResult (UniDepth) -- for metric wrist placement
    fusion=None,                   # FusionReport (L2) -- for interaction_state
    slam=None,                     # SlamResult -- for camera_pose
    objects=None,                  # ObjectResult (GDINO+SAM2) -- masks per frame
    rig: RigType = RigType.HEAD_MOUNTED,
    task: str | None = None,
    episode_id: str | None = None,
    consent: ConsentStatus = ConsentStatus.PENDING,
    pii_status: PiiStatus = PiiStatus.PENDING,
    max_frames: int | None = None,
    artifact_dir: Path | None = None,
    video_uri: str | None = None,
) -> CanonicalEpisode:
    """Assemble a schema-v3 CanonicalEpisode (MANO 45 populated) from perception results."""
    from actuate.perception.depth import backproject, smooth_root_depth, solve_root_depth

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    fps = float(meta.get("fps_nominal", meta.get("fps", 30.0)))
    eid = episode_id or f"{capture_hash[:16]}_ep00"
    from actuate.config import get_rig

    camera_names = get_rig(rig).cameras
    if not camera_names:
        raise ValueError(f"rig {rig.value!r} declares no camera names")
    cam = camera_names[0]
    if video_uri is None:
        try:
            from actuate.ingest.run import _session_video

            video_uri = _session_video(session_dir).resolve().as_uri()
        except FileNotFoundError:
            # Synthetic/unit sessions may intentionally carry no pixels. Production paths
            # always resolve the real source above.
            video_uri = f"processed/{meta['session_id']}/redacted_compressed.mp4"

    K = None
    if depth is not None:
        from actuate.ingest.run import load_camera_matrix

        K = load_camera_matrix(session_dir)
        if K is None:
            K = depth.intrinsics

    # The canonical clock describes the measured capture, not only the instants where a
    # hand detector happened to fire.  WiLoR intentionally stores entries only for frames
    # containing detections, while UniDepth stores every processed source frame.  Using the
    # hand-result keys here silently dropped all no-hand frames (999/1770 on a real A100
    # validation run), made otherwise-contiguous footage look sparse, and disabled action
    # annotation.  Preserve every frame measured by any dense perception channel; an empty
    # ``hands`` mapping is the honest representation of "no hand detected".
    frame_ids = sorted(
        set(hands.frames)
        | (set(depth.frames) if depth is not None else set())
        | (set(objects.frames) if objects is not None else set())
    )
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    depth_refs: dict[int, DepthRef] = {}
    depth_artifact = None
    if depth is not None and artifact_dir is not None:
        depth_refs, depth_artifact = _write_depth_artifact(
            depth, frame_ids, Path(artifact_dir), cam, K
        )
    objects_artifact = (
        _write_object_artifact(objects, Path(artifact_dir))
        if objects is not None and artifact_dir is not None
        else None
    )

    # per-frame root depth (metric wrist placement) -- Part C
    root_z: dict[int, float] = {}
    if depth is not None and K is not None:
        n = (max(frame_ids) + 1) if frame_ids else 0
        raw = np.full(n, np.nan)
        for i in frame_ids:
            hs = hands.frames.get(i, [])
            df = depth.frames.get(i)
            if not hs or df is None:
                continue
            h = next((x for x in hs if x.side == Side.RIGHT), hs[0])
            raw[i] = solve_root_depth(df.depth_m, df.confidence, h.keypoints_2d, h.keypoints_3d)
        sm = smooth_root_depth(raw)
        root_z = {i: float(sm[i]) for i in frame_ids if np.isfinite(sm[i])}

    notes: dict[str, str] = {
        "source": "Phase 3 perception (WiLoR MANO + UniDepth + SLAM + L2 fusion), schema v3.",
        "wrist_placement": (
            "Metric: wrist root depth solved from UniDepth at the hand keypoints (Part C "
            "solve_root_depth + smoothing), back-projected with estimated intrinsics -- NOT "
            "WiLoR's bbox pseudo-depth. Still monocular, so translation is rough (see STATUS "
            "Part C), but it is measured depth, not apparent-size inference."
        ),
        "contact": (
            "NOT carried: a bare-hand rig measures no contact, and the schema forbids a "
            "vision-inferred contact from masquerading as one. Fusion's soft contact is viz-only."
        ),
    }
    hand_model = str(getattr(hands, "notes", {}).get("hand_model") or "wilor").lower()
    if hand_model == "wilor":
        hand_constraint = (
            "WiLoR models are CC-BY-NC-ND-4.0; MANO's standard grant is "
            "non-commercial; the WiLoR detector also uses Ultralytics."
        )
    else:
        hand_constraint = (
            f"{hand_model} produced MANO output; MANO's standard grant is non-commercial, "
            "and every downloaded checkpoint/front-end asset requires separate review."
        )
    # State the active models' current terms directly rather than trusting cache-era notes.
    # A cached result may predate a wording correction; its legal provenance does not change
    # when it is deserialised.
    usage_constraints = [hand_constraint]
    if depth is not None:
        usage_constraints.append(
            "UniDepth software and published weights are CC-BY-NC-4.0."
        )
    if usage_constraints:
        notes["usage_constraints"] = (
            " ".join(usage_constraints)
            + " INTERNAL RESEARCH ONLY unless separate commercial rights for the complete "
            "runtime chain have been signed."
        )
    if depth_artifact is not None:
        notes["depth_artifact"] = (
            f"Dense metric depth and UniDepth relative confidence are stored in "
            f"{depth_artifact}; DepthRef.frame_index indexes its first array dimension."
        )
        notes["depth_confidence"] = (
            "The real UniDepthV2 relative-confidence raster is preserved in the depth NPZ. "
            "It is not a calibrated probability and is therefore withheld from the unit-"
            "interval certificate score until a labeled calibration set exists."
        )
    if objects_artifact is not None:
        notes["objects_artifact"] = (
            f"Object labels, detection scores, 2D boxes, and depth-derived metric centroids "
            f"are stored in {objects_artifact}. Canonical objects carry the masks; orientation "
            "remains absent because no CAD mesh/FoundationPose result exists."
        )

    frames = []
    for k, i in enumerate(frame_ids):
        hs = hands.frames.get(i, [])
        hand_states: dict[Side, HandState] = {}
        provenance: dict[str, Provenance] = {}
        confidence: dict[str, float] = {}

        for h in hs:
            if i in root_z and K is not None:
                root_cam = backproject(h.keypoints_2d[0], root_z[i], K)
            else:
                root_cam = np.asarray(h.keypoints_3d[0], dtype=np.float64)
            kp = np.asarray(h.keypoints_3d, dtype=np.float64)
            abs_kp = root_cam + (kp - kp[0])
            hand_states[h.side] = HandState(
                mano=MANOParams(
                    betas=tuple(float(x) for x in np.asarray(h.betas).reshape(-1)[:10]),
                    theta=tuple(float(x) for x in np.asarray(h.hand_pose).reshape(-1)[:45]),
                    global_orient=tuple(float(x) for x in np.asarray(h.global_orient).reshape(3)),
                ),
                keypoints_3d=tuple(tuple(float(v) for v in row) for row in abs_kp),
                wrist_pose=_wrist_pose_from(h, root_cam),
            )
        if hand_states:
            provenance["hands"] = Provenance.VISION_PRIMARY
            hand_scores = [
                float(h.detection_confidence)
                for h in hs
                if getattr(h, "detection_confidence", None) is not None
                and np.isfinite(h.detection_confidence)
            ]
            if hand_scores:
                confidence["hands"] = float(np.clip(np.mean(hand_scores), 0.0, 1.0))

        frame_depth = {}
        df = depth.frames.get(i) if depth is not None else None
        if i in depth_refs and df is not None:
            frame_depth[cam] = depth_refs[i]
            provenance["depth"] = Provenance.VISION_PRIMARY

        cam_pose = None
        if slam is not None and getattr(slam, "poses", None) and i < len(slam.poses):
            cam_pose = slam.poses[i]
            if cam_pose is not None:
                dropout = getattr(slam, "interpolated_over_dropout", None)
                synthesized = (
                    dropout is not None and i < len(dropout) and bool(dropout[i])
                )
                # A pose spanning a sensor dropout is synthesized, not measured. Otherwise
                # it remains vision_fallback because translation is not metric.
                provenance["camera_pose"] = (
                    Provenance.APPROXIMATED if synthesized else Provenance.VISION_FALLBACK
                )

        # Fusion is keyed by the original source-frame ID.  Positional indexing becomes
        # incorrect as soon as the canonical clock includes frames without hand detections.
        fused = fusion.frames.get(i) if fusion is not None else None
        istate = fused.interaction_state if fused is not None else None
        if istate is not None:
            provenance["interaction_state"] = Provenance.VISION_FALLBACK

        # Grasp is a real L2-derived state signal, not a default. It lives in the confidence
        # map for schema-v5 compatibility; the vector builder refuses frames where it is absent.
        if fused is not None:
            grasp_values = [float(v) for v in fused.grasp.values() if np.isfinite(v)]
            if grasp_values:
                confidence["grasp"] = float(np.clip(np.mean(grasp_values), 0.0, 1.0))

        # Objects: carry the SAM2 MASK per tracked object. Pose stays None -- 6-DoF needs
        # FoundationPose (a mesh + a bigger GPU), which is stubbed; a position-only pose would
        # fabricate an orientation the schema's SE3 implies. The mask + object identity per
        # frame is the honest, real signal (what is being manipulated).
        obj_states: dict[str, ObjectState] = {}
        if objects is not None:
            object_frames = objects.frames.get(i, [])
            for o in object_frames:
                obj_states[str(o.track_id)] = ObjectState(
                    mask=MaskRef(rle=o.mask_rle), pose=None
                )
            if obj_states:
                provenance["objects"] = Provenance.VISION_PRIMARY
                confidence["objects"] = float(np.clip(
                    np.mean([o.score for o in object_frames]), 0.0, 1.0
                ))

        frames.append(
            CanonicalFrame(
                t=float(i / fps),
                rig=rig,
                episode_id=eid,
                frame_idx=i,
                images={
                    name: ImageRef(uri=video_uri, frame_index=i)
                    for name in camera_names
                },
                camera_pose=cam_pose,
                hands=hand_states,
                objects=obj_states,
                depth=frame_depth,
                interaction_state=istate,
                confidence=confidence,
                provenance=provenance,
            )
        )

    return CanonicalEpisode(
        episode_id=eid,
        capture_id=capture_hash,
        source_content_hash=capture_hash,
        rig=rig,
        frames=tuple(frames),
        task=task,
        control_mode=ControlMode.EE,
        finger_action_repr=FingerActionRepr.RELATIVE,
        consent=consent,
        pii_status=pii_status,
        derivation_notes=notes,
    )
