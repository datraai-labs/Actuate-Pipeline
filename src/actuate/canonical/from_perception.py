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
from scipy.spatial.transform import Rotation

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
    CanonicalEpisode,
    CanonicalFrame,
    HandState,
    ImageRef,
    MANOParams,
    MaskRef,
    ObjectState,
    SE3,
)


def _wrist_pose_from(hand, root_cam: np.ndarray) -> SE3:
    """SE3 wrist pose: metric position (root_cam) + orientation from WiLoR global_orient."""
    go = np.asarray(hand.global_orient, dtype=np.float64).reshape(3)
    q_xyzw = Rotation.from_rotvec(go).as_quat()  # (x,y,z,w)
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
) -> CanonicalEpisode:
    """Assemble a schema-v3 CanonicalEpisode (MANO 45 populated) from perception results."""
    from actuate.perception.depth import backproject, smooth_root_depth, solve_root_depth

    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    fps = float(meta.get("fps_nominal", meta.get("fps", 30.0)))
    eid = episode_id or f"{capture_hash[:16]}_ep00"
    cam = "head" if rig is RigType.HEAD_MOUNTED else "wrist"
    video = f"processed/{meta['session_id']}/redacted_compressed.mp4"
    K = depth.intrinsics if depth is not None else None

    frame_ids = sorted(hands.frames)
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

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

    states = fusion.states if fusion is not None else None
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

        cam_pose = None
        if slam is not None and getattr(slam, "poses", None) and i < len(slam.poses):
            cam_pose = slam.poses[i]
            if cam_pose is not None:
                # only as good as the weakest component; translation is not metric
                provenance["camera_pose"] = Provenance.VISION_FALLBACK

        istate = None
        if states is not None and k < len(states):
            istate = states[k]
            provenance["interaction_state"] = Provenance.VISION_FALLBACK

        # Objects: carry the SAM2 MASK per tracked object. Pose stays None -- 6-DoF needs
        # FoundationPose (a mesh + a bigger GPU), which is stubbed; a position-only pose would
        # fabricate an orientation the schema's SE3 implies. The mask + object identity per
        # frame is the honest, real signal (what is being manipulated).
        obj_states: dict[str, ObjectState] = {}
        if objects is not None:
            for o in objects.frames.get(i, []):
                obj_states[str(o.track_id)] = ObjectState(
                    mask=MaskRef(rle=o.mask_rle), pose=None
                )
            if obj_states:
                provenance["objects"] = Provenance.VISION_PRIMARY

        frames.append(
            CanonicalFrame(
                t=float(i / fps),
                rig=rig,
                episode_id=eid,
                frame_idx=i,
                images={cam: ImageRef(uri=video, frame_index=i)},
                camera_pose=cam_pose,
                hands=hand_states,
                objects=obj_states,
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
