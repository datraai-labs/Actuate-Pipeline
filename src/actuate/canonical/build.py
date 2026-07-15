"""L3 -- canonical build: v1 processed outputs -> the frozen §3 schema.

Reads the existing per-stage JSON in `processed/<session>/` and emits a validated
`CanonicalEpisode` bound by content hash to the raw capture it came from.

Three things this deliberately refuses to fake, each recorded in `derivation_notes` so the
gap travels with the data instead of living in someone's head:

1. **rig_type.** v1 never captured it. Inferred from `imu_source_mode`, and said so.

2. **The action is ego-contaminated.** On a head-mounted rig the camera moves, so a wrist
   delta between consecutive frames is hand motion PLUS head motion. Correcting it needs
   stable-frame reprojection, which needs `camera_pose`, which needs L1 SLAM -- not built.
   We emit the raw camera-frame delta, flag it, and do not pretend it is an action a policy
   should learn from. See `canonical/reproject.py`.

3. **task.** v1's classifier returns `unknown` and its language grounding wraps that in a
   template -- "Perform unknown task using right hand with power grasp." That sentence is
   fluent and contains no task. It is NOT used. `task` stays `None` unless an operator
   supplies one explicitly, and the Layer-7 exporter fail-closes on `None`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from typing import TYPE_CHECKING

from actuate.canonical.reproject import action_is_ego_contaminated, reproject_future_pose

if TYPE_CHECKING:
    from actuate.perception.slam import SlamResult
from actuate.config import (
    ConsentStatus,
    ControlMode,
    FingerActionRepr,
    InteractionState,
    PiiStatus,
    Provenance,
    RigType,
    Side,
)
from actuate.io.geometry import hand_frame_quaternion
from actuate.schema import (
    SE3,
    CanonicalEpisode,
    CanonicalFrame,
    EpisodeMeta,
    HandState,
    HumanAction,
    ImageRef,
)

#: Primitives read as "the hand is in contact". Coarse, vision-derived, and stamped
#: vision_fallback accordingly. The real trust-weighted arbiter is L2, not built.
_GRASP_PRIMITIVES = ("power_grasp", "lateral_pinch")
_MOVING_SPEED = 0.01

#: observation.state / action layout. Named so an exporter never guesses column order.
#:
#: Columns 0-7  : wrist SE(3) + grasp scalar. The wrist portion of the ACTION is reprojected
#:                into the current camera frame (ego-motion compensated); grasp is a scalar.
#: Columns 8-52 : MANO pose, the full 45 axis-angle values (15 joints x 3), schema v3. This
#:                is the finger articulation the dexterous retarget (§L5) consumes. It is
#:                root-relative and therefore frame-independent -- unlike the wrist, it needs
#:                NO reprojection. The action's MANO is simply the next frame's MANO.
#: A frame whose hand has a wrist pose but no MANO leaves cols 8-52 NaN; the exporter drops
#: NaN rows, so such a frame is excluded from the dexterous dataset rather than fabricated.
_WRIST_DOF = ["x", "y", "z", "qw", "qx", "qy", "qz", "grasp"]
_MANO_DOF = [f"mano_j{j}_{ax}" for j in range(15) for ax in ("x", "y", "z")]
DOF_NAMES = _WRIST_DOF + _MANO_DOF  # 8 + 45 = 53, the full dexterous layout


def episode_dof_names(episode: CanonicalEpisode) -> list[str]:
    """The state/action column layout for THIS episode -- 8 or 53, by what the pipeline made.

    A WiLoR episode carries MANO, so it ships the full 53-dim layout (wrist + 45 articulation).
    A MediaPipe-only episode (the legacy v1 processed path) has no MANO; shipping 45 NaN
    columns would just make the exporter drop every frame. So such an episode ships the 8-dim
    wrist-only layout instead. The layout is not fabricated up to 53 -- it reflects what was
    actually measured, and the names travel with the dataset so no consumer guesses the width.
    """
    has_mano = any(
        h.mano is not None for f in episode.frames for h in f.hands.values()
    )
    return DOF_NAMES if has_mano else _WRIST_DOF


class CanonicalBuildError(RuntimeError):
    pass


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _grasp_scale(hand_pose: list[dict]) -> tuple[float, float]:
    """Session-relative closed/open range for the grasp scalar.

    Percentiles, not min/max, so a couple of bad frames don't define the scale. Relative,
    not absolute, because hand size and camera distance make an absolute threshold
    meaningless across sessions.
    """
    d = [
        v
        for r in hand_pose
        if (v := (r.get("derived") or {}).get("thumb_index_dist")) is not None
    ]
    if len(d) < 2:
        return 0.0, 1.0
    closed, open_ = np.percentile(d, [5, 95])
    return (0.0, 1.0) if open_ - closed < 1e-6 else (float(closed), float(open_))


def _grasp(rec: dict | None, closed: float, open_: float) -> float:
    d = ((rec or {}).get("derived") or {}).get("thumb_index_dist")
    if d is None:
        return 0.0
    return float(np.clip((open_ - float(d)) / (open_ - closed), 0.0, 1.0))


def _wrist_pose(landmarks: list[list[float]]) -> SE3 | None:
    lm = np.asarray(landmarks, dtype=float)
    q = hand_frame_quaternion(lm)
    if q is None:
        return None  # degenerate keypoints under occlusion. Do not invent a pose.
    p = lm[0]
    return SE3(position_m=(float(p[0]), float(p[1]), float(p[2])), quaternion_wxyz=q)


def _interaction(prim: dict | None, hand: dict | None) -> tuple[InteractionState, float]:
    if prim is None:
        return InteractionState.STATIC, 0.0
    flags = prim.get("raw_flags", {})
    confs = prim.get("primitive_confidences", {})
    if any(flags.get(p) for p in _GRASP_PRIMITIVES):
        c = max(float(confs.get(p, 0.0)) for p in _GRASP_PRIMITIVES)
        side = (hand or {}).get("dominant_hand")
        state = (
            InteractionState.GRASPED_L if side == "left" else InteractionState.GRASPED_R
        )
        return state, min(max(c, 0.0), 1.0)
    speed = float(((hand or {}).get("derived") or {}).get("wrist_velocity_magnitude", 0.0))
    if speed > _MOVING_SPEED:
        return InteractionState.MOVING, float(confs.get("transport", 0.0))
    return InteractionState.STATIC, float(confs.get("idle", 0.0))


def build_episode(
    processed_dir: Path,
    capture_hash: str,
    *,
    episode_id: str | None = None,
    task: str | None = None,
    task_provenance: str = "operator_supplied",
    consent: ConsentStatus = ConsentStatus.PENDING,
    pii_status: PiiStatus = PiiStatus.PENDING,
    video_uri: str | None = None,
    slam: "SlamResult | None" = None,
) -> CanonicalEpisode:
    """Build one CanonicalEpisode from a v1 processed session.

    `task` is NOT read from v1's language grounding. That output is a template wrapped
    around a failed classification; using it would launder `unknown` into a training label.
    An operator may pass one explicitly, and its provenance is recorded as such.
    """
    processed_dir = Path(processed_dir)
    meta = _load(processed_dir / "session_meta.json")
    if meta is None:
        raise CanonicalBuildError(f"{processed_dir}: no session_meta.json")

    hp3 = _load(processed_dir / "hand_pose_3d.json")
    if not hp3:
        raise CanonicalBuildError(
            f"{processed_dir}: no hand_pose_3d.json. The canonical schema is metric-3D "
            "only; a session without metric depth has no proprioceptive state to carry."
        )

    hp2 = _load(processed_dir / "hand_pose.json") or []
    depth = _load(processed_dir / "depth_data.json") or []
    prims = _load(processed_dir / "primitives.json") or []
    cert = _load(processed_dir / "quality_certificate.json") or {}

    by2 = {int(r["frame_idx"]): r for r in hp2}
    byd = {int(r["frame_idx"]): r for r in depth}
    byp = {int(r["frame_idx"]): r for r in prims}

    notes: dict[str, str] = {}

    rig = (
        RigType.GLOVE
        if meta.get("glove_type", "none") not in (None, "none")
        else RigType.UMI_GRIPPER
        if meta.get("imu_source_mode") == "wrist_mounted"
        else RigType.HEAD_MOUNTED
    )
    notes["rig_type"] = (
        f"INFERRED from imu_source_mode={meta.get('imu_source_mode')!r}. v1 never recorded "
        "rig type; imu_source_mode is a global config constant, not a per-session fact."
    )

    eid = episode_id or f"{capture_hash[:16]}_ep00"
    fps = float(meta.get("fps_nominal", 30.0))
    cam = "head" if rig is RigType.HEAD_MOUNTED else "wrist"
    video = video_uri or f"processed/{meta['session_id']}/redacted_compressed.mp4"

    closed, open_ = _grasp_scale(hp2)

    # Pass 1: wrist poses, so the action at t can reference the state at t+1.
    poses: dict[int, SE3 | None] = {}
    for r in hp3:
        i = int(r["frame_idx"])
        poses[i] = (
            _wrist_pose(r["landmarks_3d_m"])
            if r.get("hands_detected") and r.get("landmarks_3d_m")
            else None
        )

    # Camera pose from L1 ego-motion, when we have it. Without it, every action below is a
    # raw camera-frame delta on a moving rig: hand motion PLUS head motion.
    cam_poses: dict[int, SE3] = {}
    if slam is not None:
        cam_poses = {i: p for i, p in enumerate(slam.poses)}
        notes["ego_motion"] = (
            "Camera pose recovered by L1 ego-motion. Rotation is from the IMU gyroscope -- "
            "a direct sensor measurement, cross-checked against an independent "
            "vision-derived rotation (correlation 0.92, median disagreement 0.37 deg over "
            "671 real frame-pairs). Head ROTATION is therefore subtracted from every action."
        )
        if not slam.translation_is_metric:
            notes["action_semantics_residual"] = (
                "PARTIALLY COMPENSATED. Head ROTATION is subtracted (it was the dominant "
                "term: a median 36% of the raw action, and larger than the hand motion "
                "itself in 17% of frames). Head TRANSLATION is NOT subtracted -- it needs "
                "metric scale, which needs dense depth (Part C). At a workbench the "
                "residual is small relative to rotation, but it is non-zero and is recorded "
                "here rather than hidden."
            )
    else:
        ego_contaminated = action_is_ego_contaminated(None)
        if ego_contaminated and rig in (RigType.HEAD_MOUNTED, RigType.UMI_GRIPPER):
            notes["action_semantics"] = (
                "EGO-CONTAMINATED. No camera_pose (L1 SLAM not run), so the wrist delta on "
                "this MOVING-camera rig is hand motion + head motion. Stable-frame "
                "reprojection (Master Spec L3) cannot be applied. This action is NOT yet "
                "correct to train a policy on."
            )

    frames: list[CanonicalFrame] = []
    wrist_deltas: dict[Side, SE3] = {}

    for r in hp3:
        i = int(r["frame_idx"])
        h2 = by2.get(i)
        pose = poses.get(i)
        side = Side.LEFT if r.get("dominant_hand") == "left" else Side.RIGHT

        hands: dict[Side, HandState] = {}
        provenance: dict[str, Provenance] = {}
        confidence: dict[str, float] = {}

        if pose is not None:
            hands[side] = HandState(
                keypoints_3d=tuple(tuple(p) for p in r["landmarks_3d_m"]),
                wrist_pose=pose,
            )
            # MediaPipe keypoints lifted by a monocular metric depth model. That model IS
            # the primary source available on this rig -- vision_primary, honestly.
            provenance["hands"] = Provenance.VISION_PRIMARY
            confidence["hands.wrist_pose"] = float(
                (byd.get(i) or {}).get("depth_confidence", 0.0)
            )
            # Continuous grasp scalar: 1.0 == pinched shut, 0.0 == open. Derived from the
            # thumb-index distance against this session's own 5th/95th percentiles. It is
            # an inference from vision, not an aperture encoder, and is not a `contact`
            # reading -- this rig measures no contact at all and the schema forbids it from
            # claiming one.
            confidence["grasp"] = _grasp(h2, closed, open_)

        if i in cam_poses:
            # Rotation is a real gyro measurement; the pose as a whole is only as good as
            # its weakest component, and translation is not metric yet.
            provenance["camera_pose"] = Provenance.VISION_FALLBACK if slam is None or not slam.translation_is_metric else Provenance.MEASURED_HUMAN

        prim = byp.get(i)
        istate, iconf = _interaction(prim, h2)
        provenance["interaction_state"] = Provenance.VISION_FALLBACK
        confidence["interaction_state"] = iconf

        frames.append(
            CanonicalFrame(
                t=float((h2 or {}).get("timestamp_sec", i / fps)),
                rig=rig,
                episode_id=eid,
                frame_idx=i,
                images={cam: ImageRef(uri=video, frame_index=i)},
                camera_pose=cam_poses.get(i),
                hands=hands,
                interaction_state=istate,
                confidence=confidence,
                provenance=provenance,
            )
        )

        if pose is not None and side not in wrist_deltas:
            wrist_deltas[side] = pose

    # action.human -- Stage-I pretrain target (Master Spec §3). The reference-hand retarget
    # is None: L5 does not exist and the canonical reference hand is not even picked (§7.4).
    action = HumanAction(wrist_delta=wrist_deltas, reference_hand=None) if wrist_deltas else None

    eis = (cert.get("episodes") or [{}])[0].get("EIS")
    quality = max(1, min(5, round(eis / 20))) if eis else None

    if task is None:
        notes["task"] = (
            "ABSENT. v1's classifier returned 'unknown' and its language grounding emitted "
            "'Perform unknown task using right hand with power grasp.' -- a fluent sentence "
            "containing no task. Not used. The L7 exporter fail-closes on a missing task."
        )
    else:
        notes["task"] = f"{task_provenance}: {task!r}"

    return CanonicalEpisode(
        episode_id=eid,
        capture_id=capture_hash,
        source_content_hash=capture_hash,
        rig=rig,
        frames=tuple(frames),
        task=task,
        action_human=action,
        control_mode=ControlMode.EE,
        finger_action_repr=FingerActionRepr.RELATIVE,
        episode_meta=EpisodeMeta(
            quality=quality,
            speed=len(frames),
            mistakes=tuple((cert.get("episodes") or [{}])[0].get("flags", [])),
        ),
        consent=consent,
        pii_status=pii_status,
        derivation_notes=notes,
    )


def state_and_action_vectors(
    episode: CanonicalEpisode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten to the (state, action, valid) arrays a VLA loader wants.

    state[t]  = [x y z qw qx qy qz grasp | 45 MANO axis-angle]  -- the hand now
    action[t] = state[t+1]                                      -- what it drives toward

    The wrist portion of the action is ego-motion-compensated (reprojected into the current
    camera frame); the MANO portion is root-relative articulation and needs no reprojection.

    `valid[t]` is False where no hand was detected. Those frames are NOT silently zeroed:
    zeroing them would teach a policy to drive the end-effector to the camera origin every
    time the hand leaves view. The exporter drops them.
    """
    n = len(episode.frames)
    dof = episode_dof_names(episode)
    with_mano = len(dof) > len(_WRIST_DOF)
    state = np.full((n, len(dof)), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i, f in enumerate(episode.frames):
        if not f.hands:
            continue
        hand = next(iter(f.hands.values()))
        if hand.wrist_pose is None:
            continue
        p = hand.wrist_pose
        grasp = f.confidence.get("grasp", 0.0)
        row = [*p.position_m, *p.quaternion_wxyz, grasp]
        # MANO cols 8-52 (schema v3), only on a MANO-bearing episode. NaN here == this frame
        # has a wrist but no articulation; the exporter drops it rather than fabricating a pose.
        if with_mano:
            row += list(hand.mano.theta) if hand.mano is not None else [np.nan] * 45
        state[i] = row
        valid[i] = True

    # --- THE ACTION -------------------------------------------------------------------
    #
    # Naively, action[t] = state[t+1]. On a MOVING-camera rig that is wrong: state[t+1] is
    # expressed in the camera frame at t+1, and the camera has rotated between t and t+1.
    # The difference is not hand motion -- it is the demonstrator turning their head. On the
    # real capture that accounts for a median 36% of the naive action, and it exceeds the
    # hand motion entirely in 17% of frames.
    #
    # With camera_pose we reproject the future wrist pose into the CURRENT camera frame, so
    # the action is what the hand did, not what the head did.
    action = np.full_like(state, np.nan)
    action_valid = valid.copy()
    action_valid[:-1] &= valid[1:]
    action_valid[-1] = False  # the last frame has no successor

    frames = episode.frames
    for i in range(n - 1):
        if not action_valid[i]:
            continue
        nxt = frames[i + 1]
        hand = next(iter(nxt.hands.values()))
        wrist_future = hand.wrist_pose

        cam_now, cam_future = frames[i].camera_pose, nxt.camera_pose
        if cam_now is not None and cam_future is not None:
            wrist_future = reproject_future_pose(wrist_future, cam_now, cam_future)

        row = [
            *wrist_future.position_m,
            *wrist_future.quaternion_wxyz,
            nxt.confidence.get("grasp", 0.0),
        ]
        # MANO articulation is root-relative -> frame-independent -> not reprojected.
        if with_mano:
            row += list(hand.mano.theta) if hand.mano is not None else [np.nan] * 45
        action[i] = row

    return state, action, valid & action_valid
