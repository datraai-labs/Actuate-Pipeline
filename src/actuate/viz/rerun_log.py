"""Log a pipeline episode to Rerun -- the "see your pipeline" capability. Master Spec §F.

`log_episode(episode, hands=, depth=, objects=, fusion=, slam=, video_path=)` writes every
modality that is present to a Rerun recording, all on ONE `frame` timeline so the viewer scrubs
video, depth, hand, objects, and state together.

The design principle: log what actually exists, honestly. A bare-hand head-mounted capture has
no 6-DoF object pose and a wrist trajectory that Part C showed is unreliable -- those either do
not appear or appear clearly labelled, rather than being faked to fill the view.

### Placing the hand at metric depth (so it is not floating)

WiLoR's keypoints are root-relative. To put the hand in the same 3D frame as the depth point
cloud, its ROOT depth is solved from the depth map at the hand keypoints (Part C's
`solve_root_depth` + `smooth_root_depth`) and each keypoint is back-projected. So the hand
sits inside the scene cloud at ~0.6 m, not at an arbitrary origin. This is the one place the
otherwise-unreliable metric depth is good enough: a rough but correct-order placement for
viewing, explicitly not a trained action.

The camera uses OpenCV RDF coordinates (+x right, +y down, +z forward), so depth is +z.
"""

from __future__ import annotations

import numpy as np

from actuate.config import Side
from actuate.viz.conventions import (
    FINGERTIPS,
    HAND_COLORS,
    STATE_COLORS,
    contact_color,
    skeleton_strips,
)

_CAM = "world/camera"
_IMG = "world/camera/image"


def _rr():
    import rerun as rr

    return rr


def _set_frame(i: int) -> None:
    _rr().set_time("frame", sequence=i)


def log_static(annotation_labels: dict[int, str] | None = None) -> None:
    """Log time-independent scaffolding: the world coordinate convention."""
    rr = _rr()
    rr.log("world", rr.ViewCoordinates.RDF, static=True)


def log_video_frame(i: int, bgr: np.ndarray) -> None:
    rr = _rr()
    _set_frame(i)
    rr.log(_IMG, rr.Image(bgr[:, :, ::-1]))  # BGR -> RGB


def log_depth_frame(i: int, depth_m: np.ndarray, K: np.ndarray, rgb: np.ndarray | None = None,
                    stride: int = 8) -> None:
    """Depth as a 2D DepthImage AND a back-projected 3D point cloud (strided for the viewer)."""
    rr = _rr()
    _set_frame(i)
    rr.log(f"{_IMG}/depth", rr.DepthImage(depth_m.astype(np.float32), meter=1.0))

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    xs, ys = xs.ravel(), ys.ravel()
    z = depth_m[ys, xs]
    ok = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
    cols = None
    if rgb is not None:
        cols = rgb[ys, xs][:, ::-1]  # BGR->RGB
    rr.log(f"{_CAM}/depth_cloud", rr.Points3D(pts, colors=cols, radii=0.002))


def log_camera_pose(i: int, pose, K: np.ndarray, wh: tuple[int, int]) -> None:
    """Camera Transform3D (world<-camera) + pinhole, so the image hangs in the 3D scene."""
    rr = _rr()
    _set_frame(i)
    if pose is not None:
        t = np.asarray(pose.position_m, dtype=np.float64)
        q = np.asarray(pose.quaternion_wxyz, dtype=np.float64)  # wxyz
        rr.log(_CAM, rr.Transform3D(translation=t, quaternion=[q[1], q[2], q[3], q[0]]))
    w, h = wh
    rr.log(_IMG, rr.Pinhole(image_from_camera=K, width=w, height=h))


def log_trajectory(poses) -> None:
    """The full SLAM camera path as one 3D polyline (static -- the whole route at once)."""
    rr = _rr()
    pts = [list(p.position_m) for p in poses if p is not None]
    if len(pts) >= 2:
        rr.log("world/trajectory", rr.LineStrips3D([pts], colors=[(200, 200, 0)]), static=True)


def _hand_abs_keypoints(hand, root_cam: np.ndarray) -> np.ndarray:
    """Absolute camera-frame keypoints: root position + root-relative WiLoR keypoints."""
    kp = np.asarray(hand.keypoints_3d, dtype=np.float64)
    return root_cam + (kp - kp[0])


def log_hand(i: int, hand, root_cam: np.ndarray, contact: dict | None = None) -> None:
    """3D hand: keypoints (fingertips coloured by contact confidence) + skeleton."""
    rr = _rr()
    _set_frame(i)
    side = hand.side
    base = f"world/hand/{side.value.lower()}"
    abs_kp = _hand_abs_keypoints(hand, root_cam)

    colors = [HAND_COLORS[side]] * 21
    if contact:
        for fname, tip in FINGERTIPS.items():
            from actuate.config import Finger

            f = Finger[fname.upper()]
            cp = contact.get(f)
            if cp is not None:
                colors[tip] = contact_color(cp.confidence)
    rr.log(base, rr.Points3D(abs_kp, colors=colors, radii=0.004))
    rr.log(f"{base}/skeleton", rr.LineStrips3D(skeleton_strips(abs_kp),
                                               colors=[HAND_COLORS[side]]))


def log_wrist_action(i: int, root_cam: np.ndarray, delta: np.ndarray) -> None:
    """The wrist action delta as a 3D arrow from the current wrist position."""
    rr = _rr()
    _set_frame(i)
    rr.log("world/hand/action", rr.Arrows3D(origins=[root_cam], vectors=[delta],
                                            colors=[(255, 0, 255)]))


def log_objects(i: int, object_frames, hw: tuple[int, int]) -> None:
    """2D boxes + a merged segmentation image (one class id per track)."""
    rr = _rr()
    from actuate.perception.objects.rle import decode_rle

    _set_frame(i)
    if not object_frames:
        return
    mins, sizes, labels, class_ids = [], [], [], []
    seg = np.zeros(hw, dtype=np.uint16)
    for o in object_frames:
        x0, y0, x1, y1 = o.bbox
        mins.append([x0, y0])
        sizes.append([x1 - x0, y1 - y0])
        labels.append(f"{o.label}#{o.track_id}")
        class_ids.append(o.track_id)
        m = decode_rle(o.mask_rle)
        if m.shape == hw:
            seg[m] = o.track_id
    rr.log(f"{_IMG}/objects", rr.Boxes2D(mins=mins, sizes=sizes, labels=labels,
                                         class_ids=class_ids))
    rr.log(f"{_IMG}/masks", rr.SegmentationImage(seg))


def log_state(i: int, state) -> None:
    """Interaction state as a coloured 3D point + a text log line, both on the frame timeline."""
    rr = _rr()
    _set_frame(i)
    col = STATE_COLORS.get(state, (128, 128, 128))
    rr.log("world/state", rr.Points3D([[0, 0, 0]], colors=[col], radii=0.02,
                                      labels=[state.value]))
    rr.log("state", rr.TextLog(state.value))


def log_contact(i: int, side: Side, contact: dict) -> None:
    """Per-finger contact confidence as scalar time series (a bar/line per finger)."""
    rr = _rr()
    _set_frame(i)
    for finger, cp in contact.items():
        rr.log(f"contact/{side.value.lower()}/{finger.value}", rr.Scalars(cp.confidence))


def log_episode(
    episode=None,
    *,
    hands=None,
    depth=None,
    objects=None,
    fusion=None,
    slam=None,
    video_frames: list | None = None,
    intrinsics: np.ndarray | None = None,
    max_frames: int | None = None,
) -> dict[str, int]:
    """Log every present modality to the ACTIVE Rerun recording, on one `frame` timeline.

    Everything is optional -- pass what the pipeline produced. `episode` (a CanonicalEpisode)
    supplies wrist-action arrows and metadata; the perception result objects supply the rest.
    Returns a count of how many frames each modality logged, so a caller (or the gate) can
    verify the recording is non-empty without opening a viewer.

    The hand is placed at metric depth via Part C's root-depth solver when both `hands` and
    `depth` are present, so it sits in the depth cloud rather than floating.
    """
    from actuate.perception.depth import backproject, smooth_root_depth, solve_root_depth

    counts = {k: 0 for k in
              ("video", "depth", "camera", "hand", "objects", "state", "contact", "action")}
    log_static()

    K = intrinsics if intrinsics is not None else (depth.intrinsics if depth is not None else None)

    # frame set = union of what the sources cover, bounded by max_frames
    ids: set[int] = set()
    if video_frames is not None:
        ids |= set(range(len(video_frames)))
    for src in (hands, depth, objects, fusion):
        if src is not None and hasattr(src, "frames"):
            ids |= set(src.frames)
    frame_ids = sorted(ids)
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    wh = None
    if video_frames:
        h0, w0 = video_frames[0].shape[:2]
        wh = (w0, h0)

    # --- pre-solve per-frame root depth for hand placement (Part C) ---------------------
    root_z: dict[int, float] = {}
    if hands is not None and depth is not None and K is not None:
        raw = np.full(max(frame_ids) + 1, np.nan)
        for i in frame_ids:
            hs = hands.frames.get(i, [])
            df = depth.frames.get(i)
            if not hs or df is None:
                continue
            h = next((x for x in hs if x.side == Side.RIGHT), hs[0])
            raw[i] = solve_root_depth(df.depth_m, df.confidence, h.keypoints_2d, h.keypoints_3d)
        sm = smooth_root_depth(raw)
        root_z = {i: float(sm[i]) for i in frame_ids if np.isfinite(sm[i])}

    # camera trajectory (static, whole path)
    if slam is not None and getattr(slam, "poses", None):
        log_trajectory(slam.poses)

    prev_root: dict[Side, np.ndarray] = {}
    for i in frame_ids:
        if video_frames is not None and i < len(video_frames):
            log_video_frame(i, video_frames[i])
            counts["video"] += 1
            wh = (video_frames[i].shape[1], video_frames[i].shape[0])

        if (slam is not None and getattr(slam, "poses", None)
                and i < len(slam.poses) and K is not None and wh):
            log_camera_pose(i, slam.poses[i], K, wh)
            counts["camera"] += 1

        df = depth.frames.get(i) if depth is not None else None
        if df is not None and K is not None:
            rgb = video_frames[i] if (video_frames and i < len(video_frames)) else None
            log_depth_frame(i, df.depth_m, K, rgb=rgb)
            counts["depth"] += 1

        if objects is not None and wh:
            of = objects.frames.get(i)
            if of:
                log_objects(i, of, (wh[1], wh[0]))
                counts["objects"] += 1

        ff = fusion.frames.get(i) if fusion is not None else None
        if ff is not None:
            log_state(i, ff.interaction_state)
            counts["state"] += 1
            for side, fingers in ff.contact.items():
                log_contact(i, side, fingers)
                counts["contact"] += 1

        if hands is not None:
            for h in hands.frames.get(i, []):
                # place the hand: root at metric depth if solved, else at its virtual root
                if i in root_z and K is not None:
                    root_cam = backproject(h.keypoints_2d[0], root_z[i], K)
                else:
                    root_cam = np.asarray(h.keypoints_3d[0], dtype=np.float64)
                contact = ff.contact.get(h.side) if ff is not None else None
                log_hand(i, h, root_cam, contact)
                counts["hand"] += 1

                # wrist action = displacement of this hand's root vs the previous frame
                if h.side in prev_root:
                    log_wrist_action(i, root_cam, root_cam - prev_root[h.side])
                    counts["action"] += 1
                prev_root[h.side] = root_cam

    return counts
