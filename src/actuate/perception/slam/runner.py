"""`perception.slam.run(store, method="auto")` -- Master Spec §L1 ego-motion.

Selects a backend from the sensor streams the capture actually has, rather than from what
the spec would prefer:

    RGB + IMU (gyro)  -> "vio"        gyro rotation + visual translation   [BUILT]
    RGB only          -> "vio"        vision-only rotation, no IMU         [BUILT, weaker]
    RGB + IMU, want a global map, loop closure
                      -> "orb_slam3"  [NOT BUILT -- see orb_slam3.py]

The backend is swappable by design. What matters downstream is `camera_pose` per frame;
how it was obtained is recorded in provenance, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from actuate.config import Provenance
from actuate.perception.sampling import sampled_indices
from actuate.perception.slam.vio import (
    SlamResult,
    approximate_intrinsics,
    gyro_relative_rotations,
    vision_relative_rotation,
)
from actuate.schema import SE3

_TRACK_WIDTH = 640


class SlamError(RuntimeError):
    pass


def _rotation_to_se3(R: Rotation, t: np.ndarray) -> SE3:
    q = R.as_quat()  # scipy: (x, y, z, w)
    return SE3(
        position_m=(float(t[0]), float(t[1]), float(t[2])),
        quaternion_wxyz=(float(q[3]), float(q[0]), float(q[1]), float(q[2])),
    )


def _load_gyro(session_dir: Path, n_frames: int) -> np.ndarray | None:
    """Per-video-frame gyro. v1's sync stage already resampled it onto the frame axis."""
    h5 = session_dir / "session.h5"
    if not h5.exists():
        return None
    import h5py

    with h5py.File(h5, "r") as f:
        if "imu/gyro" not in f:
            return None
        g = np.asarray(f["imu/gyro"][:], dtype=np.float64)
    if len(g) != n_frames:
        return None
    return g


def run(
    session_dir: Path,
    method: str = "auto",
    max_frames: int | None = None,
    stride: int = 1,
    frames: list[int] | None = None,
) -> SlamResult:
    """Estimate per-frame camera pose. Reads only; returns the result for the caller to store.

    ``max_frames``/``frames`` select the video frames used for the visual cross-check. Camera
    poses remain indexed on the full original frame axis so downstream perception sampled at
    frame 1799 can read pose 1799 instead of silently losing it.

    ``stride > 1`` further subsamples the visual cross-check (the gyro channel is always
    full-rate). The cross-check is a verification, not a product, so it need not decode every
    frame.
    """
    session_dir = Path(session_dir)
    meta = json.loads((session_dir / "session_meta.json").read_text())
    n_frames = int(meta["frame_count"])
    fps = float(meta.get("fps_nominal", 30.0))
    dt = 1.0 / fps
    W, H = int(meta["video_width"]), int(meta["video_height"])

    selected = frames if frames is not None else sampled_indices(n_frames, max_frames)
    selected = sorted({int(i) for i in selected if 0 <= int(i) < n_frames})
    selected = selected[::stride]

    gyro = _load_gyro(session_dir, int(meta["frame_count"]))
    has_imu = gyro is not None

    if method == "auto":
        method = "vio"
    if method == "orb_slam3":
        from actuate.perception.slam.orb_slam3 import run as orb_run

        return orb_run(session_dir)
    if method != "vio":
        raise SlamError(f"unknown slam method {method!r}")

    notes: dict[str, str] = {}
    provenance: dict[str, Provenance] = {}

    # --- rotation ---------------------------------------------------------------------
    if has_imu:
        rel_rot = gyro_relative_rotations(gyro[:n_frames], dt)
        rot_mag = np.array([r.magnitude() for r in rel_rot])
        provenance["camera_pose.rotation"] = Provenance.MEASURED_HUMAN
        notes["rotation"] = (
            "From the IMU gyroscope, integrated over each 33 ms frame interval. This is a "
            "direct sensor measurement, not an inference. Drift accumulates with time and "
            "33 ms is far too short for it to matter."
        )
    else:
        rel_rot = [Rotation.identity()] * n_frames
        rot_mag = np.zeros(n_frames)
        provenance["camera_pose.rotation"] = Provenance.VISION_PRIMARY
        notes["rotation"] = "No IMU on this capture; rotation comes from vision alone."

    # --- the cross-check + translation direction, from the camera -----------------------
    from actuate.ingest.run import _session_video

    video = _session_video(session_dir)

    K = approximate_intrinsics(_TRACK_WIDTH, int(H * _TRACK_WIDTH / W))
    notes["intrinsics"] = (
        "APPROXIMATED from an assumed 82 deg HFOV -- this capture carries no calibration "
        "for its device. UniDepthV2 (Part C) estimates intrinsics from the image itself, "
        "which is the real fix. Rotation is far less sensitive to intrinsics error than "
        "translation, which is another reason rotation is the trustworthy channel here."
    )
    provenance["camera_pose.intrinsics"] = Provenance.APPROXIMATED

    # The cross-check compares consecutive KEPT frames. Sampling/stride can span several
    # frame intervals, so the gyro side must be accumulated over the same span or the two
    # sensors would be answering different questions.
    rot_vision = np.full(n_frames, np.nan)
    rot_gyro_span = np.full(n_frames, np.nan)
    rv_gyro: dict[int, np.ndarray] = {}
    rv_vision: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(str(video))
    scale = _TRACK_WIDTH / W
    prev, prev_i = None, None
    contiguous = selected == list(range(len(selected)))
    for i in selected:
        if not contiguous:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (_TRACK_WIDTH, int(H * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            R_v, _t_v, _n = vision_relative_rotation(prev, gray, K)
            if R_v is not None:
                rot_vision[prev_i] = R_v.magnitude()
                # Gyro over the SAME span: compose the per-frame relative rotations.
                span = Rotation.identity()
                for j in range(prev_i, i):
                    span = span * rel_rot[j]
                rot_gyro_span[prev_i] = span.magnitude()
                # Keep the VECTORS. The gate compares rotations, not magnitudes.
                rv_gyro[prev_i] = span.as_rotvec()
                rv_vision[prev_i] = R_v.as_rotvec()
        prev, prev_i = gray, i
    cap.release()

    # --- translation --------------------------------------------------------------------
    # Direction from vision; scale is NOT metric -- there is no dense depth yet. We emit a
    # zero translation rather than a wrong one, and say so. Rotation is what dominates the
    # contamination (median 13.4 mm/frame of the 35.8 mm action), and rotation IS metric.
    translations = np.zeros((n_frames, 3))
    provenance["camera_pose.translation"] = Provenance.VISION_FALLBACK
    notes["translation"] = (
        "NOT COMPENSATED. The essential matrix gives translation only up to scale, and "
        "metric scale needs dense depth (Part C, UniDepthV2) or IMU double-integration "
        "(which drifts badly over 95 s). Emitting a wrongly-scaled translation would be "
        "worse than emitting none: it would inject a systematic error into every action. "
        "So camera translation is ZERO here, and the residual contamination from head "
        "TRANSLATION (small at a workbench, unlike head rotation) remains. Part C upgrades "
        "this to metric PnP."
    )

    # --- solve the IMU -> CAMERA extrinsic ------------------------------------------------
    #
    # The gyroscope measures rotation in the IMU's frame. Reprojection needs it in the
    # CAMERA's frame. The two are bolted to the same head but their axes are NOT aligned,
    # and we have no extrinsic calibration for this rig.
    #
    # We can solve it, because we have the same rotation measured by two sensors. For a pure
    # rotation, a rotation VECTOR transforms as  v_cam = R_ext @ v_imu, so R_ext is the
    # orthogonal Procrustes (Kabsch) solution aligning the paired rotation vectors. Weight by
    # magnitude: large head turns have far better signal-to-noise than sub-degree ones.
    R_ext = np.eye(3)
    if has_imu and len(rv_gyro) >= 20:
        idx = sorted(set(rv_gyro) & set(rv_vision))
        G = np.array([rv_gyro[i] for i in idx])
        V = np.array([rv_vision[i] for i in idx])
        w = np.linalg.norm(G, axis=1)
        keep = w > np.radians(0.5)  # a near-zero rotation has no meaningful axis
        if keep.sum() >= 20:
            Gk, Vk, wk = G[keep], V[keep], w[keep]
            H = (Gk * wk[:, None]).T @ Vk
            U, _S, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R_ext = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

            ext = Rotation.from_matrix(R_ext)
            rel_rot = [ext * r * ext.inv() for r in rel_rot]   # conjugate into the camera frame
            rv_gyro = {i: (R_ext @ v) for i, v in rv_gyro.items()}
            rot_mag = np.array([r.magnitude() for r in rel_rot])

            notes["imu_to_camera_extrinsic"] = (
                "SOLVED, not assumed. The IMU and camera axes are not aligned and this rig "
                f"has no extrinsic calibration. Recovered from {int(keep.sum())} paired "
                "rotations (gyro vs vision) by weighted Kabsch: "
                f"{np.round(ext.as_euler('xyz', degrees=True), 1).tolist()} deg (xyz euler). "
                "Applying gyro rotation WITHOUT this rotates about the wrong axes and injects "
                "error instead of removing it -- which is what happened before the gate was "
                "fixed to compare rotations rather than magnitudes."
            )

    # --- integrate to world-frame poses ---------------------------------------------------
    poses: list[SE3] = []
    R_acc = Rotation.identity()
    t_acc = np.zeros(3)
    for i in range(n_frames):
        poses.append(_rotation_to_se3(R_acc, t_acc))
        R_acc = R_acc * rel_rot[i]
        t_acc = t_acc + R_acc.apply(translations[i])

    return SlamResult(
        poses=poses,
        rotation_rad=rot_mag,
        rotation_rad_gyro_span=rot_gyro_span,
        rotation_rad_vision=rot_vision,
        translation_is_metric=False,
        _rotvec_gyro=rv_gyro,
        _rotvec_vision=rv_vision,
        imu_to_camera=R_ext,
        provenance=provenance,
        notes=notes,
    )
