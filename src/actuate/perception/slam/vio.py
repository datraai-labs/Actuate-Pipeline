"""Visual-inertial odometry -- recover per-frame camera pose on a moving-camera rig.

### Why this exists

Without camera pose, the "action" we emit is not an action. Measured on the real capture:

    head is rotating 94% of the time (median 18 deg/s, peak 108 deg/s)
    median wrist delta emitted as the action      : 35.8 mm/frame
    median displacement caused by HEAD ROTATION   : 13.4 mm/frame
    -> median 36% of the action is the demonstrator turning their head
    -> in 17% of frames, head motion EXCEEDS hand motion entirely

A policy trained on that learns to predict head motion as if it were hand motion.

### Why odometry and not SLAM

Stable-frame reprojection needs the RELATIVE pose between frame t and t+k, where k is at
most one action chunk (~16 frames, 0.5 s). It never needs a globally consistent map, and it
never needs loop closure. Full SLAM (ORB-SLAM3, DROID) solves a strictly harder problem --
which is also the part that is hard to build. We solve the problem we have.

### Where the numbers come from, and how much to trust each

**Rotation: from the gyroscope.** Integrated over a 33 ms frame interval. Gyro integration
drifts, but drift accumulates with time, and 33 ms is short enough that it is negligible --
this is a direct sensor measurement, not an inference. It is also the dominant contaminant.
Provenance: MEASURED_HUMAN (a real IMU on the rig).

**Rotation is CROSS-CHECKED against vision.** An essential matrix from ORB correspondences
gives an entirely independent estimate of the same rotation, from a different physical
sensor. If the two agree across thousands of real frames, the ego-motion is real. Agreement
between independent sensors is much stronger evidence than "the trajectory looks
non-degenerate", which a buggy implementation would also produce.

**Translation: estimated, and weaker.** The essential matrix gives translation only up to
scale. Metric scale needs either dense metric depth (-> PnP) or IMU double-integration
(which drifts badly). We have no dense depth yet -- Part C (UniDepthV2) delivers it -- so
translation here is scaled from IMU acceleration over the frame interval and is the least
trustworthy quantity in this module. It is stamped VISION_FALLBACK, and
`SlamResult.translation_is_metric` is False so no downstream consumer can mistake it for
sensor truth.

Head translation over 33 ms at a workbench is small compared to the rotation term, so
compensating rotation alone already removes most of the contamination. The residual is
recorded, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from actuate.config import Provenance
from actuate.schema import SE3

#: Feature tracking runs downscaled. 1080p buys nothing for ego-motion and costs 9x the
#: compute; the rotation signal lives in large-scale image motion.
_TRACK_WIDTH = 640

#: Lucas-Kanade pyramidal optical flow.
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

_MIN_TRACKS = 40  # below this, the visual estimate is not trustworthy


@dataclass
class SlamResult:
    """Per-frame camera pose in a world frame, plus an honest account of its quality."""

    #: World-frame camera pose per video frame. Index == frame_idx.
    poses: list[SE3]

    #: Per-frame relative rotation magnitude (rad), from the gyro. The measured quantity.
    rotation_rad: np.ndarray

    #: Gyro rotation accumulated over the SAME span the visual estimate covers. The two
    #: must answer the same question or the comparison is meaningless.
    rotation_rad_gyro_span: np.ndarray

    #: The independent vision-derived rotation, for the cross-check. NaN where too few
    #: features were tracked to estimate one.
    rotation_rad_vision: np.ndarray

    #: Is the translation metric, or scale-guessed? False until dense depth exists (Part C).
    translation_is_metric: bool

    #: Full rotation VECTORS (not magnitudes), per frame, both in the CAMERA frame. The gate
    #: compares these as rotations; magnitudes alone are frame-invariant and cannot fail on
    #: an axis error -- which is exactly how the old gate passed while the code was broken.
    _rotvec_gyro: dict[int, np.ndarray] = field(default_factory=dict)
    _rotvec_vision: dict[int, np.ndarray] = field(default_factory=dict)

    #: The solved IMU->camera rotation. Identity means the frames were already aligned.
    imu_to_camera: np.ndarray | None = None

    provenance: dict[str, Provenance] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.poses)

    def rotation_agreement(self) -> dict[str, float]:
        """Do the gyro and the camera agree about how the head turned?

        ### This gate was WRONG, and the way it was wrong is worth keeping in view

        It used to compare rotation MAGNITUDES: |R_gyro| vs |R_vision|. It reported r=0.92
        and a 0.37 deg median disagreement, and it was passing while the code was broken.

        Magnitude is FRAME-INVARIANT. ||R|| does not depend on which axes you express the
        rotation in. So a magnitude comparison is structurally incapable of seeing an axis
        error -- and there was one: the IMU and camera axes are misaligned by a large
        rotation (~-6/-39/-51 deg), and gyro rotation was being applied as though it were
        camera rotation.

        A gate that cannot fail on the bug in front of it is not a gate.

        ### What it does now

        Compares the rotations AS ROTATIONS: the geodesic angle of the residual

            residual = R_gyro^-1 @ R_vision      ->      angle = ||log(residual)||

        This is zero only when the two rotations are the same rotation -- same magnitude
        AND same axis. It cannot be fooled by a frame mismatch.

        `axis_angle_deg` is reported separately so the two failure modes stay
        distinguishable: a magnitude error and an axis error are different bugs.
        """
        m = np.isfinite(self.rotation_rad_vision) & np.isfinite(self.rotation_rad_gyro_span)
        if m.sum() < 10 or not self._rotvec_gyro or not self._rotvec_vision:
            return {"n": int(m.sum())}

        idx = [i for i in range(len(m)) if m[i] and i in self._rotvec_gyro]
        if len(idx) < 10:
            return {"n": len(idx)}

        g = np.array([self._rotvec_gyro[i] for i in idx])
        v = np.array([self._rotvec_vision[i] for i in idx])

        # Geodesic residual: the angle you would still have to rotate through.
        Rg = Rotation.from_rotvec(g)
        Rv = Rotation.from_rotvec(v)
        residual_deg = np.degrees((Rg.inv() * Rv).magnitude())

        # Axis disagreement, kept separate so the two bugs stay distinguishable.
        gn = np.linalg.norm(g, axis=1)
        vn = np.linalg.norm(v, axis=1)
        ok = (gn > np.radians(0.3)) & (vn > np.radians(0.3))  # tiny rotations have no axis
        axis_deg = (
            np.degrees(
                np.arccos(
                    np.clip((g[ok] * v[ok]).sum(1) / (gn[ok] * vn[ok]), -1.0, 1.0)
                )
            )
            if ok.sum() > 5
            else np.array([np.nan])
        )

        a, b = self.rotation_rad_gyro_span[m], self.rotation_rad_vision[m]
        return {
            "n": len(idx),
            # THE gate. Zero only if the two are the same rotation.
            "median_residual_deg": float(np.median(residual_deg)),
            "p90_residual_deg": float(np.percentile(residual_deg, 90)),
            # The axis error the old gate could not see.
            "median_axis_err_deg": float(np.median(axis_deg)),
            # Kept for context only -- NOT a gate. It is frame-invariant and cannot fail
            # on an axis error.
            "magnitude_correlation_NOT_A_GATE": float(np.corrcoef(a, b)[0, 1]),
        }

    def axes_are_aligned(self, tol_deg: float = 15.0) -> bool:
        """Are the IMU and camera frames aligned enough to use gyro rotation directly?

        If not, gyro rotation must be transformed by the IMU->camera extrinsic before it can
        be used for reprojection. `runner` solves that extrinsic; this reports whether it
        was needed.
        """
        ag = self.rotation_agreement()
        med = ag.get("median_axis_err_deg")
        return med is not None and np.isfinite(med) and med < tol_deg

    def is_degenerate(self) -> bool:
        """All-identity or all-NaN is what a broken SLAM produces. Refuse to ship it."""
        if not self.poses:
            return True
        pos = np.array([p.position_m for p in self.poses])
        if not np.isfinite(pos).all():
            return True
        return float(np.degrees(self.rotation_rad.sum())) < 1.0  # no rotation at all in 95s


def approximate_intrinsics(width: int, height: int, hfov_deg: float = 82.0) -> np.ndarray:
    """Pinhole K from an assumed horizontal FOV.

    THIS IS A GUESS. The capture carries no calibration for its device. UniDepthV2 (Part C)
    estimates intrinsics from the image itself, which is the real fix; until then this is
    stamped APPROXIMATED and any consumer should treat translation accordingly. Rotation is
    far less sensitive to intrinsics error than translation is, which is another reason the
    rotation channel is the trustworthy one here.
    """
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def gyro_relative_rotations(gyro: np.ndarray, dt: float) -> list[Rotation]:
    """Relative rotation per frame interval, by integrating angular velocity.

    gyro: (N, 3) rad/s, already resampled onto the video frame axis.

    Drift accumulates with TIME, and each interval here is 33 ms. Over one frame the error
    is negligible; we never integrate the gyro over long horizons for this purpose, because
    reprojection only ever needs relative pose over <= one action chunk.
    """
    out: list[Rotation] = []
    for i in range(len(gyro) - 1):
        # Midpoint of the interval -- trapezoidal, not left-endpoint.
        w = 0.5 * (gyro[i] + gyro[i + 1])
        out.append(Rotation.from_rotvec(w * dt))
    out.append(Rotation.identity())
    return out


def vision_relative_rotation(
    prev_gray: np.ndarray, gray: np.ndarray, K: np.ndarray
) -> tuple[Rotation | None, np.ndarray | None, int]:
    """An INDEPENDENT rotation estimate, from the camera alone.

    Tracks corners with Lucas-Kanade, recovers the essential matrix, decomposes it. Returns
    (rotation, unit translation direction, n_inliers). None when the scene gives too few
    correspondences to say anything -- which is honest, and happens on blurred frames.
    """
    p0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if p0 is None or len(p0) < _MIN_TRACKS:
        return None, None, 0

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **_LK)
    if p1 is None:
        return None, None, 0
    ok = st.ravel() == 1
    a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
    if len(a) < _MIN_TRACKS:
        return None, None, len(a)

    E, mask = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None, None, len(a)

    n, R, t, _ = cv2.recoverPose(E, a, b, K, mask=mask)
    if n < _MIN_TRACKS // 2:
        return None, None, int(n)

    return Rotation.from_matrix(R), t.ravel(), int(n)
