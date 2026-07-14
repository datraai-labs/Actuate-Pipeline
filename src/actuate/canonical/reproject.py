"""Camera-centered stable-frame reprojection -- Master Spec §L3.

The gate: *"future actions must land in the current device frame consistently."*

### Why this exists

On a moving-camera rig (head-mounted, UMI), per-frame hand keypoints are expressed in the
**camera frame at that instant**. The camera is on a person's head, and it moves. So the
naive "action = wrist position at t+1 minus wrist position at t" is not the hand's motion:

    delta_camera(t) = hand_motion(t) + camera_motion(t)

A policy trained on that learns to predict head motion as if it were hand motion. The
contamination is largest exactly when it matters -- during a reach, when the demonstrator
is also turning to look.

Reprojection removes it by expressing the future wrist pose in the **current** camera
frame, using the ego-motion between the two instants:

    T_cam(t)->cam(t+k)  =  inv(T_world->cam(t)) @ T_world->cam(t+k)
    wrist_in_frame_t    =  T_cam(t)->cam(t+k) @ wrist_cam(t+k)

### Why it is not currently possible on our data

That requires `camera_pose` -- a world-frame SE(3) per frame, from L1 SLAM (ORB-SLAM3) or
Aria MPS. **L1 ego-motion is not built.** Our one real capture has no camera_pose at all.

So `reproject_future_pose` **raises** when ego-motion is missing rather than silently
returning the contaminated delta. A function that quietly degrades to the wrong answer is
worse than one that refuses: the caller then has to decide, in the open, what to do about
it -- which is `canonical.build`'s job, and which it records in `derivation_notes` rather
than hiding.
"""

from __future__ import annotations

import numpy as np

from actuate.schema import SE3


class EgoMotionUnavailable(RuntimeError):
    """Reprojection was requested but there is no camera_pose to reproject through.

    Not a warning. On a moving-camera rig, an un-reprojected action is contaminated by
    head motion, and shipping it as if it were hand motion is the failure this exception
    exists to make impossible to do by accident.
    """


def _quat_to_R(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _R_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def to_matrix(pose: SE3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(pose.quaternion_wxyz)
    T[:3, 3] = pose.position_m
    return T


def from_matrix(T: np.ndarray) -> SE3:
    return SE3(
        position_m=tuple(float(v) for v in T[:3, 3]),
        quaternion_wxyz=_R_to_quat(T[:3, :3]),
    )


def relative(a: SE3, b: SE3) -> SE3:
    """The pose of `b` expressed in `a`'s frame: inv(a) @ b."""
    return from_matrix(np.linalg.inv(to_matrix(a)) @ to_matrix(b))


def reproject_future_pose(
    wrist_future: SE3,
    camera_pose_now: SE3 | None,
    camera_pose_future: SE3 | None,
) -> SE3:
    """Express a future wrist pose (given in the FUTURE camera frame) in the CURRENT one.

    Raises EgoMotionUnavailable if either camera pose is missing -- see the module
    docstring. There is deliberately no `strict=False` escape hatch.
    """
    if camera_pose_now is None or camera_pose_future is None:
        raise EgoMotionUnavailable(
            "stable-frame reprojection needs camera_pose at both t and t+k, and L1 "
            "ego-motion (ORB-SLAM3 / Aria MPS) is not built. Without it, a wrist delta on "
            "a moving-camera rig conflates hand motion with head motion -- it is not an "
            "action. Refusing to return the contaminated value."
        )

    T_now_from_future = np.linalg.inv(to_matrix(camera_pose_now)) @ to_matrix(
        camera_pose_future
    )
    return from_matrix(T_now_from_future @ to_matrix(wrist_future))


def action_is_ego_contaminated(camera_pose_now: SE3 | None) -> bool:
    """True when the action for this frame is a raw camera-frame delta on a moving rig.

    Callers that emit such an action MUST record it -- see `CanonicalEpisode.derivation_notes`.
    """
    return camera_pose_now is None
