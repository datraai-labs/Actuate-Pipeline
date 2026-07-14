"""Stable-frame reprojection — and its refusal to guess.

Master Spec §L3's gate: *"future actions must land in the current device frame
consistently."* On a moving-camera rig, the un-reprojected wrist delta is

    hand_motion + camera_motion

A policy trained on that learns to predict head motion as if it were hand motion. So the
reprojection either has ego-motion and is correct, or it does not and **refuses**. There is
no `strict=False`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from actuate.canonical import (
    EgoMotionUnavailable,
    action_is_ego_contaminated,
    relative,
    reproject_future_pose,
)
from actuate.canonical.reproject import from_matrix, to_matrix
from actuate.schema import SE3

IDENT = (1.0, 0.0, 0.0, 0.0)


def _yaw(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# --- it refuses rather than degrades ------------------------------------------------------


def test_reprojection_REFUSES_without_ego_motion():
    """The whole point. A function that quietly returns the contaminated value is worse
    than one that stops."""
    wrist = SE3(position_m=(0.1, 0.2, 0.9), quaternion_wxyz=IDENT)

    with pytest.raises(EgoMotionUnavailable, match="conflates hand motion with head motion"):
        reproject_future_pose(wrist, None, None)

    with pytest.raises(EgoMotionUnavailable):
        reproject_future_pose(wrist, SE3(position_m=(0, 0, 0), quaternion_wxyz=IDENT), None)


def test_the_contamination_flag_is_true_when_camera_pose_is_absent():
    assert action_is_ego_contaminated(None) is True
    assert action_is_ego_contaminated(SE3(position_m=(0, 0, 0), quaternion_wxyz=IDENT)) is False


# --- and it is correct when it can be ------------------------------------------------------


def test_a_stationary_camera_leaves_the_pose_alone():
    cam = SE3(position_m=(1.0, 2.0, 3.0), quaternion_wxyz=_yaw(30))
    wrist = SE3(position_m=(0.1, 0.2, 0.9), quaternion_wxyz=IDENT)

    out = reproject_future_pose(wrist, cam, cam)
    assert np.allclose(out.position_m, wrist.position_m, atol=1e-9)


def test_pure_head_rotation_is_REMOVED_from_the_action():
    """THE bug this exists to fix.

    The hand is motionless in the world. The head turns 20 degrees. Un-reprojected, the
    wrist appears to swing across the camera frame — and a policy would learn to move the
    arm. Reprojected, the action is (correctly) zero motion.
    """
    cam_t = SE3(position_m=(0.0, 0.0, 0.0), quaternion_wxyz=IDENT)
    cam_t1 = SE3(position_m=(0.0, 0.0, 0.0), quaternion_wxyz=_yaw(20))

    # A world-fixed hand at (0, 0, 1) in cam_t's frame. After the head turns, the SAME
    # world point sits somewhere else in cam_t1's frame:
    world_hand = np.array([0.0, 0.0, 1.0, 1.0])
    hand_in_cam_t1 = np.linalg.inv(to_matrix(cam_t1)) @ world_hand
    wrist_future = SE3(
        position_m=tuple(float(v) for v in hand_in_cam_t1[:3]), quaternion_wxyz=IDENT
    )

    # Naive delta: the hand "moved" by this much. It did not.
    naive = np.linalg.norm(np.array(wrist_future.position_m) - np.array([0.0, 0.0, 1.0]))
    assert naive > 0.3, "the head turn should look like a big hand motion, un-reprojected"

    # Reprojected into cam_t: back where it started. Zero action, which is the truth.
    out = reproject_future_pose(wrist_future, cam_t, cam_t1)
    assert np.allclose(out.position_m, (0.0, 0.0, 1.0), atol=1e-9), (
        "reprojection failed to remove head motion from the action"
    )


def test_pure_hand_motion_survives_reprojection():
    """The converse: reprojection must not eat real hand motion."""
    cam = SE3(position_m=(0.0, 0.0, 0.0), quaternion_wxyz=IDENT)
    moved = SE3(position_m=(0.0, 0.0, 1.2), quaternion_wxyz=IDENT)

    out = reproject_future_pose(moved, cam, cam)
    assert np.allclose(out.position_m, (0.0, 0.0, 1.2), atol=1e-9)


# --- SE(3) plumbing ------------------------------------------------------------------------


def test_matrix_round_trip():
    p = SE3(position_m=(0.1, -0.2, 0.3), quaternion_wxyz=_yaw(37))
    back = from_matrix(to_matrix(p))
    assert np.allclose(back.position_m, p.position_m, atol=1e-12)
    assert np.allclose(np.abs(back.quaternion_wxyz), np.abs(p.quaternion_wxyz), atol=1e-9)


def test_relative_of_a_pose_to_itself_is_identity():
    p = SE3(position_m=(1.0, 2.0, 3.0), quaternion_wxyz=_yaw(15))
    r = relative(p, p)
    assert np.allclose(r.position_m, (0, 0, 0), atol=1e-12)
    assert np.allclose(np.abs(r.quaternion_wxyz), (1, 0, 0, 0), atol=1e-9)
