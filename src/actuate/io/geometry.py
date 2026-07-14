"""SE(3) helpers for lifting hand keypoints into the canonical frame.

Deliberately small and dependency-light (NumPy only). If this grows beyond a rotation
matrix and a quaternion conversion, switch to `roma` rather than hand-rolling more
quaternion math — Implementation Spec §5 names it precisely to avoid that class of bug.
"""

from __future__ import annotations

import numpy as np

# MediaPipe / MANO-style 21-keypoint hand topology.
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
THUMB_TIP = 4
INDEX_TIP = 8


def hand_frame_quaternion(landmarks_3d_m: np.ndarray) -> tuple[float, float, float, float] | None:
    """Derive a wrist orientation from the hand's own keypoint geometry.

    The palm defines a frame: the wrist->index_MCP direction and the wrist->pinky_MCP
    direction span the palm plane, and their cross product is the palm normal. We
    orthonormalize that into a rotation matrix and convert to a quaternion.

    This is a *derived* orientation, not a measured one — MediaPipe gives no wrist
    rotation, so callers must record it as estimated_vision_primary, never as measured.
    A real hand-mesh model (WiLoR, per Implementation Spec §3.1) returns MANO wrist
    rotation directly and should replace this.

    Returns None when the keypoints are degenerate (collinear/coincident), which does
    happen on real frames under occlusion — the caller must handle it rather than get a
    silently garbage pose.
    """
    if landmarks_3d_m.shape[0] <= PINKY_MCP:
        return None

    wrist = landmarks_3d_m[WRIST]
    index_mcp = landmarks_3d_m[INDEX_MCP]
    pinky_mcp = landmarks_3d_m[PINKY_MCP]

    x_axis = index_mcp - wrist
    aux = pinky_mcp - wrist

    nx = np.linalg.norm(x_axis)
    if nx < 1e-6 or np.linalg.norm(aux) < 1e-6:
        return None
    x_axis = x_axis / nx

    z_axis = np.cross(x_axis, aux)  # palm normal
    nz = np.linalg.norm(z_axis)
    if nz < 1e-6:  # collinear: no palm plane exists
        return None
    z_axis = z_axis / nz

    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    return _rotation_matrix_to_quaternion_wxyz(R)


def _rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Shepperd's method — branch on the largest diagonal term for numerical stability."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
