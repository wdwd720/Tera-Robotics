"""Small numpy quaternion utilities (Hamilton convention, wxyz order).

Torch-free and mujoco-free, so motion primitives and policies above the HAL can do
orientation math without importing a backend.
"""
from __future__ import annotations

import numpy as np


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a unit quaternion, wrapped to [-pi, pi]."""
    q = q / np.linalg.norm(q)
    if q[0] < 0:  # canonical hemisphere
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3)
    return (q[1:] / s) * angle


def quat_error(cur: np.ndarray, des: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking cur to des: axisangle(des * conj(cur))."""
    return quat_to_axisangle(quat_mul(des, quat_conj(cur)))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to axis."""
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v
