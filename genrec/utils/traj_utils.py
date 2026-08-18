"""Pure-numpy camera-trajectory helpers (quaternion SLERP, pose interpolation).

Extracted from the GenRec research code so the inference entry points do not
depend on the 3D Gaussian Splatting fitting stack. No torch / I/O dependencies.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) unit quaternions in (w, x, y, z) order."""
    R = R.astype(np.float64)
    N = R.shape[0]
    q = np.empty((N, 4), dtype=np.float64)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(N):
        Ri = R[i]
        t = trace[i]
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            qw = 0.25 * s
            qx = (Ri[2, 1] - Ri[1, 2]) / s
            qy = (Ri[0, 2] - Ri[2, 0]) / s
            qz = (Ri[1, 0] - Ri[0, 1]) / s
        elif Ri[0, 0] > Ri[1, 1] and Ri[0, 0] > Ri[2, 2]:
            s = math.sqrt(1.0 + Ri[0, 0] - Ri[1, 1] - Ri[2, 2]) * 2.0
            qw = (Ri[2, 1] - Ri[1, 2]) / s
            qx = 0.25 * s
            qy = (Ri[0, 1] + Ri[1, 0]) / s
            qz = (Ri[0, 2] + Ri[2, 0]) / s
        elif Ri[1, 1] > Ri[2, 2]:
            s = math.sqrt(1.0 + Ri[1, 1] - Ri[0, 0] - Ri[2, 2]) * 2.0
            qw = (Ri[0, 2] - Ri[2, 0]) / s
            qx = (Ri[0, 1] + Ri[1, 0]) / s
            qy = 0.25 * s
            qz = (Ri[1, 2] + Ri[2, 1]) / s
        else:
            s = math.sqrt(1.0 + Ri[2, 2] - Ri[0, 0] - Ri[1, 1]) * 2.0
            qw = (Ri[1, 0] - Ri[0, 1]) / s
            qx = (Ri[0, 2] + Ri[2, 0]) / s
            qy = (Ri[1, 2] + Ri[2, 1]) / s
            qz = 0.25 * s
        q[i] = (qw, qx, qy, qz)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) (w, x, y, z) -> (N, 3, 3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _slerp_pair(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SLERP between two quaternions for an array of `t` in [0, 1].
    Both quaternions in (w, x, y, z). Returns (len(t), 4)."""
    # Take shortest path.
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        # Near-parallel — fall back to lerp + renormalize.
        out = (1.0 - t)[:, None] * q0[None, :] + t[:, None] * q1[None, :]
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0[:, None] * q0[None, :] + s1[:, None] * q1[None, :]


def _interpolate_trajectory(
    c2w: np.ndarray,    # (M, 4, 4)
    K: np.ndarray,      # (M, 3, 3)
    factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert (factor - 1) interpolated poses between every consecutive pair.

    SLERP for rotation, linear for translation, linear for intrinsics.
    Output length = (M - 1) * factor + 1. factor <= 1 returns inputs unchanged.
    """
    M = c2w.shape[0]
    if factor <= 1 or M < 2:
        return c2w, K

    quats = _rotmat_to_quat(c2w[:, :3, :3])  # (M, 4)
    trans = c2w[:, :3, 3].astype(np.float64)  # (M, 3)
    Kf = K.astype(np.float64)                 # (M, 3, 3)

    out_len = (M - 1) * factor + 1
    out_c2w = np.zeros((out_len, 4, 4), dtype=np.float32)
    out_c2w[:, 3, 3] = 1.0
    out_K = np.zeros((out_len, 3, 3), dtype=np.float32)

    # `factor` samples per segment at t = 0, 1/factor, ..., (factor-1)/factor.
    # The segment endpoint at t=1 is the start of the next segment, except for
    # the very last frame which we append at the end.
    t_seg = np.arange(factor, dtype=np.float64) / float(factor)  # length=factor
    for i in range(M - 1):
        q_interp = _slerp_pair(quats[i], quats[i + 1], t_seg)        # (factor, 4)
        R_interp = _quat_to_rotmat(q_interp)                         # (factor, 3, 3)
        t_interp = (1.0 - t_seg)[:, None] * trans[i] + t_seg[:, None] * trans[i + 1]
        K_interp = (1.0 - t_seg)[:, None, None] * Kf[i] + t_seg[:, None, None] * Kf[i + 1]
        s = i * factor
        out_c2w[s : s + factor, :3, :3] = R_interp.astype(np.float32)
        out_c2w[s : s + factor, :3, 3] = t_interp.astype(np.float32)
        out_K[s : s + factor] = K_interp.astype(np.float32)

    # Final endpoint copies the last source pose verbatim.
    out_c2w[-1] = c2w[-1]
    out_K[-1] = K[-1]
    return out_c2w, out_K
