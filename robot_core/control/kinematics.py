"""Kinematics helpers for the torque-controlled sim backend: site pose, site
Jacobian, world-frame pose error, and damped least-squares IK.

These use the MuJoCo model/data directly. Hardware backends provide their own
torque control and do not import this module.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..types import N_JOINTS


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Return the site pose as (7,) [x y z qw qx qy qz] in the world frame."""
    pos = data.site_xpos[site_id].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return np.concatenate([pos, quat])


def site_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int, n_arm: int = N_JOINTS
) -> np.ndarray:
    """Stacked (6, n_arm) site Jacobian [jacp; jacr], arm columns only."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])[:, :n_arm]


def pose_error(cur7: np.ndarray, des7: np.ndarray) -> np.ndarray:
    """6-D pose error des - cur: position delta plus world-frame axis-angle from quats.

    Uses mju_subQuat(res, qa, qb), which yields the rotation taking qb to qa, so the
    orientation error is computed as (des, cur).
    """
    perr = des7[:3] - cur7[:3]
    rerr = np.zeros(3)
    mujoco.mju_subQuat(rerr, des7[3:7], cur7[3:7])
    return np.concatenate([perr, rerr])


def dls_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target7: np.ndarray,
    q_seed: np.ndarray,
    *,
    n_arm: int = N_JOINTS,
    damping: float = 1e-2,
    iters: int = 100,
    tol: float = 1e-4,
    step: float = 1.0,
) -> tuple[np.ndarray, int, float]:
    """Damped least-squares IK over the arm DoFs on a scratch MjData copy.

    Does not disturb the live data. Returns (q_arm, iterations, final_error_norm).
    """
    dc = mujoco.MjData(model)
    mujoco.mj_copyData(dc, model, data)
    dc.qpos[:n_arm] = q_seed
    lo = model.jnt_range[:n_arm, 0]
    hi = model.jnt_range[:n_arm, 1]
    err_norm = np.inf
    it = 0
    for it in range(iters):
        mujoco.mj_forward(model, dc)
        cur = site_pose(model, dc, site_id)
        err = pose_error(cur, target7)
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            break
        jac = site_jacobian(model, dc, site_id, n_arm)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(6), err)
        dc.qpos[:n_arm] = np.clip(dc.qpos[:n_arm] + step * dq, lo, hi)
    return dc.qpos[:n_arm].copy(), it, err_norm
