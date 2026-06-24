"""Cartesian impedance control with gravity/Coriolis compensation.

Builds an operational-space wrench from a pose error and end-effector twist, maps it
to joint torque with the site Jacobian transpose, and adds qfrc_bias. Stiffness is
anisotropic: firm along the insertion axis, soft laterally during a search.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..types import N_JOINTS
from .kinematics import pose_error, site_jacobian, site_pose

# Insertion axis in the world/base frame for this task (the bay opens along -x and the
# drive seats moving +x). Lateral compliance during search is in the perpendicular axes.
INSERT_AXIS = np.array([1.0, 0.0, 0.0])

# Stiffness profiles per mode: [kx, ky, kz, krx, kry, krz].
STIFFNESS = {
    "hold": np.array([900.0, 900.0, 900.0, 150.0, 150.0, 150.0]),
    "firm": np.array([1500.0, 1500.0, 1500.0, 200.0, 200.0, 200.0]),
    # search: firm along x (insertion) and z (hold height against gravity), soft in y
    # (lateral search), but firm rotation so the long drive cannot cock and jam in the
    # tight slot.
    "search": np.array([400.0, 120.0, 500.0, 150.0, 150.0, 150.0]),
}


class CartesianImpedance:
    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site_id: int,
        n_arm: int = N_JOINTS,
        damping_ratio: float = 1.0,
    ) -> None:
        self.model = model
        self.ee_site_id = ee_site_id
        self.n_arm = n_arm
        self.damping_ratio = damping_ratio
        # Optional world-frame feedforward wrench, e.g. to hold a grasped payload whose
        # weight is not in the arm's qfrc_bias (the drive is a separate freejoint body).
        self.ff_wrench = np.zeros(6)

    def stiffness(self, mode: str) -> np.ndarray:
        if mode not in STIFFNESS:
            raise ValueError(f"unknown impedance mode: {mode}")
        return STIFFNESS[mode]

    def torque(
        self,
        data: mujoco.MjData,
        target7: np.ndarray,
        mode: str = "hold",
    ) -> np.ndarray:
        """Joint torques (n_arm,) driving the ee_site toward target7 under mode."""
        cur = site_pose(self.model, data, self.ee_site_id)
        err = pose_error(cur, target7)
        jac = site_jacobian(self.model, data, self.ee_site_id, self.n_arm)
        twist = jac @ data.qvel[: self.n_arm]
        kp = self.stiffness(mode)
        kd = 2.0 * self.damping_ratio * np.sqrt(kp)
        wrench = kp * err - kd * twist + self.ff_wrench
        tau = jac.T @ wrench
        tau += data.qfrc_bias[: self.n_arm]  # gravity + Coriolis feedforward
        return tau
