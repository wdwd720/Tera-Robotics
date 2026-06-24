"""SafetyController: every action and torque passes through it.

Pure numpy, no backend dependency. Clamps action deltas, target poses, and joint
torques, and raises a soft e-stop on excessive joint velocity or wrist force.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Action, N_JOINTS


@dataclass
class SafetyLimits:
    max_dpos: float = 0.02  # m per action tick (position delta norm)
    max_drot: float = 0.10  # rad per action tick (rotation delta norm)
    # Workspace box (base frame). Contains the cradle, the bay, and the home pose.
    workspace_min: tuple[float, float, float] = (0.25, -0.30, 0.08)
    workspace_max: tuple[float, float, float] = (0.78, 0.30, 0.60)
    # Per-joint torque limits (6,). Populated from the model actuator ctrlrange.
    tau_limit: np.ndarray = field(default_factory=lambda: np.full(N_JOINTS, 50.0))
    vel_estop: float = 6.0  # rad/s, any joint
    ft_estop: float = 60.0  # N, wrist force norm
    insert_force_max: float = 25.0  # N, bounded insert-axis push


class SafetyController:
    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()
        self.estop_reason: str | None = None

    @property
    def tripped(self) -> bool:
        return self.estop_reason is not None

    def reset(self) -> None:
        self.estop_reason = None

    def clamp_action(self, action: Action) -> Action:
        """Clamp the per-tick position and rotation deltas by norm; grip to [0, 1]."""
        dpos = action.dpos
        n = float(np.linalg.norm(dpos))
        if n > self.limits.max_dpos:
            dpos = dpos * (self.limits.max_dpos / n)
        drot = action.drot_xyz
        nr = float(np.linalg.norm(drot))
        if nr > self.limits.max_drot:
            drot = drot * (self.limits.max_drot / nr)
        grip = float(np.clip(action.grip, 0.0, 1.0))
        return Action.from_parts(dpos, drot, grip)

    def clamp_pose_to_box(self, pos: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.limits.workspace_min)
        hi = np.asarray(self.limits.workspace_max)
        return np.clip(np.asarray(pos, dtype=float), lo, hi)

    def clamp_torque(self, tau: np.ndarray) -> np.ndarray:
        lim = self.limits.tau_limit
        return np.clip(np.asarray(tau, dtype=float), -lim, lim)

    def check(self, joint_vel: np.ndarray, ft_force: np.ndarray) -> str | None:
        """Soft e-stop on excessive joint velocity or wrist force. Latches a reason."""
        if np.any(np.abs(joint_vel) > self.limits.vel_estop):
            self.estop_reason = (
                f"joint velocity {float(np.max(np.abs(joint_vel))):.2f} "
                f"> {self.limits.vel_estop}"
            )
        elif float(np.linalg.norm(ft_force)) > self.limits.ft_estop:
            self.estop_reason = (
                f"wrist force {float(np.linalg.norm(ft_force)):.2f} > {self.limits.ft_estop}"
            )
        return self.estop_reason

    def insert_force_exceeded(self, insert_force: float) -> bool:
        return abs(insert_force) > self.limits.insert_force_max
