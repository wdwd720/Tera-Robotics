"""The one hardware abstraction layer. Every backend implements this interface.

Nothing above the HAL (data engine, training, eval, policies, dashboard) knows which
backend is running. This interface is torch-free and mujoco-free.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import Action, Observation


class Robot(ABC):
    """Backend-agnostic robot contract.

    supports_torque declares whether the backend provides true joint torque control.
    Backends that cannot (position-only arms) set it False and run a position shim.
    """

    supports_torque: bool = True

    # High-level loop.
    @abstractmethod
    def reset(self) -> Observation:
        """Reset to the home configuration and return the first observation."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Apply a 7-D end-effector delta and return the resulting observation."""

    @abstractmethod
    def get_obs(self, with_images: bool = True) -> Observation:
        """Read the current observation. Images are rendered only if requested."""

    @abstractmethod
    def seated(self) -> bool:
        """Single seat-detection hook shared by sim and hardware call sites."""

    def set_stiffness_mode(self, mode: str) -> None:
        """Optional compliance hint for the low-level controller (hold/firm/search).

        Backends that cannot vary stiffness may ignore this (the default no-op).
        """

    def estopped(self) -> bool:
        """True once a soft e-stop has latched. Cleared by reset()."""
        return False

    def set_payload_compensation(self, force_z: float) -> None:
        """Optional: feedforward an upward force (N) to hold a grasped payload whose
        weight the low-level controller does not otherwise model. Default no-op."""

    def attach_payload(self) -> None:
        """Optional: declare a firm grasp so the backend treats the held object as rigid.

        In sim this welds the object to the gripper. On real hardware the physical grip
        does this, so it is a no-op. Compliance still comes from the arm controller."""

    def detach_payload(self) -> None:
        """Optional: release a previously attached payload. Default no-op."""

    # Low-level contract every backend must expose.
    @abstractmethod
    def set_joint_torque(self, tau6: np.ndarray) -> None:
        """Command joint torques (MIT mode on hardware)."""

    @abstractmethod
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_pos, joint_vel), each (6,)."""

    @abstractmethod
    def read_ft(self) -> np.ndarray:
        """Return the 6-axis wrist wrench (fx fy fz tx ty tz) in the wrist site frame."""

    @abstractmethod
    def set_gripper(self, opening: float) -> None:
        """Command the gripper opening in [0, 1], 1 = open."""

    @abstractmethod
    def render(self, camera: str, height: int = 128, width: int = 128) -> np.ndarray:
        """Render a camera frame as (H, W, 3) uint8."""

    def close(self) -> None:
        """Release any backend resources."""

    def info(self) -> dict:
        """Optional backend diagnostics."""
        return {}
