"""Pose estimation seam. The scripted policy and learned coarse policy consume task
poses through this interface. In sim it returns ground truth; on real hardware it is
backed by a detector plus FT. Keeping the seam clean means the policy code is identical
across backends.

This module is torch-free and does not import mujoco. The sim implementation reads
ground-truth site positions through the backend's site_xpos accessor (duck-typed), so
no backend dependency leaks above the HAL.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Perception(ABC):
    """Task-relevant poses, all as world-frame [x, y, z] positions unless noted."""

    @abstractmethod
    def grasp_point(self) -> np.ndarray:
        """World position of the point on the drive the gripper should grasp."""

    @abstractmethod
    def connector_offset(self, ee_pos: np.ndarray) -> np.ndarray:
        """Vector from the ee reference point to the drive connector tip, while grasped."""

    @abstractmethod
    def bay_mouth(self) -> np.ndarray:
        """World position of the bay opening (pre-insertion alignment)."""

    @abstractmethod
    def bay_socket(self) -> np.ndarray:
        """World position of the seat target deep in the bay."""


class SimGroundTruthPerception(Perception):
    """Reads exact poses from the sim backend. Used to drive the scripted expert and to
    generate demos. The backend must expose site_xpos(name)."""

    def __init__(self, robot) -> None:
        self.robot = robot

    def grasp_point(self) -> np.ndarray:
        return self.robot.site_xpos("drive_grasp")

    def connector_offset(self, ee_pos: np.ndarray) -> np.ndarray:
        return self.robot.site_xpos("drive_connector") - np.asarray(ee_pos)

    def bay_mouth(self) -> np.ndarray:
        return self.robot.site_xpos("bay_mouth")

    def bay_socket(self) -> np.ndarray:
        return self.robot.site_xpos("bay_socket")


class NoisyPerception(Perception):
    """Wraps a perception source and adds a fixed seeded offset to the reported bay pose.

    Used to induce recovery scenarios: the coarse alignment is off, so the force-search
    primitive must recover and still seat. The grasp stays accurate (a firm grip is held)."""

    def __init__(self, inner: Perception, pos_noise: float, seed: int = 0) -> None:
        self.inner = inner
        rng = np.random.default_rng(seed)
        # In-plane error only (x, y): the insertion height (z) is held firm by the
        # controller, and offsets beyond the slot clearance in z are not recoverable.
        self._offset = np.array([rng.uniform(-pos_noise, pos_noise), rng.uniform(-pos_noise, pos_noise), 0.0])

    def grasp_point(self) -> np.ndarray:
        return self.inner.grasp_point()

    def connector_offset(self, ee_pos: np.ndarray) -> np.ndarray:
        return self.inner.connector_offset(ee_pos)

    def bay_mouth(self) -> np.ndarray:
        return self.inner.bay_mouth() + self._offset

    def bay_socket(self) -> np.ndarray:
        return self.inner.bay_socket() + self._offset
