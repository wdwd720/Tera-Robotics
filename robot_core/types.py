"""The fixed observation/action contract shared by every backend.

This module is the single source of truth for dimensions and layouts. It is
torch-free and mujoco-free: nothing here depends on a particular backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Dimensions.
N_JOINTS = 6
ACTION_DIM = 7
STATE_DIM = 26  # 6 joint_pos + 6 joint_vel + 1 gripper + 7 ee_pose + 6 ft

# Gripper: actuator ctrl is meters in [0, GRIP_CTRL_MAX]; the normalized observation
# is opening in [0, 1] with 1 = open.
GRIP_CTRL_MAX = 0.06

# 26-D state layout (the flat proprioception vector for LeRobot observation.state).
# images and sim_time are deliberately NOT part of this vector.
STATE_SLICES: dict[str, slice] = {
    "joint_pos": slice(0, 6),
    "joint_vel": slice(6, 12),
    "gripper": slice(12, 13),
    "ee_pose": slice(13, 20),
    "ft": slice(20, 26),
}

# Action layout: [dx, dy, dz, drx, drz, dry, grip].
#   [0:3] base-frame position delta (meters)
#   [3:6] axis-angle rotation delta (radians), delivered in (rx, rz, ry) SLOT order
#   [6]   target gripper opening in [0, 1]
# The rotation block is NOT in xyz order. ROT_SLOT_TO_XYZ gives the action indices
# that hold the (x, y, z) rotation components, so xyz = action[[3, 5, 4]].
POS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 6)
GRIP_INDEX = 6
ROT_SLOT_TO_XYZ: tuple[int, int, int] = (3, 5, 4)

IMAGE_KEYS = ("front", "wrist")


def concat_state(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    gripper: float,
    ee_pose: np.ndarray,
    ft: np.ndarray,
) -> np.ndarray:
    """Build the canonical 26-D state vector. The one place state is assembled."""
    state = np.empty(STATE_DIM, dtype=np.float32)
    state[STATE_SLICES["joint_pos"]] = np.asarray(joint_pos, dtype=np.float32)
    state[STATE_SLICES["joint_vel"]] = np.asarray(joint_vel, dtype=np.float32)
    state[STATE_SLICES["gripper"]] = np.float32(gripper)
    state[STATE_SLICES["ee_pose"]] = np.asarray(ee_pose, dtype=np.float32)
    state[STATE_SLICES["ft"]] = np.asarray(ft, dtype=np.float32)
    return state


def action_rotation_xyz(delta: np.ndarray) -> np.ndarray:
    """Extract the axis-angle rotation delta in xyz order from the slot layout."""
    return np.array([delta[k] for k in ROT_SLOT_TO_XYZ], dtype=np.float64)


@dataclass(frozen=True)
class Observation:
    """One synchronized observation. ft is reported in the wrist site local frame."""

    images: dict[str, np.ndarray]  # name -> (H, W, 3) uint8
    joint_pos: np.ndarray  # (6,) float32, rad
    joint_vel: np.ndarray  # (6,) float32, rad/s
    gripper: float  # [0, 1], 1 = open
    ee_pose: np.ndarray  # (7,) float32, x y z qw qx qy qz, base frame
    ft: np.ndarray  # (6,) float32, fx fy fz tx ty tz, wrist site frame
    sim_time: float

    @property
    def state(self) -> np.ndarray:
        """The 26-D proprioception vector for observation.state."""
        return concat_state(
            self.joint_pos, self.joint_vel, self.gripper, self.ee_pose, self.ft
        )

    def validate(self) -> None:
        assert self.joint_pos.shape == (N_JOINTS,), self.joint_pos.shape
        assert self.joint_vel.shape == (N_JOINTS,), self.joint_vel.shape
        assert self.ee_pose.shape == (7,), self.ee_pose.shape
        assert self.ft.shape == (6,), self.ft.shape
        assert 0.0 <= self.gripper <= 1.0, self.gripper
        for name, img in self.images.items():
            assert img.ndim == 3 and img.shape[2] == 3, (name, img.shape)
            assert img.dtype == np.uint8, (name, img.dtype)
        assert self.state.shape == (STATE_DIM,)


@dataclass
class Action:
    """A 7-D end-effector delta. See the action layout above."""

    delta: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM))

    def __post_init__(self) -> None:
        self.delta = np.asarray(self.delta, dtype=np.float64).reshape(ACTION_DIM)

    @property
    def dpos(self) -> np.ndarray:
        return self.delta[POS_SLICE].copy()

    @property
    def drot_xyz(self) -> np.ndarray:
        return action_rotation_xyz(self.delta)

    @property
    def grip(self) -> float:
        return float(self.delta[GRIP_INDEX])

    @classmethod
    def zeros(cls) -> "Action":
        return cls(np.zeros(ACTION_DIM))

    @classmethod
    def from_parts(cls, dpos: np.ndarray, drot_xyz: np.ndarray, grip: float) -> "Action":
        d = np.zeros(ACTION_DIM)
        d[POS_SLICE] = dpos
        ix, iy, iz = ROT_SLOT_TO_XYZ
        d[ix], d[iy], d[iz] = float(drot_xyz[0]), float(drot_xyz[1]), float(drot_xyz[2])
        d[GRIP_INDEX] = grip
        return cls(d)
