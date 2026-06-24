"""Backend-agnostic motion primitives built on the Robot HAL.

Each primitive drives the end-effector with 7-D ee-delta actions and reads back ee
pose and force from observations. Nothing here imports a backend. The scripted reseat
policy composes these with target poses computed from perception.
"""
from __future__ import annotations

import numpy as np

from ..math_utils import perp_basis, quat_error, quat_to_mat
from ..robot import Robot
from ..types import Action

INSERT_AXIS_X = np.array([1.0, 0.0, 0.0])


def _pose_error(ee_pose: np.ndarray, target7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dpos = target7[:3] - ee_pose[:3]
    drot = quat_error(ee_pose[3:7], target7[3:7])
    return dpos, drot


def move_to_pose(
    robot: Robot,
    target7: np.ndarray,
    grip: float,
    *,
    mode: str = "hold",
    pos_tol: float = 2e-3,
    rot_tol: float = 0.03,
    max_steps: int = 800,
    gain: float = 1.0,
) -> bool:
    """Servo the ee to an absolute target pose, holding the given gripper opening."""
    robot.set_stiffness_mode(mode)
    obs = robot.get_obs(with_images=False)
    for _ in range(max_steps):
        dpos, drot = _pose_error(obs.ee_pose, target7)
        if np.linalg.norm(dpos) < pos_tol and np.linalg.norm(drot) < rot_tol:
            return True
        obs = robot.step(Action.from_parts(dpos * gain, drot * gain, grip))
        if robot.estopped():
            return False
    dpos, _ = _pose_error(obs.ee_pose, target7)
    return float(np.linalg.norm(dpos)) < 2 * pos_tol and not robot.estopped()


def grasp(robot: Robot, close_to: float = 0.1, *, settle: int = 50) -> bool:
    """Close the gripper to a width below contact and confirm an object is held."""
    for _ in range(settle):
        obs = robot.step(Action.from_parts(np.zeros(3), np.zeros(3), close_to))
        if robot.estopped():
            return False
    # Fingers stalled above fully-closed means something is between them.
    return obs.gripper > 0.2


def open_gripper(robot: Robot, *, settle: int = 25) -> bool:
    for _ in range(settle):
        robot.step(Action.from_parts(np.zeros(3), np.zeros(3), 1.0))
        if robot.estopped():
            return False
    return True


def lift(robot: Robot, dz: float, grip: float, **kw) -> bool:
    target = robot.get_obs(with_images=False).ee_pose.copy()
    target[2] += dz
    return move_to_pose(robot, target, grip, **kw)


def retreat(robot: Robot, dz: float = 0.10, grip: float = 1.0, **kw) -> bool:
    return lift(robot, dz, grip, **kw)


def align_at_mouth(robot: Robot, mouth_target7: np.ndarray, grip: float, **kw) -> bool:
    """Move the held drive to the pre-insertion pose at the bay mouth."""
    return move_to_pose(robot, mouth_target7, grip, **kw)


def insert_with_search(
    robot: Robot,
    base_pose7: np.ndarray,
    insert_axis: np.ndarray = INSERT_AXIS_X,
    *,
    max_depth: float = 0.06,
    advance: float = 0.0006,
    spiral_radius: float = 0.006,
    spiral_turns: float = 4.0,
    grip: float = 0.1,
    force_max: float = 15.0,
    max_steps: int = 2500,
) -> bool:
    """Advance slowly along the insertion axis while overlaying an Archimedean spiral
    laterally, with soft lateral stiffness and a bounded insert-axis push force.

    base_pose7 is the pre-insertion ee pose (at the mouth). Returns True once seated.
    """
    robot.set_stiffness_mode("search")
    axis = np.asarray(insert_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u, v = perp_basis(axis)
    depth = 0.0
    for _ in range(max_steps):
        if robot.seated():
            robot.set_stiffness_mode("hold")
            return True
        obs = robot.get_obs(with_images=False)
        frac = depth / max_depth if max_depth > 0 else 0.0
        theta = 2.0 * np.pi * spiral_turns * frac
        radius = spiral_radius * frac
        offset = radius * (np.cos(theta) * u + np.sin(theta) * v)
        target_pos = base_pose7[:3] + depth * axis + offset
        dpos = target_pos - obs.ee_pose[:3]
        drot = quat_error(obs.ee_pose[3:7], base_pose7[3:7])

        # Bound the insert-axis push: project the wrist force into the world frame.
        f_world = quat_to_mat(obs.ee_pose[3:7]) @ obs.ft[:3]
        f_insert = float(f_world @ axis)
        if abs(f_insert) < force_max and depth < max_depth:
            depth += advance
        elif abs(f_insert) >= force_max:
            depth = max(depth - 0.5 * advance, 0.0)

        robot.step(Action.from_parts(dpos, drot, grip))
        if robot.estopped():
            return False
    robot.set_stiffness_mode("hold")
    return robot.seated()


def confirm_seat(robot: Robot, grip: float, *, hold_ticks: int = 25) -> bool:
    """Hold the seated pose and require the seat condition to stay true."""
    seated_count = 0
    for _ in range(hold_ticks):
        robot.step(Action.from_parts(np.zeros(3), np.zeros(3), grip))
        if robot.estopped():
            return False
        seated_count = seated_count + 1 if robot.seated() else 0
    return seated_count >= max(hold_ticks // 2, 5)
