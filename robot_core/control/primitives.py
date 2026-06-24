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


def _hold_step(robot: Robot, target7: np.ndarray, grip: float):
    """One step servoing toward a fixed target pose. Because step() interprets the action
    as a delta from the current ee, feeding (target - current) each tick gives a real
    position setpoint (the error grows as it sags, generating restoring force). Feeding a
    zero delta would instead let an unmodeled payload drag the arm down."""
    ee = robot.get_obs(with_images=False).ee_pose
    dpos, drot = _pose_error(ee, target7)
    return robot.step(Action.from_parts(dpos, drot, grip))


def move_to_pose(
    robot: Robot,
    target7: np.ndarray,
    grip: float,
    *,
    mode: str = "hold",
    pos_tol: float = 2e-3,
    rot_tol: float = 0.03,
    speed: float = 0.10,
    max_steps: int = 1200,
    control_dt: float = 0.02,
) -> bool:
    """Servo the ee to an absolute target pose along a speed-limited straight line.

    A waypoint advances from the start toward the target at `speed` (m/s); each step the
    commanded delta is just (waypoint - current ee). Because the waypoint moves slowly the
    tracking error stays small, so the pull force stays gentle and a grasped payload is not
    yanked or tilted. Holds the given gripper opening throughout.
    """
    robot.set_stiffness_mode(mode)
    start = robot.get_obs(with_images=False).ee_pose[:3].copy()
    total = float(np.linalg.norm(target7[:3] - start))
    traveled = 0.0
    for _ in range(max_steps):
        obs = robot.get_obs(with_images=False)
        traveled = min(total, traveled + speed * control_dt)
        frac = 1.0 if total < 1e-9 else traveled / total
        waypoint = start + frac * (target7[:3] - start)
        dpos = waypoint - obs.ee_pose[:3]
        drot = quat_error(obs.ee_pose[3:7], target7[3:7])
        if (
            frac >= 1.0
            and np.linalg.norm(target7[:3] - obs.ee_pose[:3]) < pos_tol
            and np.linalg.norm(drot) < rot_tol
        ):
            return True
        robot.step(Action.from_parts(dpos, drot, grip))
        if robot.estopped():
            return False
    final = robot.get_obs(with_images=False)
    return (
        float(np.linalg.norm(target7[:3] - final.ee_pose[:3])) < 2 * pos_tol
        and not robot.estopped()
    )


def grasp(robot: Robot, close_to: float = 0.0, *, ramp_steps: int = 20, settle: int = 25) -> bool:
    """Close the gripper to a width below contact and confirm an object is held.

    The close is ramped so the fingers do not slam the object (which would spike the
    wrist force into the e-stop). The arm holds its pose throughout."""
    target = robot.get_obs(with_images=False).ee_pose.copy()
    start = robot.get_obs(with_images=False).gripper
    for i in range(ramp_steps):
        g = start + (close_to - start) * ((i + 1) / ramp_steps)
        _hold_step(robot, target, g)
        if robot.estopped():
            return False
    for _ in range(settle):
        obs = _hold_step(robot, target, close_to)
        if robot.estopped():
            return False
    # Fingers stalled above fully-closed means something is between them.
    return obs.gripper > 0.2


def hold(robot: Robot, steps: int, grip: float) -> bool:
    """Hold the current pose for a number of steps (e.g. to let a payload settle)."""
    target = robot.get_obs(with_images=False).ee_pose.copy()
    for _ in range(steps):
        _hold_step(robot, target, grip)
        if robot.estopped():
            return False
    return True


def open_gripper(robot: Robot, *, settle: int = 25) -> bool:
    target = robot.get_obs(with_images=False).ee_pose.copy()
    for _ in range(settle):
        _hold_step(robot, target, 1.0)
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
    seat_force: float = 6.0,
    max_steps: int = 2500,
    monitor=None,
) -> bool:
    """Advance slowly along the insertion axis while overlaying an Archimedean spiral
    laterally, with soft lateral stiffness and a bounded insert-axis push force.

    base_pose7 is the pre-insertion ee pose (at the mouth). Once the seat condition is
    seen the drive is pushed firmly home until the insert-axis force rises (the connector
    bottoming against the socket back), which is the real seat-detent signal. Returns
    True when firmly seated. If given, monitor(info) is called each step.
    """
    robot.set_stiffness_mode("search")
    # Conservative path for backends without true torque control (position shim): bound
    # the push force lower since force is estimated, not measured/commanded directly.
    if not getattr(robot, "supports_torque", True):
        force_max = min(force_max, 8.0)
        seat_force = min(seat_force, 4.0)
    axis = np.asarray(insert_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    u, v = perp_basis(axis)
    depth = 0.0
    seated_seen = False
    for _ in range(max_steps):
        obs = robot.get_obs(with_images=False)
        f_world = quat_to_mat(obs.ee_pose[3:7]) @ obs.ft[:3]
        f_insert = float(f_world @ axis)
        if robot.seated():
            seated_seen = True
        # Firmly home: seated and the connector pressed against the socket back.
        if seated_seen and abs(f_insert) >= seat_force:
            robot.set_stiffness_mode("hold")
            return True

        frac = depth / max_depth if max_depth > 0 else 0.0
        theta = 2.0 * np.pi * spiral_turns * frac
        radius = spiral_radius * frac
        offset = radius * (np.cos(theta) * u + np.sin(theta) * v)
        target_pos = base_pose7[:3] + depth * axis + offset
        dpos = target_pos - obs.ee_pose[:3]
        drot = quat_error(obs.ee_pose[3:7], base_pose7[3:7])

        if abs(f_insert) < force_max and depth < max_depth:
            depth += advance
        elif abs(f_insert) >= force_max:
            depth = max(depth - 0.5 * advance, 0.0)

        if monitor is not None:
            monitor({"depth": depth, "f_insert": f_insert, "ft": obs.ft.copy()})

        robot.step(Action.from_parts(dpos, drot, grip))
        if robot.estopped():
            return False
    robot.set_stiffness_mode("hold")
    return robot.seated()


def confirm_seat(robot: Robot, grip: float, *, hold_ticks: int = 25) -> bool:
    """Hold the seated pose and require the seat condition to stay true."""
    target = robot.get_obs(with_images=False).ee_pose.copy()
    seated_count = 0
    for _ in range(hold_ticks):
        _hold_step(robot, target, grip)
        if robot.estopped():
            return False
        seated_count = seated_count + 1 if robot.seated() else 0
    return seated_count >= max(hold_ticks // 2, 5)
