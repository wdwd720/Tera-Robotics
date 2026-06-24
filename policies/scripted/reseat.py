"""The scripted drive-reseat skill: the baseline, the data generator, and the permanent
low-level controller. From home it moves to the staging drive, grasps, lifts, aligns at
the bay mouth, inserts with force search, confirms the seat, and retreats.

Backend-agnostic: it drives the Robot HAL and reads task poses from a Perception seam
(sim ground truth now, a real detector later).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.pose_estimation import Perception, SimGroundTruthPerception
from robot_core.control.primitives import (
    confirm_seat,
    grasp,
    hold,
    insert_with_search,
    lift,
    move_to_pose,
    open_gripper,
    retreat,
)
from robot_core.robot import Robot

GRIP_HOLD = 0.0  # fully-closed command: the force limit gives a firm, rigid grip so the
# drive does not slip or pitch in the jaws during contact-rich insertion


@dataclass
class ReseatResult:
    success: bool
    seated: bool
    grasped: bool
    estop: str | None
    peak_insert_force: float
    insert_force_trace: list[float] = field(default_factory=list)
    timeline: list[tuple[str, float]] = field(default_factory=list)

    def print_report(self) -> None:
        print("phase timeline (sim_time s):")
        for name, t in self.timeline:
            print(f"  {name:14s} {t:7.3f}")
        print(f"grasped={self.grasped} seated={self.seated} success={self.success}")
        print(f"peak insert force: {self.peak_insert_force:.2f} N  estop={self.estop}")
        tail = self.insert_force_trace[-8:]
        print("insert-axis force near seat:", [round(f, 2) for f in tail])


def run_reseat(
    robot: Robot,
    perception: Perception | None = None,
    *,
    do_reset: bool = True,
    verbose: bool = False,
) -> ReseatResult:
    perception = perception or SimGroundTruthPerception(robot)
    if do_reset:
        robot.reset()
    down = robot.get_obs(with_images=False).ee_pose[3:7].copy()  # tool-down orientation

    timeline: list[tuple[str, float]] = []

    def mark(name: str) -> None:
        timeline.append((name, robot.get_obs(with_images=False).sim_time))

    def target(pos: np.ndarray) -> np.ndarray:
        return np.concatenate([pos, down])

    def ft_force_mag() -> float:
        # Frame-invariant: the distal weight magnitude, independent of tool tilt.
        return float(np.linalg.norm(robot.get_obs(with_images=False).ft[:3]))

    mark("start")
    baseline_weight = ft_force_mag()  # distal preload, empty hand

    # 1, 2: approach above the drive, then descend to the grasp point.
    grasp_pt = perception.grasp_point()
    ok = move_to_pose(robot, target(grasp_pt + np.array([0, 0, 0.10])), grip=1.0)
    mark("above_grasp")
    ok = ok and move_to_pose(robot, target(grasp_pt), grip=1.0, pos_tol=2e-3)
    mark("at_grasp")

    # 3: grasp, then declare a firm grasp (rigid hold) for the carry and insert.
    grasped = grasp(robot, close_to=GRIP_HOLD)
    if grasped:
        robot.attach_payload()
    mark("grasped")

    # 4: lift gently off the cradle, measure and feed forward the payload weight (the
    # drive is a separate body, not in the arm's gravity comp), then lift well clear of
    # the bay support rail before translating to the mouth.
    ok = ok and lift(robot, 0.05, grip=GRIP_HOLD, mode="hold", speed=0.05, pos_tol=0.02)
    hold(robot, 60, GRIP_HOLD)  # let the drive settle into a steady hang before measuring
    payload = max(0.0, ft_force_mag() - baseline_weight)
    robot.set_payload_compensation(payload)
    ok = ok and lift(robot, 0.14, grip=GRIP_HOLD, mode="firm", speed=0.10)
    mark("lifted")

    # 5: align the connector just outside the bay mouth.
    mouth = perception.bay_mouth()
    socket = perception.bay_socket()
    ee_pos = robot.get_obs(with_images=False).ee_pose[:3]
    conn_off = perception.connector_offset(ee_pos)  # connector - ee, world frame
    pre_connector = np.array([mouth[0] - 0.02, socket[1], socket[2]])
    ok = ok and move_to_pose(
        robot, target(pre_connector - conn_off), grip=GRIP_HOLD, mode="firm",
        speed=0.10, pos_tol=2e-3,
    )
    mark("at_mouth")

    # 6: insert with force search along +x.
    base = robot.get_obs(with_images=False).ee_pose.copy()
    trace: list[float] = []
    peak = 0.0

    def mon(info: dict) -> None:
        nonlocal peak
        trace.append(info["f_insert"])
        peak = max(peak, abs(info["f_insert"]))

    seated = insert_with_search(
        robot, base, max_depth=0.09, grip=GRIP_HOLD, force_max=15.0,
        spiral_radius=0.0025, monitor=mon,
    )
    mark("inserted")

    # 7: confirm the seat holds.
    confirmed = confirm_seat(robot, grip=GRIP_HOLD)
    mark("confirmed")

    # 8: release and retreat.
    robot.set_payload_compensation(0.0)
    robot.detach_payload()
    open_gripper(robot)
    retreat(robot, 0.10, grip=1.0)
    mark("retreated")

    result = ReseatResult(
        success=bool(seated and confirmed and not robot.estopped()),
        seated=robot.seated(),
        grasped=grasped,
        estop=robot.info().get("estop") if hasattr(robot, "info") else None,
        peak_insert_force=peak,
        insert_force_trace=trace,
        timeline=timeline,
    )
    if verbose:
        result.print_report()
    return result


if __name__ == "__main__":
    from robot_core.robots.mujoco_robot import MujocoRobot

    r = MujocoRobot()
    run_reseat(r, verbose=True)
