"""Shared fixtures. The MuJoCo backend builds quickly; physics/control tests need no
GL context (rendering is lazy and only exercised by tests marked `gl`)."""
import pytest

from robot_core.robots.mujoco_robot import MujocoRobot


@pytest.fixture
def robot():
    r = MujocoRobot()
    yield r
    r.close()
