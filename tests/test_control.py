"""M2: impedance holds a pose, IK converges without mutating live state, the Jacobian
is correct, the action rotation slots map as specified, and the force-search primitive
runs in free space without tripping the e-stop."""
import mujoco
import numpy as np

from robot_core.control.kinematics import site_jacobian, site_pose
from robot_core.control.primitives import insert_with_search, move_to_pose
from robot_core.math_utils import quat_error
from robot_core.types import Action, action_rotation_xyz


def test_impedance_holds_pose(robot):
    ee0 = robot.get_obs(with_images=False).ee_pose.copy()
    # Command a 2 cm move and confirm it reaches, then holds with small error.
    target = ee0.copy()
    target[:3] += np.array([0.0, 0.02, -0.02])
    assert move_to_pose(robot, target, grip=1.0, pos_tol=2e-3, max_steps=600)
    for _ in range(100):
        obs = robot.step(Action.from_parts(np.zeros(3), np.zeros(3), 1.0))
    err = np.linalg.norm(obs.ee_pose[:3] - target[:3])
    assert err < 3e-3, err
    assert not robot.estopped()


def test_ik_converges_and_does_not_mutate(robot):
    grasp = robot.site_xpos("drive_grasp")
    ee = robot.get_obs(with_images=False).ee_pose
    target = np.concatenate([grasp + np.array([0, 0, 0.10]), ee[3:7]])
    before = robot.data.qpos.copy()
    q, iters, err = robot.ik(target)
    assert err < 1e-3, (err, iters)
    # IK runs on a scratch copy and must not disturb the live state.
    assert np.allclose(robot.data.qpos, before)


def test_site_jacobian_finite_difference(robot):
    m, d = robot.model, robot.data
    sid = robot.ee_site
    jac = site_jacobian(m, d, sid)  # (6, 6)
    dc = mujoco.MjData(m)
    mujoco.mj_copyData(dc, m, d)
    eps = 1e-6
    p0 = site_pose(m, dc, sid)[:3]
    for j in range(6):
        dc.qpos[:] = d.qpos
        dc.qpos[j] += eps
        mujoco.mj_forward(m, dc)
        dp = (site_pose(m, dc, sid)[:3] - p0) / eps
        assert np.allclose(dp, jac[:3, j], atol=1e-3), (j, dp, jac[:3, j])


def test_action_rotation_slot_mapping():
    # A 0.1 rad rotation about z lives in the drz slot (index 4), not xyz order.
    a = Action.from_parts(np.zeros(3), np.array([0.0, 0.0, 0.1]), grip=0.5)
    assert np.isclose(a.delta[4], 0.1)
    assert np.allclose(action_rotation_xyz(a.delta), [0.0, 0.0, 0.1])


def test_quat_error_known_z_rotation():
    cur = np.array([1.0, 0.0, 0.0, 0.0])
    des = np.array([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])  # +0.1 rad about z
    assert np.allclose(quat_error(cur, des), [0.0, 0.0, 0.1], atol=1e-6)


def test_force_search_free_space_no_estop(robot):
    base = robot.get_obs(with_images=False).ee_pose.copy()
    seated = insert_with_search(
        robot, base, max_depth=0.04, max_steps=500, force_max=15.0
    )
    assert not robot.estopped()
    assert seated is False  # free space above the bay: nothing to seat into
