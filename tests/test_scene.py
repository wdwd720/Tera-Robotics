"""M1: the scene compiles with the expected layout, the drive rests in the cradle,
the FT sensor reads only the gravity preload, IK reaches above the drive, every named
entity exists, and the backend steps stably."""
import numpy as np
import pytest

from robot_core.types import Action, N_JOINTS


def test_layout_nq_nv_and_joint_order(robot):
    m = robot.model
    assert m.nq == 15, m.nq
    assert m.nv == 14, m.nv
    import mujoco

    order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    assert order == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "gl",
        "gr",
        "drive_free",
    ], order
    # The drive freejoint qpos starts right after the 6 arm + 2 finger DoFs.
    assert robot.drive_qadr == 8, robot.drive_qadr
    assert robot.drive_dofadr == 8, robot.drive_dofadr


def test_named_entities_present(robot):
    m = robot.model
    for s in ["ee_site", "ft_site", "drive_connector", "drive_grasp", "bay_socket", "bay_mouth"]:
        assert m.site(s).id >= 0
    for s in ["wrist_force", "wrist_torque"]:
        assert m.sensor(s).id >= 0
    for c in ["front", "side", "wrist"]:
        assert m.camera(c).id >= 0
    for a in ["m_pan", "m_lift", "m_elbow", "m_wrist1", "m_wrist2", "m_wrist3", "m_grip"]:
        assert m.actuator(a).id >= 0


def test_drive_rests_in_cradle(robot):
    # cradle top is at z = 0.150; drive half-height 0.01305 => center ~0.163.
    drive_z = robot.data.qpos[robot.drive_qadr + 2]
    assert 0.160 < drive_z < 0.167, drive_z
    quat = robot.data.qpos[robot.drive_qadr + 3 : robot.drive_qadr + 7]
    assert abs(quat[0]) > 0.999, quat  # near-identity orientation (no tilt)
    drive_vel = robot.data.qvel[robot.drive_dofadr : robot.drive_dofadr + 6]
    assert np.linalg.norm(drive_vel) < 1e-3, np.linalg.norm(drive_vel)


def test_ft_reads_gravity_preload(robot):
    ft = robot.read_ft()
    fmag = float(np.linalg.norm(ft[:3]))
    # Only the distal gripper/finger weight, no contact spike.
    assert 1.0 < fmag < 4.0, fmag
    assert fmag < 5.0


def test_ik_reaches_above_drive(robot):
    grasp = robot.site_xpos("drive_grasp")
    ee = robot.get_obs(with_images=False).ee_pose
    target = np.concatenate([grasp + np.array([0, 0, 0.12]), ee[3:7]])
    q, iters, err = robot.ik(target)
    assert err < 1e-3, (err, iters)


def test_step_100_ticks_stable(robot):
    ee0 = robot.get_obs(with_images=False).ee_pose[:3]
    for _ in range(100):
        obs = robot.step(Action.zeros())
    assert not robot.safety.tripped, robot.safety.estop_reason
    assert np.all(np.isfinite(robot.data.qpos))
    ee1 = obs.ee_pose[:3]
    # Holding a zero-delta target: the ee should barely move.
    assert np.linalg.norm(ee1 - ee0) < 5e-3, np.linalg.norm(ee1 - ee0)


@pytest.mark.gl
def test_render_shapes(robot):
    for cam in ["front", "wrist"]:
        img = robot.render(cam, 96, 96)
        assert img.shape == (96, 96, 3)
        assert img.dtype == np.uint8
