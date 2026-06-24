"""M1: SafetyController clamps and e-stops fire correctly. Pure, no sim."""
import numpy as np

from robot_core.safety import SafetyController, SafetyLimits
from robot_core.types import Action


def make() -> SafetyController:
    lim = SafetyLimits(tau_limit=np.full(6, 50.0))
    return SafetyController(lim)


def test_clamp_action_position_norm():
    s = make()
    a = Action.from_parts(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0.5)
    out = s.clamp_action(a)
    assert np.isclose(np.linalg.norm(out.dpos), s.limits.max_dpos)


def test_clamp_action_rotation_norm():
    s = make()
    a = Action.from_parts(np.zeros(3), np.array([0.0, 0.0, 2.0]), 0.5)
    out = s.clamp_action(a)
    assert np.isclose(np.linalg.norm(out.drot_xyz), s.limits.max_drot)


def test_clamp_action_grip_clipped():
    s = make()
    assert s.clamp_action(Action.from_parts(np.zeros(3), np.zeros(3), 5.0)).grip == 1.0
    assert s.clamp_action(Action.from_parts(np.zeros(3), np.zeros(3), -5.0)).grip == 0.0


def test_clamp_action_passthrough_small():
    s = make()
    dpos = np.array([0.005, 0.0, -0.003])
    out = s.clamp_action(Action.from_parts(dpos, np.zeros(3), 0.3))
    assert np.allclose(out.dpos, dpos)


def test_clamp_pose_to_box():
    s = make()
    p = s.clamp_pose_to_box(np.array([2.0, -2.0, 2.0]))
    assert np.allclose(p, np.array(s.limits.workspace_max[:1] + s.limits.workspace_min[1:2] + s.limits.workspace_max[2:3]))


def test_clamp_torque_saturates():
    s = make()
    tau = s.clamp_torque(np.array([100.0, -100.0, 10.0, 0.0, 0.0, 0.0]))
    assert tau[0] == 50.0 and tau[1] == -50.0 and tau[2] == 10.0


def test_estop_on_velocity():
    s = make()
    reason = s.check(np.array([0, 0, 10.0, 0, 0, 0]), np.zeros(3))
    assert reason is not None and s.tripped


def test_estop_on_force():
    s = make()
    reason = s.check(np.zeros(6), np.array([100.0, 0, 0]))
    assert reason is not None and s.tripped


def test_no_estop_nominal():
    s = make()
    assert s.check(np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0]), np.array([2.0, 0.0, 1.0])) is None
    assert not s.tripped


def test_insert_force_exceeded():
    s = make()
    assert s.insert_force_exceeded(30.0)
    assert not s.insert_force_exceeded(10.0)
