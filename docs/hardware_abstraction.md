# Hardware abstraction layer

There is exactly one `Robot` interface (`robot_core/robot.py`). Every backend implements
it, and nothing above the HAL knows which backend is running. This is what defers the B2B
(sell the software, run on a partner arm) versus full-stack (ship our own hardware)
decision: the same data, training, eval, and dashboard code runs against all three
backends.

## Backends

- `robots/mujoco_robot.py` (backend 1): MuJoCo sim. `supports_torque = True`.
- `robots/b601_robot.py` (backend 2): the custom b601 arm. `supports_torque = True`.
- `robots/offtheshelf_robot.py` (backend 3): a borrowed arm (SO-ARM, ViperX, UR5e, xArm)
  via a position shim. `supports_torque = False`.

## The contract

Observation: `images` (dict name -> HxWx3 uint8, at least `front` and `wrist`),
`joint_pos` (6), `joint_vel` (6), `gripper` ([0,1], 1 = open), `ee_pose` (7, base frame),
`ft` (6, wrist site frame), `sim_time`; with a 26-D `state` property. Action: 7-D
end-effector delta `[dx, dy, dz, drx, drz, dry, grip]`. Low-level: `set_joint_torque`,
read joint pos/vel, `read_ft`, gripper by opening, camera frames. Optional hints:
`set_stiffness_mode`, `set_payload_compensation`, `attach_payload`/`detach_payload`.

## b601 driver seam (`B601Driver`)

The real b601 fills a small driver, marked TODO at each call:

- `set_joint_torque(tau6)`: MIT-mode torque commands to the six actuators (over CAN).
- `read_joint_state()`: joint position and velocity from the encoders, at the control rate.
- `read_ee_pose()`: end-effector pose from the controller's forward kinematics.
- `read_ft()`: 6-axis wrist wrench from a real FT sensor, else a current-based or
  model-based estimate (set `supports_torque` accordingly).
- `set_gripper(opening)` / `gripper_opening()` / `grip_detected()`.
- `capture(cameras, image_hw)`: synchronized capture for `front` and `wrist`.

`B601Robot.step` converts the ee-delta to joint torque via the b601 kinematics and
Cartesian impedance (mirroring `robot_core/control` on the real URDF). `seated()` infers
the seat from the force signature (insertion-force drop after the detent) plus perception,
behind the same hook the sim uses.

## Off-the-shelf position shim (`OffTheShelfDriver`)

If the arm lacks torque mode, the impedance controller's desired wrench becomes a small
Cartesian setpoint nudge under the vendor position interface, and force is estimated from
joint current or a wrist sensor. `supports_torque = False`, so `insert_with_search` takes
the conservative path (lower bounded push and seat forces).

## Backend-agnostic proof

`tests/test_hardware_seam.py` drives a `FakeB601Driver` and a `FakeOffTheShelfDriver`
through the same `RecordingRobot` + schema + validator pipeline the sim backend uses, and
the episode round-trips. This shows the data engine depends only on the HAL contract, not
on any backend. The same applies to training and eval, which reach the sim only through the
`Robot` interface.
