# Architecture

## Layers

```
policies / data / eval / dashboard      (above the HAL, torch-free except policy wrappers)
            |  uses only the Robot interface, never imports mujoco
        Robot (HAL)                      robot_core/robot.py
            |
   MujocoRobot | b601Robot | OffTheShelfRobot   robot_core/robots/
            |
   torque-level control                  robot_core/control/ (sim backend)
            |
        MuJoCo physics                    sim/assets/rebot_drive_reseat.xml
```

There is exactly one `Robot` interface. The first backend is MuJoCo sim. The same data,
training, eval, and dashboard code runs against every backend with no changes.

## The contract (robot_core/types.py)

Observation: `images` (dict name -> HxWx3 uint8, at least `front` and `wrist`),
`joint_pos` (6), `joint_vel` (6), `gripper` (float in [0,1], 1 = open), `ee_pose`
(7: x y z qw qx qy qz, base frame), `ft` (6: wrist force xyz + torque xyz, wrist site
frame), `sim_time`. The `state` property concatenates proprioception into a 26-D float32
vector for LeRobot:

```
state[0:6]   joint_pos
state[6:12]  joint_vel
state[12]    gripper
state[13:20] ee_pose
state[20:26] ft
```

Action: 7-D end-effector delta `[dx, dy, dz, drx, drz, dry, grip]`. The first three are a
base-frame position delta (m). The next three are an axis-angle rotation delta (rad) in
(rx, rz, ry) slot order, so xyz = `action[[3, 5, 4]]` (`ROT_SLOT_TO_XYZ`). `grip` is a
target opening in [0,1]. The low-level controller converts the target into joint torques.

## Scene (sim/assets/rebot_drive_reseat.xml)

Self-contained MJCF, primitive geoms only, no external meshes. A 6-DoF arm of capsule
links on a 0.40 m base column, a parallel-jaw gripper (two slide fingers coupled by an
equality constraint, one position actuator), a wrist FT site with force/torque sensors, a
one-bay server chassis with sub-millimeter to ~1.5 mm clearances, a 3.5 inch drive on a
freejoint, and a staging cradle. Cameras: `front`, `side`, `wrist`. A `home` keyframe.

qpos layout (nq=15, nv=14): `j1..j6, gl, gr, drive(x y z qw qx qy qz)`. The drive freejoint
is declared after the arm so the finger DoFs precede it. The backend builds all indices
from named model access; no scattered literals.

Torque actuators use gear 1 so `ctrl` equals joint torque, mirroring MIT mode. Insertion is
along world +x; the bay opens toward the arm (-x).

## Control (robot_core/control/)

- `kinematics.py`: site pose, site Jacobian (`mj_jacSite`, arm columns), world-frame pose
  error via `mju_subQuat`, and damped least-squares IK on a scratch `MjData` copy.
- `impedance.py`: Cartesian impedance. Operational-space wrench `Kp*err - Kd*twist` mapped
  to torque by the Jacobian transpose, plus `qfrc_bias` for gravity/Coriolis. Anisotropic
  stiffness by mode: `hold`, `firm`, and `search` (firm along the insertion axis and in z,
  soft laterally for the connector search).

## Rendering

Camera rendering is lazy: physics and control need no GL context, so most tests run without
one. macOS arm64 uses the default CGL offscreen backend (no `MUJOCO_GL`); a headless Linux
GPU box uses `MUJOCO_GL=egl`.
