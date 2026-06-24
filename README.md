# reBot

reBot is the learning, control, data, and autonomy stack for datacenter maintenance
robotics. The first task is autonomous hard-drive reseating: a 6-DoF torque-controlled
arm grasps a 3.5 inch drive and seats it into a bay connector under force feedback with
sub-millimeter compliance.

## Prime directive: one hardware abstraction layer

There is exactly one `Robot` interface (`robot_core/robot.py`). Every backend implements
it: MuJoCo sim now (`robot_core/robots/mujoco_robot.py`), the custom b601 arm later, and a
borrowed off-the-shelf arm as a third. Nothing above the HAL (data engine, training, eval,
policies, dashboard) knows or cares which backend is running. The same demo runs in sim
now and on the b601 later with zero changes above the HAL.

## Architecture

- `robot_core/` types, safety, the `Robot` HAL, backends, and torque-level control.
- `sim/` the self-contained MJCF scene and domain randomization.
- `data/` episode schema, scripted recording, replay, validation, LeRobot conversion.
- `policies/` the scripted reseat skill, the train wrapper, and closed-loop inference.
- `eval/` task suite and metrics.
- `perception/`, `teleop/`, `app/` perception seam, teleop, and dashboard.

The core (`robot_core`, `sim`, `data`, `eval`) is torch-free and CPU-runnable. Torch and
LeRobot live only behind the `[train]`/`[infer]` extras in a separate Python 3.11 venv.

## Setup

```bash
bash scripts/setup.sh
. .venv/bin/activate
pytest -q
```

`scripts/setup.sh` builds the torch-free core venv (`.venv`) and, when Python 3.11 is
available, a separate `.venv-train` for the ML extras. Training itself runs on a cloud GPU
box, not locally.

## Build order

Harness, then the scripted skill, then the data engine, then eval, then training. There is
a working drive-reseat demo (`policies/scripted/reseat.py`) before any neural network
exists.
