# Data schema

The native reBot episode format (`data/schema.py`) is torch-free and decoupled from
LeRobot. One episode is a compressed `.npz` of stacked per-frame arrays plus a JSON
sidecar of metadata. The LeRobot converter (`data/convert_to_lerobot.py`, M5) maps this
to a `LeRobotDataset`, so if the LeRobot API drifts only the converter changes.

## Per-frame arrays (`episode_<i>.npz`)

| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `images.front` | (T, H, W, 3) | uint8 | RGB, HWC, [0,255] (stored as `images_front`) |
| `images.wrist` | (T, H, W, 3) | uint8 | RGB, HWC, [0,255] (stored as `images_wrist`) |
| `joint_pos` | (T, 6) | float32 | rad |
| `joint_vel` | (T, 6) | float32 | rad/s |
| `gripper` | (T,) | float32 | [0,1], 1 = open |
| `ee_pose` | (T, 7) | float32 | x y z qw qx qy qz, base frame |
| `ft` | (T, 6) | float32 | wrist force xyz + torque xyz, wrist site frame |
| `sim_time` | (T,) | float32 | seconds |
| `state` | (T, 26) | float32 | `build_state(joint_pos, joint_vel, gripper, ee_pose, ft)` |
| `action` | (T, 7) | float32 | ee delta [dx dy dz drx drz dry grip] |

`build_state` (= `robot_core.types.concat_state`) is the single source of truth for the
26-D vector; the validator recomputes it from the components and rejects any mismatch.

Frames are recorded as `(s_t, a_t)`: the observation at the current state paired with the
action commanded from it, captured every `record_every` control steps (default 3, giving
about 17 fps at the 50 Hz control rate).

## Per-episode metadata (`episode_<i>.json`)

`schema_version`, `episode_index`, `seed` (the DR master seed, required for
reproducibility), `task` (natural-language string), `fps`, `image_hw`, `cameras`,
`num_frames`, `seat_success`, `insertion_time_s`, `peak_insert_force`, `search_retries`,
`controller`, and `dr_params` (the realized domain randomization).

## Dataset manifest (`dataset_manifest.json`)

`n_episodes`, `n_success`, `success_rate`, `total_frames`, `fps`, `image_hw`, `cameras`.

## Domain randomization (`sim/domain_randomization.py`)

Seeded per episode. Axes: bay pose, staged drive pose, sliding friction, drive mass and
inertia, lighting, and camera jitter. The sampler is pure numpy (deterministic per seed);
`apply` mutates the model before reset. The realized `DRParams` is recorded per episode.
Drive yaw is present in the config but disabled by default (it needs an oriented grasp).

## Generation and validation

```bash
scripts/generate_demos.sh 200 datasets/rebot_scripted 128
```

Records N randomized episodes (the scripted skill is the expert), logs the seat-success
rate, and runs `data/dataset_validator.py` as a gate. Local datasets stay small (about
1.3 MB/episode at 96x96, about 2.5 MB at 128x128); bulk 224x224 rendering for training
happens on the GPU box with `MUJOCO_GL=egl`. A 100-episode local run records 99% seat
success.
