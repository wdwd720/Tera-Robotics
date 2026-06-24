"""Domain randomization over the axes that vary in a real datacenter. Sampling is pure
numpy and seeded (unit-testable without mujoco); apply() lazily imports mujoco and
mutates the model. The realized DRParams are recorded in every episode for reproducibility.

The perception seam reports ground-truth bay and drive poses, so pose randomization is
handled automatically by the scripted skill. Friction, mass, lighting, and camera jitter
exercise the physics and the vision the learned policy will face.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Geoms whose sliding friction is randomized (the contact-rich surfaces).
_FRICTION_GEOMS = (
    "g_drive",
    "g_connector",
    "g_bay_bottom",
    "g_bay_top",
    "g_bay_yp",
    "g_bay_ym",
    "g_bay_back",
    "g_bay_rail",
    "g_finger_left",
    "g_finger_right",
)
_CAMERAS = ("front", "side")


@dataclass
class DRConfig:
    bay_pos: float = 0.01  # +/- m per axis on the chassis
    drive_pos: float = 0.02  # +/- m in x, y for the staged drive
    drive_yaw: float = 0.0  # +/- rad (disabled by default; needs an oriented grasp)
    friction: tuple[float, float] = (0.7, 1.3)  # scale on sliding friction
    drive_mass: tuple[float, float] = (0.7, 1.4)  # scale on drive mass + inertia
    light_pos: float = 0.3  # +/- m on light positions
    light_diffuse: tuple[float, float] = (0.6, 1.0)  # scale on light diffuse
    cam_pos: float = 0.02  # +/- m camera jitter


@dataclass
class DRParams:
    bay_pos_delta: list
    drive_pos_delta: list
    drive_yaw: float
    friction_scale: float
    mass_scale: float
    light_pos_delta: list
    light_diffuse: float
    cam_pos_delta: list
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def sample(config: DRConfig, seed: int) -> DRParams:
    """Sample a realized randomization from a per-episode seed. Pure numpy."""
    rng = np.random.default_rng(seed)
    return DRParams(
        bay_pos_delta=rng.uniform(-config.bay_pos, config.bay_pos, 3).tolist(),
        drive_pos_delta=rng.uniform(-config.drive_pos, config.drive_pos, 2).tolist(),
        drive_yaw=float(rng.uniform(-config.drive_yaw, config.drive_yaw)),
        friction_scale=float(rng.uniform(*config.friction)),
        mass_scale=float(rng.uniform(*config.drive_mass)),
        light_pos_delta=rng.uniform(-config.light_pos, config.light_pos, 3).tolist(),
        light_diffuse=float(rng.uniform(*config.light_diffuse)),
        cam_pos_delta=rng.uniform(-config.cam_pos, config.cam_pos, 3).tolist(),
        seed=seed,
    )


def apply(robot, params: DRParams) -> None:
    """Apply a realized randomization to the backend's model. Call before reset().

    Mutates model.body_pos (bay), the home keyframe drive pose, geom friction, drive
    mass/inertia, lights, and cameras. Lazily imports mujoco so the sampler stays
    importable on a torch-/mujoco-free box.
    """
    import mujoco  # noqa: F401 (kept local to keep the sampler backend-free)

    m = robot.model

    # Bay pose: the chassis is a fixed body, so body_pos is its world pose. Its walls,
    # rail, and sites move with it.
    cid = m.body("chassis").id
    m.body_pos[cid] = m.body_pos[cid] + np.asarray(params.bay_pos_delta)

    # Staged drive pose in the home keyframe (x, y; z keeps it resting on the cradle).
    key, qadr = robot.home_key, robot.drive_qadr
    m.key_qpos[key, qadr : qadr + 2] = m.key_qpos[key, qadr : qadr + 2] + np.asarray(
        params.drive_pos_delta
    )

    # Friction scale on the contact-rich surfaces.
    for name in _FRICTION_GEOMS:
        gid = m.geom(name).id
        m.geom_friction[gid, 0] = m.geom_friction[gid, 0] * params.friction_scale

    # Drive mass and inertia (scale together to stay consistent).
    did = m.body("drive").id
    m.body_mass[did] = m.body_mass[did] * params.mass_scale
    m.body_inertia[did] = m.body_inertia[did] * params.mass_scale

    # Lighting.
    for li in range(m.nlight):
        m.light_pos[li] = m.light_pos[li] + np.asarray(params.light_pos_delta)
        m.light_diffuse[li] = np.clip(m.light_diffuse[li] * params.light_diffuse, 0.0, 1.0)

    # Camera jitter.
    for cam in _CAMERAS:
        cam_id = m.camera(cam).id
        m.cam_pos[cam_id] = m.cam_pos[cam_id] + np.asarray(params.cam_pos_delta)
