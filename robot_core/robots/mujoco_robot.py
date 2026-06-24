"""MuJoCo sim backend (backend 1). Implements the Robot HAL with torque-level
Cartesian impedance control. Physics and control are CPU-runnable; camera rendering
is lazy so most tests need no GL context.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from ..control.impedance import INSERT_AXIS, CartesianImpedance
from ..control.kinematics import dls_ik, site_pose
from ..robot import Robot
from ..safety import SafetyController, SafetyLimits
from ..types import GRIP_CTRL_MAX, IMAGE_KEYS, N_JOINTS, Action, Observation

DEFAULT_SCENE = (
    Path(__file__).resolve().parents[2] / "sim" / "assets" / "rebot_drive_reseat.xml"
)
SEAT_TOL = 0.012  # m, connector-to-socket distance for a seat


def _select_render_backend() -> None:
    """Pick an offscreen GL backend before the first render. macOS arm64 uses the
    default CGL path (no env var); a headless Linux GPU box uses EGL."""
    if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


class MujocoRobot(Robot):
    supports_torque = True

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        *,
        image_hw: tuple[int, int] = (128, 128),
        control_substeps: int = 10,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.image_hw = image_hw
        self.substeps = control_substeps

        # Named indices (built from the model; no scattered literals).
        self.ee_site = self.model.site("ee_site").id
        self.ft_site = self.model.site("ft_site").id
        self.connector_site = self.model.site("drive_connector").id
        self.socket_site = self.model.site("bay_socket").id
        self.mouth_site = self.model.site("bay_mouth").id
        self.grasp_site = self.model.site("drive_grasp").id
        self.grip_act = self.model.actuator("m_grip").id
        self.gr_qadr = self.model.joint("gr").qposadr[0]
        self.home_key = self.model.key("home").id
        self.drive_qadr = self.model.joint("drive_free").qposadr[0]
        self.drive_dofadr = self.model.joint("drive_free").dofadr[0]

        # Safety with torque limits read from the actuator ctrlrange.
        lim = limits or SafetyLimits()
        lim.tau_limit = np.abs(self.model.actuator_ctrlrange[:N_JOINTS, 1]).copy()
        self.safety = SafetyController(lim)

        self.impedance = CartesianImpedance(self.model, self.ee_site)
        self._mode = "hold"
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._target = np.zeros(7)
        self.reset()

    # ----- high level -------------------------------------------------------
    def reset(self) -> Observation:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        mujoco.mj_forward(self.model, self.data)
        self.safety.reset()
        self._mode = "hold"
        self._target = site_pose(self.model, self.data, self.ee_site)
        self._settle(steps=50)
        # No images here so episode reset and tests need no GL context. Callers that
        # want the first frame rendered call get_obs(with_images=True).
        return self.get_obs(with_images=False)

    def step(self, action: Action) -> Observation:
        action = self.safety.clamp_action(action)
        self._target[:3] = self.safety.clamp_pose_to_box(self._target[:3] + action.dpos)
        drot = action.drot_xyz
        if np.linalg.norm(drot) > 0:
            q = np.ascontiguousarray(self._target[3:7])
            mujoco.mju_quatIntegrate(q, np.ascontiguousarray(drot), 1.0)
            self._target[3:7] = q / np.linalg.norm(q)
        grip_ctrl = action.grip * GRIP_CTRL_MAX
        for _ in range(self.substeps):
            tau = self.impedance.torque(self.data, self._target, self._mode)
            self.data.ctrl[:N_JOINTS] = self.safety.clamp_torque(tau)
            self.data.ctrl[self.grip_act] = grip_ctrl
            mujoco.mj_step(self.model, self.data)
            if self.safety.check(self.data.qvel[:N_JOINTS], self.read_ft()[:3]) is not None:
                break
        return self.get_obs(with_images=False)

    def get_obs(self, with_images: bool = True) -> Observation:
        pos, vel = self.read_joint_state()
        gripper = float(np.clip(self.data.qpos[self.gr_qadr] / GRIP_CTRL_MAX, 0.0, 1.0))
        ee = site_pose(self.model, self.data, self.ee_site).astype(np.float32)
        ft = self.read_ft().astype(np.float32)
        images: dict[str, np.ndarray] = {}
        if with_images:
            images = {cam: self.render(cam) for cam in IMAGE_KEYS}
        return Observation(
            images=images,
            joint_pos=pos.astype(np.float32),
            joint_vel=vel.astype(np.float32),
            gripper=gripper,
            ee_pose=ee,
            ft=ft,
            sim_time=float(self.data.time),
        )

    def seated(self) -> bool:
        conn = self.data.site_xpos[self.connector_site]
        sock = self.data.site_xpos[self.socket_site]
        mouth_x = self.data.site_xpos[self.mouth_site][0]
        dist = float(np.linalg.norm(conn - sock))
        return dist < SEAT_TOL and conn[0] > mouth_x

    def set_stiffness_mode(self, mode: str) -> None:
        self.impedance.stiffness(mode)  # validates
        self._mode = mode

    # ----- low-level contract ----------------------------------------------
    def set_joint_torque(self, tau6: np.ndarray) -> None:
        self.data.ctrl[:N_JOINTS] = self.safety.clamp_torque(np.asarray(tau6))

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[:N_JOINTS].copy(), self.data.qvel[:N_JOINTS].copy()

    def read_ft(self) -> np.ndarray:
        f = self.data.sensor("wrist_force").data
        t = self.data.sensor("wrist_torque").data
        return np.concatenate([f, t]).astype(np.float64)

    def read_ft_world(self) -> np.ndarray:
        """Wrist wrench rotated into the world frame, for insertion-axis bounding."""
        rot = self.data.site_xmat[self.ft_site].reshape(3, 3)
        f = self.data.sensor("wrist_force").data
        t = self.data.sensor("wrist_torque").data
        return np.concatenate([rot @ f, rot @ t])

    def insert_axis_force(self) -> float:
        """Magnitude of the world-frame wrist force along the insertion axis."""
        return float(self.read_ft_world()[:3] @ INSERT_AXIS)

    def set_gripper(self, opening: float) -> None:
        self.data.ctrl[self.grip_act] = float(np.clip(opening, 0.0, 1.0)) * GRIP_CTRL_MAX

    def render(self, camera: str = "front", height: int | None = None, width: int | None = None):
        _select_render_backend()
        h = height or self.image_hw[0]
        w = width or self.image_hw[1]
        renderer = self._renderers.get((h, w))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._renderers[(h, w)] = renderer
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def info(self) -> dict:
        conn = self.data.site_xpos[self.connector_site]
        sock = self.data.site_xpos[self.socket_site]
        return {
            "seat_dist": float(np.linalg.norm(conn - sock)),
            "estop": self.safety.estop_reason,
            "mode": self._mode,
        }

    # ----- helpers ----------------------------------------------------------
    def _settle(self, steps: int) -> None:
        for _ in range(steps):
            tau = self.impedance.torque(self.data, self._target, "hold")
            self.data.ctrl[:N_JOINTS] = self.safety.clamp_torque(tau)
            self.data.ctrl[self.grip_act] = GRIP_CTRL_MAX
            mujoco.mj_step(self.model, self.data)

    def ik(self, target7: np.ndarray, seed: np.ndarray | None = None):
        """Damped least-squares IK to a target ee pose (sim convenience, not in the HAL)."""
        seed = self.data.qpos[:N_JOINTS].copy() if seed is None else seed
        return dls_ik(self.model, self.data, self.ee_site, target7, seed)

    def site_xpos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(name).id].copy()
