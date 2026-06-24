"""Backend 2: the custom b601 arm. Implements the same Robot HAL as the sim backend.

The only thing that differs from sim is the driver seam: a small set of calls the real
b601 must fill (MIT-mode torque, encoder reads, wrist FT, gripper, synchronized cameras).
Everything above the HAL (data engine, training, eval, dashboard) runs unchanged.

A FakeB601Driver is provided so the contract can be exercised end to end without hardware
(it proves the data engine is backend-agnostic). Real driver calls are marked TODO at the
exact seam.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..robot import Robot
from ..safety import SafetyController, SafetyLimits
from ..types import IMAGE_KEYS, N_JOINTS, Action, Observation


class B601Driver(ABC):
    """The driver contract the real b601 must implement. See docs/hardware_abstraction.md."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def set_joint_torque(self, tau6: np.ndarray) -> None:
        """TODO(b601): send MIT-mode torque commands to the six actuators over CAN."""

    @abstractmethod
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """TODO(b601): read joint position and velocity from the encoders."""

    @abstractmethod
    def read_ee_pose(self) -> np.ndarray:
        """TODO(b601): end-effector pose (7,) from the controller's forward kinematics."""

    @abstractmethod
    def read_ft(self) -> np.ndarray:
        """TODO(b601): 6-axis wrist wrench from a real FT sensor, else a current-based or
        model-based estimate (set supports_torque accordingly)."""

    @abstractmethod
    def set_gripper(self, opening: float) -> None:
        """TODO(b601): command the gripper opening in [0,1]."""

    @abstractmethod
    def gripper_opening(self) -> float:
        """TODO(b601): read the current gripper opening in [0,1]."""

    @abstractmethod
    def grip_detected(self) -> bool:
        """TODO(b601): grip-detected signal (force / current threshold)."""

    @abstractmethod
    def capture(self, cameras: tuple[str, ...], image_hw: tuple[int, int]) -> dict[str, np.ndarray]:
        """TODO(b601): synchronized camera capture for the named cameras."""

    @abstractmethod
    def tick(self) -> None:
        """Advance one control step (block until the next control tick on hardware)."""


class B601Robot(Robot):
    supports_torque = True

    def __init__(
        self,
        driver: B601Driver,
        *,
        image_hw: tuple[int, int] = (128, 128),
        cameras: tuple[str, ...] = IMAGE_KEYS,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.driver = driver
        self.image_hw = image_hw
        self.cameras = cameras
        self.safety = SafetyController(limits or SafetyLimits())

    def reset(self) -> Observation:
        self.driver.reset()
        self.safety.reset()
        return self.get_obs(with_images=False)

    def step(self, action: Action) -> Observation:
        action = self.safety.clamp_action(action)
        # TODO(b601): convert the ee-delta to joint torque using the b601 URDF kinematics
        # and Cartesian impedance (mirror robot_core.control on the real arm). The stub
        # commands zero torque so the contract can be exercised with the fake driver.
        tau = self._controller_torque(action)
        self.driver.set_joint_torque(self.safety.clamp_torque(tau))
        self.driver.set_gripper(action.grip)
        self.driver.tick()
        pos, vel = self.driver.read_joint_state()
        self.safety.check(vel, self.driver.read_ft()[:3])
        return self.get_obs(with_images=False)

    def get_obs(self, with_images: bool = True) -> Observation:
        pos, vel = self.driver.read_joint_state()
        images = self.driver.capture(self.cameras, self.image_hw) if with_images else {}
        return Observation(
            images=images,
            joint_pos=np.asarray(pos, np.float32),
            joint_vel=np.asarray(vel, np.float32),
            gripper=float(self.driver.gripper_opening()),
            ee_pose=np.asarray(self.driver.read_ee_pose(), np.float32),
            ft=np.asarray(self.driver.read_ft(), np.float32),
            sim_time=0.0,
        )

    def seated(self) -> bool:
        # TODO(b601): infer seat from the force signature (insertion-force drop after the
        # detent) plus perception. Same hook as sim so call sites are shared.
        return self.driver.grip_detected() and bool(np.linalg.norm(self.driver.read_ft()[:3]) > 0)

    def estopped(self) -> bool:
        return self.safety.tripped

    def set_joint_torque(self, tau6: np.ndarray) -> None:
        self.driver.set_joint_torque(self.safety.clamp_torque(np.asarray(tau6)))

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.driver.read_joint_state()

    def read_ft(self) -> np.ndarray:
        return np.asarray(self.driver.read_ft(), np.float64)

    def set_gripper(self, opening: float) -> None:
        self.driver.set_gripper(float(np.clip(opening, 0.0, 1.0)))

    def render(self, camera: str = "front", height: int | None = None, width: int | None = None):
        hw = (height or self.image_hw[0], width or self.image_hw[1])
        return self.driver.capture((camera,), hw)[camera]

    def _controller_torque(self, action: Action) -> np.ndarray:
        # TODO(b601): real Cartesian impedance on the b601 kinematics.
        return np.zeros(N_JOINTS)


class FakeB601Driver(B601Driver):
    """A synthetic driver that returns contract-valid data, so the HAL and the data engine
    can be exercised without hardware. Not used on the real arm."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._gripper = 1.0

    def reset(self) -> None:
        self._t = 0
        self._gripper = 1.0

    def set_joint_torque(self, tau6: np.ndarray) -> None:
        self._tau = np.asarray(tau6)

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos = 0.1 * np.sin(0.01 * self._t + np.arange(N_JOINTS)).astype(np.float32)
        vel = np.zeros(N_JOINTS, np.float32)
        return pos, vel

    def read_ee_pose(self) -> np.ndarray:
        return np.array([0.45, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0], np.float32)

    def read_ft(self) -> np.ndarray:
        return np.array([0.0, 0.0, -2.3, 0.0, 0.0, 0.0], np.float32)

    def set_gripper(self, opening: float) -> None:
        self._gripper = float(np.clip(opening, 0.0, 1.0))

    def gripper_opening(self) -> float:
        return self._gripper

    def grip_detected(self) -> bool:
        return self._gripper < 0.5

    def capture(self, cameras, image_hw) -> dict[str, np.ndarray]:
        h, w = image_hw
        return {c: self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for c in cameras}

    def tick(self) -> None:
        self._t += 1
