"""Backend 3: an adapter for a borrowed off-the-shelf arm (SO-ARM, ViperX, UR5e, xArm).

If the arm lacks torque mode it runs a position-control shim: the impedance controller's
desired wrench becomes a small Cartesian setpoint nudge under the vendor's position
interface, with force estimated from joint current or a wrist sensor. supports_torque is
False so the force-search primitive takes the conservative path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..robot import Robot
from ..safety import SafetyController, SafetyLimits
from ..types import IMAGE_KEYS, N_JOINTS, Action, Observation


class OffTheShelfDriver(ABC):
    """Position-interface driver contract for a borrowed arm."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def set_joint_position(self, q6: np.ndarray) -> None:
        """TODO: command joint positions via the vendor SDK."""

    @abstractmethod
    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def read_ee_pose(self) -> np.ndarray: ...

    @abstractmethod
    def estimate_ft(self) -> np.ndarray:
        """TODO: estimate the wrist wrench from joint current or a wrist sensor."""

    @abstractmethod
    def set_gripper(self, opening: float) -> None: ...

    @abstractmethod
    def gripper_opening(self) -> float: ...

    @abstractmethod
    def capture(self, cameras: tuple[str, ...], image_hw: tuple[int, int]) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def tick(self) -> None: ...


class OffTheShelfRobot(Robot):
    supports_torque = False  # position shim -> force-search uses the conservative path

    def __init__(
        self,
        driver: OffTheShelfDriver,
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
        # Position shim: nudge the ee setpoint by the (clamped) delta, then IK to joint
        # positions. TODO: real IK on the vendor URDF. The stub holds position so the
        # contract can be exercised with the fake driver.
        q_cmd = self._shim_to_joint_positions(action)
        self.driver.set_joint_position(q_cmd)
        self.driver.set_gripper(action.grip)
        self.driver.tick()
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
            ft=np.asarray(self.driver.estimate_ft(), np.float32),
            sim_time=0.0,
        )

    def seated(self) -> bool:
        # TODO: seat inference from the estimated force signature plus perception.
        return False

    def estopped(self) -> bool:
        return self.safety.tripped

    def set_joint_torque(self, tau6: np.ndarray) -> None:
        # No torque interface: ignore (the shim uses position). Present for the contract.
        raise NotImplementedError("off-the-shelf arm has no torque interface; use the shim")

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.driver.read_joint_state()

    def read_ft(self) -> np.ndarray:
        return np.asarray(self.driver.estimate_ft(), np.float64)

    def set_gripper(self, opening: float) -> None:
        self.driver.set_gripper(float(np.clip(opening, 0.0, 1.0)))

    def render(self, camera: str = "front", height: int | None = None, width: int | None = None):
        hw = (height or self.image_hw[0], width or self.image_hw[1])
        return self.driver.capture((camera,), hw)[camera]

    def _shim_to_joint_positions(self, action: Action) -> np.ndarray:
        # TODO: integrate the ee-delta into a Cartesian setpoint and IK to joints.
        pos, _ = self.driver.read_joint_state()
        return np.asarray(pos)


class FakeOffTheShelfDriver(OffTheShelfDriver):
    """Synthetic position-interface driver for exercising the contract without hardware."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._gripper = 1.0
        self._q = np.zeros(N_JOINTS, np.float32)

    def reset(self) -> None:
        self._gripper = 1.0
        self._q = np.zeros(N_JOINTS, np.float32)

    def set_joint_position(self, q6: np.ndarray) -> None:
        self._q = np.asarray(q6, np.float32)

    def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), np.zeros(N_JOINTS, np.float32)

    def read_ee_pose(self) -> np.ndarray:
        return np.array([0.45, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0], np.float32)

    def estimate_ft(self) -> np.ndarray:
        return np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0], np.float32)

    def set_gripper(self, opening: float) -> None:
        self._gripper = float(np.clip(opening, 0.0, 1.0))

    def gripper_opening(self) -> float:
        return self._gripper

    def capture(self, cameras, image_hw) -> dict[str, np.ndarray]:
        h, w = image_hw
        return {c: self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for c in cameras}

    def tick(self) -> None:
        pass
