"""M7: a fake-driver backend passes the same record -> schema -> validate round trip the
sim backend does, proving the data engine depends only on the HAL contract."""
import numpy as np

from data.dataset_validator import validate_arrays
from data.record_scripted import RecordingRobot
from data.schema import EpisodeMeta, load_episode, save_episode
from robot_core.robots.b601_robot import B601Robot, FakeB601Driver
from robot_core.robots.offtheshelf_robot import FakeOffTheShelfDriver, OffTheShelfRobot
from robot_core.types import Action

HW = (24, 24)


def _drive_record_validate(robot, tmp_path):
    rec = RecordingRobot(robot, record_every=1)
    rec.reset()
    for _ in range(5):
        rec.step(Action.zeros())
    arrays = rec.buffer.arrays()
    validate_arrays(arrays, image_hw=HW)  # same validator the sim backend's data uses
    meta = EpisodeMeta(episode_index=0, seed=0, task="t", image_hw=HW, num_frames=len(rec.buffer))
    save_episode(tmp_path, meta, arrays)
    loaded, _ = load_episode(tmp_path / "episode_000000.npz")
    for k in arrays:
        assert np.array_equal(loaded[k], arrays[k]), k


def test_b601_fake_driver_roundtrip(tmp_path):
    robot = B601Robot(FakeB601Driver(), image_hw=HW)
    assert robot.supports_torque is True
    obs = robot.reset()
    obs.validate()
    _drive_record_validate(robot, tmp_path)


def test_offtheshelf_fake_driver_roundtrip(tmp_path):
    robot = OffTheShelfRobot(FakeOffTheShelfDriver(), image_hw=HW)
    assert robot.supports_torque is False
    robot.reset()
    _drive_record_validate(robot, tmp_path)
