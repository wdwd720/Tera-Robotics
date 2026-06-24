"""M4: native episode round trip (record -> save -> load), shapes/dtypes match the
contract, the state vector matches its components, the validator catches corruption, and
the domain-randomization sampler is deterministic. No sim required."""
import numpy as np
import pytest

from data.dataset_validator import validate_arrays
from data.schema import (
    CAMERAS,
    EpisodeBuffer,
    EpisodeMeta,
    build_state,
    load_episode,
    save_episode,
)
from robot_core.types import ACTION_DIM, Observation
from sim.domain_randomization import DRConfig, sample

HW = (16, 16)


def _make_obs(rng) -> Observation:
    return Observation(
        images={c: rng.integers(0, 256, (HW[0], HW[1], 3), dtype=np.uint8) for c in CAMERAS},
        joint_pos=rng.standard_normal(6).astype(np.float32),
        joint_vel=rng.standard_normal(6).astype(np.float32),
        gripper=float(rng.uniform(0, 1)),
        ee_pose=rng.standard_normal(7).astype(np.float32),
        ft=rng.standard_normal(6).astype(np.float32),
        sim_time=float(rng.uniform()),
    )


def _build(rng, n=5) -> EpisodeBuffer:
    buf = EpisodeBuffer(CAMERAS)
    for _ in range(n):
        buf.add_frame(_make_obs(rng), rng.standard_normal(ACTION_DIM).astype(np.float32))
    return buf


def test_record_replay_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    buf = _build(rng)
    arrays = buf.arrays()
    meta = EpisodeMeta(episode_index=0, seed=0, task="reseat", image_hw=HW, num_frames=len(buf))
    save_episode(tmp_path, meta, arrays)

    loaded, lmeta = load_episode(tmp_path / "episode_000000.npz")
    for k in arrays:
        assert np.array_equal(loaded[k], arrays[k]), f"mismatch in {k}"
    validate_arrays(loaded, image_hw=HW)
    assert lmeta["task"] == "reseat" and lmeta["seed"] == 0


def test_state_equals_components():
    rng = np.random.default_rng(1)
    arr = _build(rng, n=3).arrays()
    for t in range(3):
        s = build_state(
            arr["joint_pos"][t],
            arr["joint_vel"][t],
            arr["gripper"][t],
            arr["ee_pose"][t],
            arr["ft"][t],
        )
        assert np.allclose(arr["state"][t], s)


def test_validator_catches_corruption():
    rng = np.random.default_rng(2)
    arr = _build(rng).arrays()
    validate_arrays(arr, image_hw=HW)  # good data passes

    bad_dtype = dict(arr)
    bad_dtype["images.front"] = arr["images.front"].astype(np.float32)
    with pytest.raises(AssertionError):
        validate_arrays(bad_dtype, image_hw=HW)

    bad_state = dict(arr)
    bad_state["state"] = arr["state"] + 1.0
    with pytest.raises(AssertionError):
        validate_arrays(bad_state, image_hw=HW)


def test_dr_sampler_deterministic():
    cfg = DRConfig()
    assert sample(cfg, 42).to_dict() == sample(cfg, 42).to_dict()
    assert sample(cfg, 1).to_dict() != sample(cfg, 2).to_dict()
