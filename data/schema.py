"""Native reBot episode format. Torch-free, numpy only.

One episode is a compressed .npz of stacked per-frame arrays plus a JSON sidecar of
metadata. This format is decoupled from LeRobot: the converter (M5) maps it to a
LeRobotDataset, so if the LeRobot API drifts only the converter changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from robot_core.types import ACTION_DIM, STATE_DIM, concat_state

SCHEMA_VERSION = "rebot-ep-1"
DEFAULT_FPS = 20
DEFAULT_IMAGE_HW = (128, 128)
CAMERAS = ("front", "wrist")

# build_state is the single source of truth for the 26-D vector (re-exported).
build_state = concat_state

# Per-frame vector arrays: name -> (per-frame shape, dtype). Images are handled
# separately because their shape depends on the dataset image resolution.
VECTOR_SPEC: dict[str, tuple[tuple[int, ...], type]] = {
    "joint_pos": ((6,), np.float32),
    "joint_vel": ((6,), np.float32),
    "gripper": ((), np.float32),
    "ee_pose": ((7,), np.float32),
    "ft": ((6,), np.float32),
    "sim_time": ((), np.float32),
    "state": ((STATE_DIM,), np.float32),
    "action": ((ACTION_DIM,), np.float32),
}


@dataclass
class EpisodeMeta:
    episode_index: int
    seed: int
    task: str
    fps: int = DEFAULT_FPS
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW
    cameras: tuple[str, ...] = CAMERAS
    num_frames: int = 0
    seat_success: bool = False
    insertion_time_s: float | None = None
    peak_insert_force: float | None = None
    search_retries: int = 0
    controller: str = "scripted_force_search"
    dr_params: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodeBuffer:
    """Accumulates per-frame observations and actions, then stacks to arrays."""

    def __init__(self, cameras: tuple[str, ...] = CAMERAS) -> None:
        self.cameras = cameras
        self.images: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
        self.vectors: dict[str, list[np.ndarray]] = {k: [] for k in VECTOR_SPEC}

    def __len__(self) -> int:
        return len(self.vectors["action"])

    def add_frame(self, obs, action: np.ndarray) -> None:
        for c in self.cameras:
            self.images[c].append(np.asarray(obs.images[c], dtype=np.uint8))
        self.vectors["joint_pos"].append(np.asarray(obs.joint_pos, np.float32))
        self.vectors["joint_vel"].append(np.asarray(obs.joint_vel, np.float32))
        self.vectors["gripper"].append(np.float32(obs.gripper))
        self.vectors["ee_pose"].append(np.asarray(obs.ee_pose, np.float32))
        self.vectors["ft"].append(np.asarray(obs.ft, np.float32))
        self.vectors["sim_time"].append(np.float32(obs.sim_time))
        self.vectors["state"].append(np.asarray(obs.state, np.float32))
        self.vectors["action"].append(np.asarray(action, np.float32).reshape(ACTION_DIM))

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for c in self.cameras:
            out[f"images.{c}"] = np.stack(self.images[c]).astype(np.uint8)
        for k in VECTOR_SPEC:
            out[k] = np.stack(self.vectors[k]).astype(np.float32)
        return out


def episode_paths(out_dir: str | Path, index: int) -> tuple[Path, Path]:
    out = Path(out_dir)
    stem = f"episode_{index:06d}"
    return out / f"{stem}.npz", out / f"{stem}.json"


def save_episode(out_dir: str | Path, meta: EpisodeMeta, arrays: dict[str, np.ndarray]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = episode_paths(out, meta.episode_index)
    # npz keys cannot contain dots cleanly across tools; map images.front -> images_front.
    save_kwargs = {k.replace(".", "_"): v for k, v in arrays.items()}
    np.savez_compressed(npz_path, **save_kwargs)
    json_path.write_text(json.dumps(meta.to_dict(), indent=2))
    return npz_path


def load_episode(npz_path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        arrays = {k.replace("images_", "images."): data[k] for k in data.files}
    json_path = npz_path.with_suffix(".json")
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    return arrays, meta


def write_manifest(out_dir: str | Path, manifest: dict) -> Path:
    path = Path(out_dir) / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
