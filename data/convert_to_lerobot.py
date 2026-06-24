"""Convert native reBot episodes (.npz + JSON) into a LeRobotDataset (v0.5 line, dataset
format v3.0). This is the only file under data/ that touches lerobot/torch, and the import
is guarded inside functions so the torch-free core stays importable without the extra.

API notes (LeRobot v0.5 / dataset v3.0), verified from the release docs and source; the
API version-drifts, so the [VERIFY] points are re-checked at run time on the GPU box:
- LeRobotDataset.create(repo_id, fps, features, root=, robot_type=, use_videos=)
- ds.add_frame(frame_dict)  # frame must include a 'task' key
- ds.save_episode()         # once per episode
- ds.finalize()             # mandatory; without it the dataset will not load
Sources: huggingface.co/blog/lerobot-release-v050, docs/lerobot-dataset-v3,
github.com/huggingface/lerobot/.../lerobot_dataset.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.schema import CAMERAS, load_episode
from robot_core.types import ACTION_DIM, STATE_DIM


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # [VERIFY] import path
    except Exception as e:  # pragma: no cover - exercised only with the extra installed
        raise RuntimeError(
            "convert_to_lerobot needs the [train]/[infer] extra (lerobot + torch), "
            "installed in the Python 3.11 .venv-train or on the GPU box. The reBot core "
            "(robot_core/sim/data/eval) is torch-free."
        ) from e
    return LeRobotDataset


def build_features(image_hw: tuple[int, int], cameras=CAMERAS, use_videos: bool = True) -> dict:
    """LeRobot feature spec. Images are HWC; dtype 'video' encodes to mp4, 'image' to png."""
    h, w = image_hw
    img_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
        }
    return features


def convert(
    src_dir: str | Path,
    repo_id: str,
    *,
    root: str | Path,
    use_videos: bool = True,
    robot_type: str = "rebot_6dof",
    push: bool = False,
) -> Path:
    """Convert all native episodes under src_dir into a LeRobotDataset at root."""
    LeRobotDataset = _require_lerobot()
    src = Path(src_dir)
    npzs = sorted(src.glob("episode_*.npz"))
    if not npzs:
        raise FileNotFoundError(f"no episodes under {src}")

    _, first_meta = load_episode(npzs[0])
    fps = int(first_meta["fps"])
    image_hw = tuple(first_meta["image_hw"])
    cameras = tuple(first_meta.get("cameras", CAMERAS))
    features = build_features(image_hw, cameras, use_videos)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(root),
        robot_type=robot_type,
        use_videos=use_videos,
    )

    for npz in npzs:
        arrays, meta = load_episode(npz)
        task = meta.get("task", "Reseat the hard drive into the bay.")
        n = arrays["action"].shape[0]
        for t in range(n):
            frame = {
                "observation.state": arrays["state"][t].astype(np.float32),
                "action": arrays["action"][t].astype(np.float32),
                "task": task,  # [VERIFY] task-in-frame vs separate kwarg in installed version
            }
            for cam in cameras:
                frame[f"observation.images.{cam}"] = arrays[f"images.{cam}"][t]  # HWC uint8
            ds.add_frame(frame)
        ds.save_episode()

    ds.finalize()  # mandatory: writes parquet footers / metadata
    if push:
        ds.push_to_hub()
    return Path(root)


def verify(repo_id: str, root: str | Path) -> dict:
    """Reload the produced dataset and check the first frame matches the contract."""
    LeRobotDataset = _require_lerobot()
    ds = LeRobotDataset(repo_id, root=str(root))
    sample = ds[0]
    info = {
        "num_frames": len(ds),
        "fps": ds.meta.fps,
        "state_dim": int(sample["observation.state"].shape[-1]),
        "action_dim": int(sample["action"].shape[-1]),
        "front_ndim": int(sample["observation.images.front"].ndim),  # CHW at load
    }
    assert info["state_dim"] == STATE_DIM, info
    assert info["action_dim"] == ACTION_DIM, info
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert native episodes to a LeRobotDataset.")
    ap.add_argument("--src", default="datasets/rebot_scripted")
    ap.add_argument("--repo-id", required=True, help="e.g. $HF_USER/rebot_reseat")
    ap.add_argument("--root", default="datasets/lerobot/rebot_reseat")
    ap.add_argument("--image-mode", choices=["video", "image"], default="video")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    convert(
        args.src,
        args.repo_id,
        root=args.root,
        use_videos=(args.image_mode == "video"),
        push=args.push,
    )
    if args.verify:
        print(verify(args.repo_id, args.root))


if __name__ == "__main__":
    main()
