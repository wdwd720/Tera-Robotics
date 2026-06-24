"""Replay a recorded episode. Passive mode just loads and summarizes (no sim). Open-loop
mode re-applies the recorded domain randomization and feeds the recorded actions back
through the sim, comparing the resulting state trajectory to the recording.
"""
from __future__ import annotations

import argparse

import numpy as np

from data.dataset_validator import validate_arrays
from data.schema import load_episode


def summarize(npz_path: str) -> dict:
    arrays, meta = load_episode(npz_path)
    t = validate_arrays(arrays, image_hw=tuple(meta["image_hw"]) if meta.get("image_hw") else None)
    summary = {
        "frames": t,
        "task": meta.get("task"),
        "seed": meta.get("seed"),
        "seat_success": meta.get("seat_success"),
        "peak_insert_force": meta.get("peak_insert_force"),
        "image_hw": meta.get("image_hw"),
        "action_range": [float(arrays["action"].min()), float(arrays["action"].max())],
    }
    return summary


def open_loop_replay(npz_path: str, *, tol: float = 0.05) -> float:
    """Re-run the recorded actions in a fresh sim with the same DR. Returns the max
    state deviation. Imports the sim backend lazily (mujoco needed)."""
    from robot_core.robots.mujoco_robot import MujocoRobot
    from robot_core.types import Action
    from sim.domain_randomization import DRParams, apply

    arrays, meta = load_episode(npz_path)
    robot = MujocoRobot(image_hw=tuple(meta["image_hw"]))
    params = DRParams(**meta["dr_params"])
    apply(robot, params)
    robot.reset()
    actions = arrays["action"]
    max_dev = 0.0
    for a in actions:
        robot.step(Action(a))
    # Compare final proprio state (open-loop divergence is expected to be bounded).
    final = robot.get_obs(with_images=False).state
    dev = float(np.max(np.abs(final[:20] - arrays["state"][-1][:20])))  # ignore noisy ft
    max_dev = max(max_dev, dev)
    robot.close()
    return max_dev


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay/inspect a recorded episode.")
    ap.add_argument("npz")
    ap.add_argument("--open-loop", action="store_true")
    args = ap.parse_args()
    print(summarize(args.npz))
    if args.open_loop:
        print("max state deviation (open loop):", open_loop_replay(args.npz))


if __name__ == "__main__":
    main()
