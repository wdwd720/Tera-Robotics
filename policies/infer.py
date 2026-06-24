"""Closed-loop inference for a learned LeRobot policy in sim, with the two-level control
handoff: the learned policy drives the coarse approach (perception-driven grasp and gross
insertion), and the deterministic force-search controller does the last few millimeters of
compliant seating. The network never has to learn sub-millimeter force control.

This is the torch zone: torch/lerobot are imported lazily inside _require_policy so eval/
and the rest of the stack stay importable without the [infer] extra.
"""
from __future__ import annotations

import numpy as np

from robot_core.control.primitives import (
    confirm_seat,
    insert_with_search,
    open_gripper,
    retreat,
)
from robot_core.types import Action

GRIP_CLOSED = 0.4  # below this opening the policy is considered to have grasped


def near_mouth(ee_pos: np.ndarray, mouth_pos: np.ndarray, tol: float = 0.03) -> bool:
    """Geometric handoff gate: the ee is within tol of the bay mouth. Pure function."""
    return float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(mouth_pos))) < tol


def _require_policy(ckpt: str):
    try:
        import torch  # noqa: F401
        from lerobot.policies.act.modeling_act import ACTPolicy  # [VERIFY] import path
    except Exception as e:  # pragma: no cover - only with the extra installed
        raise RuntimeError(
            "policy inference needs the [infer] extra (lerobot + torch). The reBot core "
            "and eval are torch-free."
        ) from e
    policy = ACTPolicy.from_pretrained(ckpt)  # [VERIFY] loader API for the installed version
    policy.eval()
    policy.reset()
    return policy, torch


def _obs_to_policy_input(obs, task: str, torch):
    """Map a HAL Observation to the policy input dict. [VERIFY] the exact image format
    (CHW float vs HWC, and whether the policy applies its own normalization)."""
    inp = {
        "observation.state": torch.from_numpy(obs.state).float()[None],
        "task": task,
    }
    for cam, img in obs.images.items():
        chw = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        inp[f"observation.images.{cam}"] = chw
    return inp


def run_policy(
    robot,
    perception,
    ckpt: str,
    *,
    task: str = "Reseat the hard drive into the bay.",
    max_approach_steps: int = 600,
) -> dict:
    """Run the learned coarse approach, hand off to the scripted seater near the mouth,
    confirm, and retreat. Returns a result dict matching the eval runner's expectations."""
    policy, torch = _require_policy(ckpt)
    robot.reset()
    mouth = perception.bay_mouth()

    grasped = False
    peak = 0.0
    handed_off = False
    with torch.no_grad():
        for _ in range(max_approach_steps):
            obs = robot.get_obs(with_images=True)
            if not grasped and obs.gripper < GRIP_CLOSED:
                grasped = True
                robot.attach_payload()  # firm grasp once the policy has closed the jaws
            if near_mouth(obs.ee_pose[:3], mouth):
                handed_off = True
                break
            action = policy.select_action(_obs_to_policy_input(obs, task, torch))
            robot.step(Action(np.asarray(action).reshape(-1)[:7]))
            if robot.estopped():
                break

    # Low-level compliant seating (the same primitive the scripted expert uses).
    seated = False
    if handed_off and not robot.estopped():
        base = robot.get_obs(with_images=False).ee_pose.copy()
        trace: list[float] = []
        seated = insert_with_search(
            robot, base, grip=0.0, monitor=lambda i: trace.append(abs(i["f_insert"]))
        )
        peak = max(trace) if trace else 0.0
        confirm_seat(robot, grip=0.0)
        robot.set_payload_compensation(0.0)
        robot.detach_payload()
        open_gripper(robot)
        retreat(robot, 0.10, grip=1.0)

    return {
        "seated": robot.seated(),
        "success": bool(seated and not robot.estopped()),
        "peak_insert_force": float(peak),
        "search_retries": 0,
        "insertion_time_s": None,
        "estop": robot.info().get("estop") if hasattr(robot, "info") else None,
    }
