"""Thin wrapper over the LeRobot unified CLI. We do not hand-roll training loops: this
builds the `lerobot-train` argv and shells out, so torch lives entirely behind the
subprocess boundary (ideally in the Python 3.11 .venv-train or on the GPU box).

Targets, in order: act, diffusion, smolvla, pi0. ACT is the first target (trains in ~1-2 h
on an A100 under 8 GB VRAM from tens to low hundreds of demos).

Flags are confirmed against `lerobot-train --help` for the installed version before a real
run (the CLI version-drifts); the [VERIFY] points note where.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def build_command(
    *,
    repo_id: str,
    policy_type: str = "act",
    output_dir: str = "outputs/train/act_rebot",
    job_name: str = "act_rebot",
    steps: int = 100_000,
    batch_size: int = 8,
    device: str = "cuda",
    wandb: bool = False,
    policy_repo_id: str | None = None,
) -> list[str]:
    # [VERIFY] flag names against `lerobot-train --help` in the installed version.
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--policy.device={device}",
        f"--wandb.enable={'true' if wandb else 'false'}",
    ]
    if policy_repo_id:
        cmd.append(f"--policy.repo_id={policy_repo_id}")
    return cmd


def run(**kwargs) -> int:
    if shutil.which("lerobot-train") is None:
        raise RuntimeError(
            "lerobot-train not found. Install the [train] extra into the Python 3.11 "
            ".venv-train or run on the GPU box."
        )
    cmd = build_command(**kwargs)
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, check=True).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch lerobot-train for a reBot policy.")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--policy", default="act", choices=["act", "diffusion", "smolvla", "pi0"])
    ap.add_argument("--output-dir", default="outputs/train/act_rebot")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--policy-repo-id", default=None)
    args = ap.parse_args()
    sys.exit(
        run(
            repo_id=args.repo_id,
            policy_type=args.policy,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            wandb=args.wandb,
            policy_repo_id=args.policy_repo_id,
        )
    )


if __name__ == "__main__":
    main()
