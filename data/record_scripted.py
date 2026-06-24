"""Run the scripted skill under domain randomization and record episodes in the native
format. A thin RecordingRobot wraps the backend and captures (observation, action) frames
at a fixed decimation, rendering images only on recorded frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.schema import CAMERAS, EpisodeBuffer, EpisodeMeta, save_episode, write_manifest
from policies.scripted.reseat import run_reseat
from robot_core.robot import Robot
from robot_core.robots.mujoco_robot import MujocoRobot
from sim.domain_randomization import DRConfig, apply, sample

DEFAULT_TASK = "Reseat the hard drive into the bay."


class RecordingRobot(Robot):
    """Wraps a backend, recording (s_t, a_t) frames every `record_every` control steps.

    Images are rendered only on recorded frames. Every other HAL call delegates to the
    inner backend, so the scripted skill and perception are unchanged.
    """

    def __init__(self, inner: MujocoRobot, *, record_every: int = 3) -> None:
        self.inner = inner
        self.record_every = record_every
        self.supports_torque = inner.supports_torque
        self.buffer = EpisodeBuffer(CAMERAS)
        self._count = 0

    def step(self, action):
        if self._count % self.record_every == 0:
            obs = self.inner.get_obs(with_images=True)  # render the current state s_t
            self.buffer.add_frame(obs, action.delta)
        self._count += 1
        return self.inner.step(action)

    def reset(self):
        self.buffer = EpisodeBuffer(CAMERAS)
        self._count = 0
        return self.inner.reset()

    # Delegation.
    def get_obs(self, with_images: bool = True):
        return self.inner.get_obs(with_images)

    def seated(self) -> bool:
        return self.inner.seated()

    def estopped(self) -> bool:
        return self.inner.estopped()

    def set_stiffness_mode(self, mode):
        self.inner.set_stiffness_mode(mode)

    def set_payload_compensation(self, f):
        self.inner.set_payload_compensation(f)

    def attach_payload(self):
        self.inner.attach_payload()

    def detach_payload(self):
        self.inner.detach_payload()

    def set_joint_torque(self, tau6):
        self.inner.set_joint_torque(tau6)

    def read_joint_state(self):
        return self.inner.read_joint_state()

    def read_ft(self):
        return self.inner.read_ft()

    def set_gripper(self, o):
        self.inner.set_gripper(o)

    def render(self, *a, **k):
        return self.inner.render(*a, **k)

    def site_xpos(self, name):
        return self.inner.site_xpos(name)

    def info(self):
        return self.inner.info()

    @property
    def model(self):
        return self.inner.model

    @property
    def data(self):
        return self.inner.data

    @property
    def home_key(self):
        return self.inner.home_key

    @property
    def drive_qadr(self):
        return self.inner.drive_qadr


def _insertion_time(timeline: list[tuple[str, float]]) -> float | None:
    t = dict(timeline)
    if "inserted" in t and "at_mouth" in t:
        return float(t["inserted"] - t["at_mouth"])
    return None


def record_episode(
    index: int,
    seed: int,
    *,
    out_dir: str | Path,
    image_hw: tuple[int, int],
    record_every: int,
    dr_config: DRConfig,
    task: str = DEFAULT_TASK,
) -> EpisodeMeta:
    inner = MujocoRobot(image_hw=image_hw)
    params = sample(dr_config, seed)
    apply(inner, params)
    rec = RecordingRobot(inner, record_every=record_every)
    result = run_reseat(rec)  # resets, records frames at the decimation cadence

    control_hz = 1.0 / (inner.substeps * inner.model.opt.timestep)
    meta = EpisodeMeta(
        episode_index=index,
        seed=seed,
        task=task,
        fps=int(round(control_hz / record_every)),
        image_hw=tuple(image_hw),
        cameras=CAMERAS,
        num_frames=len(rec.buffer),
        seat_success=bool(result.seated),
        insertion_time_s=_insertion_time(result.timeline),
        peak_insert_force=float(result.peak_insert_force),
        search_retries=result.search_retries if hasattr(result, "search_retries") else 0,
        dr_params=params.to_dict(),
    )
    save_episode(out_dir, meta, rec.buffer.arrays())
    inner.close()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Record scripted reseat demos with DR.")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="datasets/rebot_scripted")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--image-hw", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--record-every", type=int, default=3)
    args = ap.parse_args()

    dr_config = DRConfig()
    out_dir = Path(args.out_dir)
    metas: list[EpisodeMeta] = []
    successes = 0
    for i in range(args.n_episodes):
        seed = args.seed_base + i
        meta = record_episode(
            i,
            seed,
            out_dir=out_dir,
            image_hw=tuple(args.image_hw),
            record_every=args.record_every,
            dr_config=dr_config,
        )
        successes += int(meta.seat_success)
        rate = successes / (i + 1)
        print(
            f"ep {i:04d} seed={seed} frames={meta.num_frames} "
            f"seated={meta.seat_success} peak={meta.peak_insert_force:.1f}N "
            f"running_success={rate:.2%}"
        )
        metas.append(meta)

    rate = successes / max(args.n_episodes, 1)
    total_frames = sum(m.num_frames for m in metas)
    write_manifest(
        out_dir,
        {
            "schema_version": metas[0].schema_version if metas else "rebot-ep-1",
            "n_episodes": len(metas),
            "n_success": successes,
            "success_rate": rate,
            "total_frames": total_frames,
            "fps": metas[0].fps if metas else None,
            "image_hw": list(args.image_hw),
            "cameras": list(CAMERAS),
        },
    )
    print(f"\nDONE: {successes}/{args.n_episodes} seated ({rate:.1%}), {total_frames} frames")


if __name__ == "__main__":
    main()
