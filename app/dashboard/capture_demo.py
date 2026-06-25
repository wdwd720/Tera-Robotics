"""Record a clean reseat demo to an mp4 from a headless render (front + wrist cameras),
with the dashboard footer (FT, gripper, phase, seated). Works on the M1 via the default
CGL backend; on the GPU box set MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.dashboard.view import compose, write_video
from data.record_scripted import RecordingRobot
from robot_core.math_utils import quat_to_mat


def _phase(obs, seated: bool) -> str:
    if seated:
        return "seated"
    if obs.gripper > 0.7:
        return "approach"
    f_insert = float((quat_to_mat(obs.ee_pose[3:7]) @ obs.ft[:3])[0])
    return "insert" if abs(f_insert) > 2.5 else "carry"


class DemoRecorder(RecordingRobot):
    """Reuses the recording wrapper's HAL delegation but collects composed dashboard
    frames instead of dataset frames."""

    def __init__(self, inner, *, record_every: int = 2) -> None:
        super().__init__(inner, record_every=record_every)
        self.frames: list[np.ndarray] = []

    def reset(self):
        self.frames = []
        return super().reset()

    def step(self, action):
        if self._count % self.record_every == 0:
            obs = self.inner.get_obs(with_images=True)
            seated = self.inner.seated()
            self.frames.append(
                compose(
                    obs.images,
                    ft=obs.ft,
                    gripper=obs.gripper,
                    seated=seated,
                    phase=_phase(obs, seated),
                    step=self._count,
                )
            )
        self._count += 1
        return self.inner.step(action)


def capture(
    out: str = "outputs/demo.mp4",
    *,
    image_hw: tuple[int, int] = (240, 240),
    record_every: int = 2,
    fps: int = 25,
):
    from policies.scripted.reseat import run_reseat
    from robot_core.robots.mujoco_robot import MujocoRobot

    robot = MujocoRobot(image_hw=image_hw)
    rec = DemoRecorder(robot, record_every=record_every)
    result = run_reseat(rec)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_video(out, rec.frames, fps=fps)
    robot.close()
    return out, result, len(rec.frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture a headless reseat demo video.")
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--image-hw", type=int, nargs=2, default=[240, 240])
    ap.add_argument("--record-every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    out, result, n = capture(
        args.out, image_hw=tuple(args.image_hw), record_every=args.record_every, fps=args.fps
    )
    print(f"wrote {out}: {n} frames, seated={result.seated}, success={result.success}")


if __name__ == "__main__":
    main()
