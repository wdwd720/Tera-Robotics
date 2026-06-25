"""M8: the dashboard composes a valid frame and writes an mp4 (no GL/sim needed)."""
import os

import numpy as np

from app.dashboard.view import compose, write_video


def _imgs(rng, hw=(64, 64)):
    return {c: rng.integers(0, 256, (hw[0], hw[1], 3), dtype=np.uint8) for c in ("front", "wrist")}


def test_compose_returns_valid_frame():
    rng = np.random.default_rng(0)
    frame = compose(
        _imgs(rng),
        ft=np.array([1.0, 0.0, -2.0, 0.0, 0.0, 0.0]),
        gripper=0.5,
        seated=False,
        phase="carry",
        step=10,
    )
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[1] == 480  # two 240px camera tiles
    assert frame.shape[0] == 240 + 96  # tile height + footer


def test_write_video(tmp_path):
    rng = np.random.default_rng(1)
    frames = [
        compose(_imgs(rng), ft=np.zeros(6), gripper=1.0, seated=(i > 1), phase="insert", step=i)
        for i in range(4)
    ]
    out = str(tmp_path / "demo.mp4")
    write_video(out, frames, fps=10)
    assert os.path.getsize(out) > 0
