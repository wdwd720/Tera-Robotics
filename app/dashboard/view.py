"""Minimal dashboard view: compose a frame from the camera images plus a status footer
(FT magnitude bar, gripper, phase, seated). Torch-free; uses opencv for drawing and the
mp4 writer. Used by the headless demo capture and re-usable by a live viewer later.
"""
from __future__ import annotations

import numpy as np

TILE = 240
FOOTER = 96
GREEN = (40, 200, 40)
RED = (40, 40, 220)
WHITE = (235, 235, 235)
GRAY = (90, 90, 90)


def _resize(img: np.ndarray, size: int):
    import cv2

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def compose(
    images: dict[str, np.ndarray],
    *,
    ft: np.ndarray,
    gripper: float,
    seated: bool,
    phase: str,
    step: int,
    cameras: tuple[str, ...] = ("front", "wrist"),
    ft_full_scale: float = 30.0,
) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB dashboard frame."""
    import cv2

    tiles = []
    for cam in cameras:
        tile = _resize(np.ascontiguousarray(images[cam]), TILE)
        cv2.putText(tile, cam, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        tiles.append(tile)
    strip = np.concatenate(tiles, axis=1)
    w = strip.shape[1]

    footer = np.full((FOOTER, w, 3), 25, dtype=np.uint8)
    fmag = float(np.linalg.norm(np.asarray(ft)[:3]))
    # FT magnitude bar.
    bar_w = int(np.clip(fmag / ft_full_scale, 0, 1) * (w - 220))
    cv2.rectangle(footer, (200, 14), (200 + (w - 220), 30), GRAY, 1)
    cv2.rectangle(footer, (200, 14), (200 + bar_w, 30), GREEN if fmag < 20 else RED, -1)
    cv2.putText(footer, f"|FT| {fmag:5.1f} N", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"phase: {phase}    grip: {gripper:.2f}    step: {step}",
        (8, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    status = "SEATED" if seated else "in progress"
    cv2.putText(footer, status, (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN if seated else WHITE, 2, cv2.LINE_AA)
    return np.concatenate([strip, footer], axis=0)


def write_video(path: str, frames: list[np.ndarray], fps: int = 30) -> str:
    """Write RGB frames to an mp4. Returns the path."""
    import cv2

    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return path
