"""Static validation of recorded episodes against the native schema. Used as a CI gate
and by the record->replay round-trip test. No sim required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.schema import CAMERAS, VECTOR_SPEC, load_episode
from robot_core.types import STATE_DIM, STATE_SLICES


def validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    image_hw: tuple[int, int] | None = None,
    cameras: tuple[str, ...] = CAMERAS,
) -> int:
    """Assert one episode's arrays match the Observation/Action contract. Returns T."""
    assert "action" in arrays, "missing action"
    t = int(arrays["action"].shape[0])
    assert t > 0, "empty episode"

    for key, (shape, _dtype) in VECTOR_SPEC.items():
        assert key in arrays, f"missing array {key}"
        a = arrays[key]
        assert a.shape == (t,) + shape, f"{key} shape {a.shape} != {(t,) + shape}"
        assert a.dtype == np.float32, f"{key} dtype {a.dtype} != float32"

    for cam in cameras:
        key = f"images.{cam}"
        assert key in arrays, f"missing image array {key}"
        img = arrays[key]
        assert img.ndim == 4 and img.shape[0] == t and img.shape[3] == 3, (
            f"{key} shape {img.shape}"
        )
        assert img.dtype == np.uint8, f"{key} dtype {img.dtype} != uint8"
        if image_hw is not None:
            assert tuple(img.shape[1:3]) == tuple(image_hw), f"{key} hw {img.shape[1:3]}"

    # state must equal build_state(components): catches a hand-rolled concat bug.
    recomputed = np.zeros((t, STATE_DIM), np.float32)
    recomputed[:, STATE_SLICES["joint_pos"]] = arrays["joint_pos"]
    recomputed[:, STATE_SLICES["joint_vel"]] = arrays["joint_vel"]
    recomputed[:, STATE_SLICES["gripper"]] = arrays["gripper"][:, None]
    recomputed[:, STATE_SLICES["ee_pose"]] = arrays["ee_pose"]
    recomputed[:, STATE_SLICES["ft"]] = arrays["ft"]
    assert np.allclose(arrays["state"], recomputed, atol=1e-5), "state != build_state(components)"

    g = arrays["gripper"]
    assert g.min() >= -1e-6 and g.max() <= 1 + 1e-6, "gripper out of [0,1]"
    return t


def validate_dataset(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    npzs = sorted(out.glob("episode_*.npz"))
    assert npzs, f"no episodes under {out}"
    report = {"n_episodes": len(npzs), "total_frames": 0, "failures": []}
    for npz in npzs:
        try:
            arrays, meta = load_episode(npz)
            hw = tuple(meta.get("image_hw")) if meta.get("image_hw") else None
            t = validate_arrays(arrays, image_hw=hw)
            assert meta.get("seed") is not None, "missing seed in sidecar"
            report["total_frames"] += t
        except AssertionError as e:
            report["failures"].append({"file": npz.name, "error": str(e)})
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a recorded dataset directory.")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    report = validate_dataset(args.out_dir)
    print(report)
    if report["failures"]:
        raise SystemExit(f"{len(report['failures'])} episode(s) failed validation")
    print(f"OK: {report['n_episodes']} episodes, {report['total_frames']} frames")


if __name__ == "__main__":
    main()
