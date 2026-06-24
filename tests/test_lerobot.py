"""M5: LeRobot conversion. The feature spec matches the contract, the converter is
torch-free at import (the heavy import is guarded inside functions), and a full
convert + CPU-train smoke runs when the [train] extra is installed (skipped otherwise so
the default suite stays green; it runs on the GPU box)."""
import importlib
import sys

import pytest

from data.convert_to_lerobot import build_features
from robot_core.types import ACTION_DIM, STATE_DIM


def _has_lerobot() -> bool:
    try:
        import lerobot  # noqa: F401

        return True
    except Exception:
        return False


def test_build_features_matches_contract():
    f = build_features((128, 128))
    assert f["observation.state"]["shape"] == [STATE_DIM]
    assert f["action"]["shape"] == [ACTION_DIM]
    assert f["observation.images.front"]["shape"] == [128, 128, 3]
    assert f["observation.images.front"]["dtype"] == "video"
    assert f["observation.images.wrist"]["names"] == ["height", "width", "channel"]


def test_converter_is_torch_free_at_import():
    sys.modules.pop("data.convert_to_lerobot", None)
    importlib.import_module("data.convert_to_lerobot")
    assert "torch" not in sys.modules, "importing the converter pulled in torch"


def test_require_lerobot_raises_without_extra():
    import data.convert_to_lerobot as c

    if _has_lerobot():
        pytest.skip("lerobot installed; guard not exercised")
    with pytest.raises(RuntimeError):
        c._require_lerobot()


@pytest.mark.lerobot
@pytest.mark.skipif(not _has_lerobot(), reason="lerobot [train] extra not installed")
def test_convert_and_cpu_smoke(tmp_path):
    import subprocess

    from data.convert_to_lerobot import convert, verify
    from data.record_scripted import record_episode
    from sim.domain_randomization import DRConfig

    src = tmp_path / "native"
    for i in range(2):
        record_episode(i, i, out_dir=src, image_hw=(64, 64), record_every=6, dr_config=DRConfig())

    root = tmp_path / "lerobot_ds"
    convert(src, "test/rebot_smoke", root=root, use_videos=False)
    info = verify("test/rebot_smoke", root)
    assert info["num_frames"] > 0

    # 2-step CPU smoke of lerobot-train: it must load the dataset and step, not converge.
    cmd = [
        "lerobot-train",
        "--dataset.repo_id=test/rebot_smoke",
        f"--dataset.root={root}",
        "--policy.type=act",
        "--steps=2",
        "--batch_size=2",
        "--policy.device=cpu",
        f"--output_dir={tmp_path / 'out'}",
        "--wandb.enable=false",
    ]
    subprocess.run(cmd, check=True)
