"""M0 smoke tests: the toolchain is present and the core stays torch-free."""
import importlib
import sys


def test_mujoco_importable() -> None:
    import mujoco

    assert mujoco.__version__


def test_numpy_importable() -> None:
    import numpy as np

    assert np.__version__


def test_core_packages_import_torch_free() -> None:
    # The torch-free core must import without pulling in torch.
    for name in ["robot_core", "sim", "data", "eval"]:
        importlib.import_module(name)
    assert "torch" not in sys.modules, "core import pulled in torch"
