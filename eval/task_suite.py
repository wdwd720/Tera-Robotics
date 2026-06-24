"""Eval scenarios: train-distribution, held-out generalization, and induced-failure
recovery. Seeded and declarative. Torch-free and mujoco-free."""
from __future__ import annotations

from dataclasses import dataclass, field

from sim.domain_randomization import DRConfig


@dataclass
class Scenario:
    name: str
    category: str  # "train" | "heldout" | "recovery"
    seed: int
    dr_config: DRConfig = field(default_factory=DRConfig)
    perception_noise: float = 0.0  # meters of bay-pose error to recover from


# Held-out config: pose ranges beyond the training distribution (generalization).
HELDOUT_CONFIG = DRConfig(bay_pos=0.02, drive_pos=0.035, friction=(0.6, 1.4), drive_mass=(0.6, 1.5))
RECOVERY_NOISE = 0.0015  # in-plane bay-pose error (~clearance scale) the search recovers


def train_suite(n: int, seed0: int = 0) -> list[Scenario]:
    return [Scenario(f"train_{i:03d}", "train", seed0 + i) for i in range(n)]


def heldout_suite(n: int, seed0: int = 10_000) -> list[Scenario]:
    return [
        Scenario(f"heldout_{i:03d}", "heldout", seed0 + i, dr_config=HELDOUT_CONFIG)
        for i in range(n)
    ]


def recovery_suite(n: int, seed0: int = 20_000) -> list[Scenario]:
    return [
        Scenario(f"recovery_{i:03d}", "recovery", seed0 + i, perception_noise=RECOVERY_NOISE)
        for i in range(n)
    ]


def default_suite(n_each: int = 10) -> list[Scenario]:
    return train_suite(n_each) + heldout_suite(n_each) + recovery_suite(n_each)
