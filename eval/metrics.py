"""Eval metrics over episode results. Pure functions, torch-free and mujoco-free."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EpisodeResult:
    name: str
    category: str
    seed: int
    seated: bool
    success: bool
    insertion_time_s: float | None
    peak_insert_force: float
    search_retries: int
    interventions: int
    estop: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _rate(results: list[EpisodeResult], key) -> float:
    return float(np.mean([key(r) for r in results])) if results else 0.0


def seat_success_rate(results: list[EpisodeResult]) -> float:
    return _rate(results, lambda r: r.seated)


def insertion_time_stats(results: list[EpisodeResult]) -> dict:
    times = [r.insertion_time_s for r in results if r.seated and r.insertion_time_s is not None]
    if not times:
        return {"mean": None, "p90": None}
    return {"mean": float(np.mean(times)), "p90": float(np.percentile(times, 90))}


def peak_force_stats(results: list[EpisodeResult]) -> dict:
    forces = [r.peak_insert_force for r in results]
    if not forces:
        return {"mean": None, "max": None}
    return {"mean": float(np.mean(forces)), "max": float(np.max(forces))}


def mean_search_retries(results: list[EpisodeResult]) -> float:
    return _rate(results, lambda r: r.search_retries)


def total_interventions(results: list[EpisodeResult]) -> int:
    return int(sum(r.interventions for r in results))


def generalization_score(train: list[EpisodeResult], heldout: list[EpisodeResult]) -> float:
    """Held-out seat rate divided by train-distribution seat rate (clamped)."""
    base = seat_success_rate(train)
    if base <= 0:
        return 0.0
    return float(seat_success_rate(heldout) / base)


def recovery_rate(recovery: list[EpisodeResult]) -> float:
    return seat_success_rate(recovery)


def summarize(by_category: dict[str, list[EpisodeResult]]) -> dict:
    """Build the full metrics report from results grouped by category."""
    allr = [r for rs in by_category.values() for r in rs]
    report: dict = {"n_episodes": len(allr), "overall_seat_rate": seat_success_rate(allr)}
    for cat, rs in by_category.items():
        report[cat] = {
            "n": len(rs),
            "seat_rate": seat_success_rate(rs),
            "insertion_time": insertion_time_stats(rs),
            "peak_force": peak_force_stats(rs),
            "mean_search_retries": mean_search_retries(rs),
            "interventions": total_interventions(rs),
        }
    if "train" in by_category and "heldout" in by_category:
        report["generalization_score"] = generalization_score(
            by_category["train"], by_category["heldout"]
        )
    if "recovery" in by_category:
        report["recovery_rate"] = recovery_rate(by_category["recovery"])
    return report
