"""Closed-loop eval for reBot. Runs a controller (scripted baseline now, learned policy
later) over a scenario suite and reports the full metric set. The scripted path is
torch-free; the policy path lazy-imports policies.infer (the torch zone), so eval/ stays
importable without the ML extra.

reBot uses its own runner rather than lerobot-eval, which is bound to its own gym envs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import EpisodeResult, summarize
from eval.task_suite import Scenario, default_suite


def _insertion_time(timeline: list[tuple[str, float]]) -> float | None:
    t = dict(timeline)
    if "inserted" in t and "at_mouth" in t:
        return float(t["inserted"] - t["at_mouth"])
    return None


def run_scenario(scenario: Scenario, *, controller: str = "scripted", ckpt: str | None = None):
    # Imports kept local so eval/ has no module-level backend/torch dependency.
    from perception.pose_estimation import NoisyPerception, SimGroundTruthPerception
    from robot_core.robots.mujoco_robot import MujocoRobot
    from sim.domain_randomization import apply, sample

    robot = MujocoRobot()
    apply(robot, sample(scenario.dr_config, scenario.seed))
    perception = SimGroundTruthPerception(robot)
    if scenario.perception_noise > 0:
        perception = NoisyPerception(perception, scenario.perception_noise, seed=scenario.seed)

    try:
        if controller == "scripted":
            from policies.scripted.reseat import run_reseat

            res = run_reseat(robot, perception)
            seated, success = res.seated, res.success
            peak, estop, retries = res.peak_insert_force, res.estop, res.search_retries
            itime = _insertion_time(res.timeline)
        elif controller == "policy":
            from policies.infer import run_policy

            res = run_policy(robot, perception, ckpt)
            seated, success = res["seated"], res["success"]
            peak, estop, retries = res["peak_insert_force"], res["estop"], res.get("search_retries", 0)
            itime = res.get("insertion_time_s")
        else:
            raise ValueError(f"unknown controller {controller}")
    finally:
        robot.close()

    return EpisodeResult(
        name=scenario.name,
        category=scenario.category,
        seed=scenario.seed,
        seated=bool(seated),
        success=bool(success),
        insertion_time_s=itime,
        peak_insert_force=float(peak),
        search_retries=int(retries),
        interventions=1 if estop else 0,
        estop=estop,
    )


def run_eval(
    suite: list[Scenario], *, controller: str = "scripted", ckpt: str | None = None, verbose: bool = False
) -> tuple[dict, list[EpisodeResult]]:
    results: list[EpisodeResult] = []
    for s in suite:
        r = run_scenario(s, controller=controller, ckpt=ckpt)
        results.append(r)
        if verbose:
            print(
                f"  {r.name:14s} seated={r.seated} peak={r.peak_insert_force:5.1f}N "
                f"estop={r.estop}"
            )
    by_cat: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    return summarize(by_cat), results


def _print_report(report: dict, controller: str) -> None:
    print(f"\n=== reBot eval report ({controller}) ===")
    print(f"episodes: {report['n_episodes']}  overall seat rate: {report['overall_seat_rate']:.1%}")
    for cat in ("train", "heldout", "recovery"):
        if cat in report:
            c = report[cat]
            it = c["insertion_time"]["mean"]
            print(
                f"  {cat:9s} n={c['n']:3d} seat={c['seat_rate']:.0%} "
                f"peak(mean/max)={c['peak_force']['mean']:.1f}/{c['peak_force']['max']:.1f}N "
                f"insert_t={'n/a' if it is None else f'{it:.2f}s'} "
                f"interventions={c['interventions']}"
            )
    if "generalization_score" in report:
        print(f"  generalization score: {report['generalization_score']:.2f}")
    if "recovery_rate" in report:
        print(f"  recovery rate: {report['recovery_rate']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the reBot eval suite.")
    ap.add_argument("--controller", default="scripted", choices=["scripted", "policy"])
    ap.add_argument("--ckpt", default=None, help="policy checkpoint (for --controller policy)")
    ap.add_argument("--n-each", type=int, default=10, help="scenarios per category")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    suite = default_suite(args.n_each)
    report, results = run_eval(suite, controller=args.controller, ckpt=args.ckpt, verbose=True)
    _print_report(report, args.controller)
    if args.out:
        Path(args.out).write_text(
            json.dumps({"report": report, "results": [r.to_dict() for r in results]}, indent=2)
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
