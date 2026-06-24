"""M6: eval metrics, the scenario suite, the two-level handoff gate, and an end-to-end
scripted-baseline eval that reports the full metric set."""
import pytest

from eval.metrics import EpisodeResult, generalization_score, seat_success_rate, summarize
from eval.task_suite import default_suite, recovery_suite, train_suite
from policies.infer import near_mouth


def _r(category: str, seated: bool) -> EpisodeResult:
    return EpisodeResult(
        name="x",
        category=category,
        seed=0,
        seated=seated,
        success=seated,
        insertion_time_s=1.0 if seated else None,
        peak_insert_force=5.0,
        search_retries=0,
        interventions=0,
        estop=None,
    )


def test_near_mouth_gate():
    assert near_mouth([0.5, 0.0, 0.3], [0.5, 0.0, 0.3])
    assert not near_mouth([0.5, 0.0, 0.3], [0.6, 0.0, 0.3])


def test_metrics_summary():
    by = {
        "train": [_r("train", True), _r("train", True)],
        "heldout": [_r("heldout", True), _r("heldout", False)],
        "recovery": [_r("recovery", True)],
    }
    rep = summarize(by)
    assert rep["overall_seat_rate"] == pytest.approx(0.8)  # 4/5
    assert rep["generalization_score"] == pytest.approx(0.5)  # 0.5 / 1.0
    assert rep["recovery_rate"] == 1.0
    assert rep["train"]["seat_rate"] == 1.0


def test_generalization_score_guard():
    assert generalization_score([_r("train", False)], [_r("heldout", True)]) == 0.0


def test_suite_composition():
    s = default_suite(3)
    assert len(s) == 9
    assert sum(x.category == "train" for x in s) == 3
    assert all(x.perception_noise > 0 for x in recovery_suite(2))
    assert all(x.perception_noise == 0 for x in train_suite(2))


@pytest.mark.slow
def test_run_eval_scripted_reports(  # end-to-end, exercises the sim
):
    from eval.run_eval import run_eval

    report, results = run_eval(default_suite(1), controller="scripted")
    assert report["n_episodes"] == 3
    for cat in ("train", "heldout", "recovery"):
        assert cat in report
        assert "peak_force" in report[cat]
    assert "generalization_score" in report
    assert "recovery_rate" in report
    assert report["train"]["seat_rate"] == 1.0  # nominal scripted baseline seats
