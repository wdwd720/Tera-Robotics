"""M3: the scripted reseat skill seats the drive on the nominal scene, the peak insert
force stays under the safety bound, and no e-stop fires."""
from robot_core.safety import SafetyLimits
from policies.scripted.reseat import run_reseat


def test_scripted_reseat_seats(robot):
    result = run_reseat(robot)
    assert result.grasped, "failed to grasp the drive"
    assert result.estop is None, f"e-stop fired: {result.estop}"
    assert result.seated, "drive not seated (seated() hook false after the sequence)"
    assert result.success, "reseat reported failure"
    # Peak insert force stays under the safety bound.
    assert result.peak_insert_force < SafetyLimits().insert_force_max, result.peak_insert_force


def test_reseat_phase_timeline(robot):
    result = run_reseat(robot)
    phases = [name for name, _ in result.timeline]
    for expected in [
        "start",
        "above_grasp",
        "at_grasp",
        "grasped",
        "lifted",
        "at_mouth",
        "inserted",
        "confirmed",
        "retreated",
    ]:
        assert expected in phases, expected
