"""Regression checks for scientific calculations and controller observation conventions."""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from apm_isaac.booster import JOINTS, observation
from apm_isaac.measurements import person_distance
from apm_isaac.world import _HeadDriver
from apm_transfer.embodied_expression import memory_expression
from apm_transfer.runtime import Decision
from apm_transfer.sensor_memory import SensorExpectation


def motion_module():
    scripts = Path(__file__).resolve().parents[1] / "experiments/M_release/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("remediation_motion", scripts / "m6_motion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tied_spearman_is_invariant_to_row_order():
    m = motion_module()
    x = [0, 0, 0, 1, 1, 1]
    y = [0, 1, 2, 2, 1, 0]
    assert m.spearman(x, y) == pytest.approx(0.)
    order = [0, 3, 1, 4, 2, 5]
    assert m.spearman(np.array(x)[order], np.array(y)[order]) == pytest.approx(0.)
    assert m.spearman(x, x) == pytest.approx(1.)
    assert m.spearman([1] * 6, y) is None
    assert m.spearman([0, 1, 2, 3, 4, np.nan], [4, 3, 2, 1, 0, 5]) == pytest.approx(-1.)


def test_official_t1_observation_has_only_leg_joints():
    default = np.arange(23) * .01
    q, dq = default + 1., np.ones(23) * 2.
    obs = observation(q, dq, default, [0., 0., -1.], [1., 2., 3.], [.2, 0., .5], 0., 1., np.zeros(12))
    assert obs.shape == (47,)
    assert len(JOINTS) == len(set(JOINTS)) == 23
    np.testing.assert_allclose(obs[:3], [0., 0., -1.])
    np.testing.assert_allclose(obs[9:11], [1., 0.])
    np.testing.assert_allclose(obs[11:23], 1.)
    np.testing.assert_allclose(obs[23:35], .2)
    q[:11] += 100
    np.testing.assert_allclose(obs, observation(q, dq, default, [0., 0., -1.], [1., 2., 3.],
                                             [.2, 0., .5], 0., 1., np.zeros(12)))


def test_physical_distance_tracks_the_person_not_the_station():
    far = person_distance([0., 0.], [1., 0.], .35, .22)
    near = person_distance([0., 0.], [.6, 0.], .35, .22)
    assert far["radial_surface_gap_m"] == pytest.approx(.43)
    assert near["radial_surface_gap_m"] == pytest.approx(.03)
    assert person_distance([0., 0.], [.5, 0.], .35, .22)["radial_surface_gap_m"] < 0


def test_expression_uses_memory_with_navigation_matched():
    d = Decision(0, "B1", "B", 0, 1, 0., .5, .7, .1, .1, .2, .1, 0., False, "pilot")
    low = memory_expression(d, match_motion=True)
    high = memory_expression(replace(d, p_risk=.9), match_motion=True)
    assert (low.tempo, low.proximity, low.hesitation) == (high.tempo, high.proximity, high.hesitation)
    assert high.scanning > low.scanning
    assert high.head_elevation < low.head_elevation
    assert high.directness < low.directness
    assert memory_expression(replace(d, p_risk=float("nan"))) == d.intent()


def test_sway_is_not_counted_as_forward_walking():
    scripts = Path(__file__).resolve().parents[1] / "experiments/M_release/scripts"
    spec = importlib.util.spec_from_file_location("remediation_gait", scripts / "analyze_gait.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.forward_velocity([0., .3, 0.], [1., 0., 0., 0.]) == pytest.approx(0.)
    half = np.sqrt(.5)
    assert module.forward_velocity([0., .2, 0.], [half, 0., 0., half]) == pytest.approx(.2)


def test_still_head_does_not_mask_sensors_and_returns_before_walking():
    class Robot:
        t = 0.
        v_cmd = w_cmd = 0.
        expression_head = np.zeros(2)

        def apply(self, v, w):
            self.last_request = (v, w)
            return False

    robot = Robot()
    driver = _HeadDriver(robot, 1)
    driver.head[:] = [.2, .1]
    driver.prev_head = driver.head.copy()
    assert not driver.own_motion((0., 0.), False)
    driver.head[0] += .02
    assert driver.own_motion((0., 0.), False)
    driver.send((.2, 0.))
    assert robot.last_request == (0., 0.)
    assert driver.head[0] == pytest.approx(.205)


def test_sensor_memory_is_person_specific_revisable_and_ignores_absent_windows():
    memory = SensorExpectation()
    before = memory.state().copy()
    memory.observe("A", None)
    assert memory.state() == before
    for _ in range(6):
        memory.observe("A", [[.3]])
        memory.observe("B", [[]])
    assert memory.readout("A")["movement_probability"] > .7
    assert memory.readout("B")["movement_probability"] < .3
    initial_a = memory.readout("A")["movement_probability"]
    for _ in range(20):
        memory.observe("A", [[]])
    assert memory.readout("A")["movement_probability"] < .25 < initial_a
    copy = SensorExpectation()
    copy.load(memory.state())
    assert copy.readout("A") == memory.readout("A")
    memory.observe("A", [[float("nan"), -.2, .1]])
    assert memory.readout("A")["movement_probability"] < .25


def test_runtime_persists_sensor_memory_without_changing_navigation(tmp_path):
    import json

    from apm_transfer.infoseek import Task
    from apm_transfer.runtime import Observation, Runtime

    root = Path(__file__).resolve().parents[1]
    model = json.loads((root / "experiments/K_recovery/configs/K_types.json").read_text())
    time = json.loads((root / "experiments/J_validation_transfer/configs/J6_model.json").read_text())
    tasks = json.loads((root / "experiments/K_recovery/configs/K_tasks.json").read_text())
    task = Task("handoff", **{k: v for k, v in tasks["tasks"]["handoff"].items() if k != "why"})
    a = Runtime(model, time["time_coef"], task, tasks["voi"], seed=3)
    b = Runtime(model, time["time_coef"], task, tasks["voi"], seed=3)
    b.sensor_expectation = SensorExpectation()
    for _ in range(6):
        assert a.decide("B1", "B") == b.decide("B1", "B")
        obs = Observation("B1", "B", 1, imu=(7,), camera=(5,), detected=True,
                          report={"camera_reaches": [[.3]]})
        a.observe(obs)
        b.observe(obs)
    state = tmp_path / "memory.json"
    b.save(state)
    c = Runtime(model, time["time_coef"], task, tasks["voi"], seed=3)
    c.load(state)
    assert c.state() == b.state()
    assert c.sensor_expectation.readout("B1")["movement_probability"] > .7
    assert c.decide("B1", "B") == b.decide("B1", "B")
