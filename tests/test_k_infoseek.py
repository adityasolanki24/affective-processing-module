"""Programme K: camera coding, multi-observation memory updates, task-grounded decisions and value of information."""
from __future__ import annotations

import numpy as np

from apm_revise.belief import TypeModel
from apm_transfer.expression_map import GRID, TimeModel, band_of, stop_distance
from apm_transfer.infoseek import Emissions, Task, best_greedy, choose, cost
from apm_transfer.nerva_memory import TypeStateModel
from apm_transfer.nerva_world import camera_code

TM = TimeModel([20.0, 10.0, -17.0, -1.5, 6.5])


def _model() -> TypeModel:
    # 2 types; IMU blocks per band (near, mid, far) over 8 categories, then a camera block.
    calm_imu = [0.4, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01]
    push_near = [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.3, 0.3]
    push_far = [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1]
    calm_cam = [0.8, 0.12, 0.05, 0.0, 0.02, 0.0, 0.0, 0.01]
    push_cam = [0.25, 0.05, 0.15, 0.2, 0.05, 0.1, 0.1, 0.1]
    E = np.array([calm_imu * 3 + calm_cam, push_near + push_near + push_far + push_cam], float) + 1e-3
    E = (E.reshape(2, 4, 8) / E.reshape(2, 4, 8).sum(2, keepdims=True)).reshape(2, 32)
    return TypeModel("k_test", np.array([0.5, 0.5]), E, np.array([0.1, 0.5]))


def test_camera_code_bins():
    assert camera_code([]) == 0
    assert camera_code([0.2]) == 1 and camera_code([0.5]) == 2 and camera_code([0.7]) == 3
    assert camera_code([0.2, 0.7]) == 6
    assert camera_code([0.1, 0.1, 0.1]) == 7


def test_observe_many_updates_once_and_learns_nothing_from_nothing():
    m = TypeStateModel(_model(), lam=0.0)
    m.observe_many("B1", "B", [], 0.0)
    assert "B1" not in m.b and m.n == 1  # nothing observed: nothing learned, but time passes
    m.observe_many("B1", "B", [2 * 8 + 7, 24 + 3], 1.0)  # strong far IMU evidence + a long lunge seen
    assert m.belief("B1", "B")[1] > 0.8 and m.n == 2


def test_greedy_respects_risk_cap_and_task_requirement():
    em = Emissions.from_model(_model().E, camera_block=True)
    handoff = Task("handoff", d_task=0.3, retry_s=12.6)
    i_calm, _ = best_greedy(np.array([1.0, 0.0]), TM, em, handoff)
    i_push, _ = best_greedy(np.array([0.0, 1.0]), TM, em, handoff)
    assert stop_distance(GRID[i_calm].proximity) <= 0.3 + 1e-9  # a calm entity: go close enough for the task
    assert stop_distance(GRID[i_push].proximity) > stop_distance(GRID[i_calm].proximity)
    assert cost(i_push, 1, np.array([0.0, 1.0]), TM, em, handoff)[1] <= handoff.risk_cap
    sep = Task("separation", d_task=None, retry_s=12.6, min_sep=0.5)
    i, _ = best_greedy(np.array([1.0, 0.0]), TM, em, sep)
    assert stop_distance(GRID[i].proximity) >= 0.5 - 1e-9


def test_voi_values_a_camera_more_than_a_weak_far_imu():
    model = _model()
    em_cam = Emissions.from_model(model.E, camera_block=True)
    em_imu = Emissions(em_cam.imu, None)
    task = Task("handoff", d_task=0.3, retry_s=12.6)
    b = np.array([0.35, 0.65])  # unsure whether the entity is still pushy
    rng = np.random.default_rng(0)
    i_c, w_c, d_c = choose("voi", b, model.pi, TM, em_cam, task, rng=rng)
    i_i, w_i, d_i = choose("voi", b, model.pi, TM, em_imu, task, rng=np.random.default_rng(0))
    assert d_c["voi"] >= d_i["voi"]
    for i, w in ((i_c, w_c), (i_i, w_i)):
        assert cost(i, w, b, TM, em_cam, task)[1] <= task.risk_cap  # information never bypasses the safety budget
    assert band_of(GRID[i_c].proximity) in (0, 1, 2)
