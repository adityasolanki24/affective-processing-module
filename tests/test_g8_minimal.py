"""G8 components: heuristic predictor shape/monotonicity, single-member predictor, memory removal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from apm_research.features import MEM_P, WIDTH
from apm_select.minimal import HeuristicPredictor

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "E2" / "predictor.json"


def test_heuristic_predictor_orders_actions():
    x = np.zeros(WIDTH)
    x[2], x[3], x[4] = 0.9, 0.5, 1.0  # traction, goal distance/8.5, cos(bearing)
    x[7], x[12], x[13] = 1.0, 0.8, 0.3  # one person present, ttc_max 0.8, memory p 0.3
    p_col, p_slip, prog = HeuristicPredictor().predict_actions(x)
    assert p_col[0] > p_col[3] > p_col[1] > p_col[2] >= 0
    assert prog[0] > prog[3] > prog[1] > prog[2] == 0
    assert np.all((p_slip >= 0) & (p_slip <= 1))
    x2 = x.copy()
    x2[MEM_P[0]] = 0.9
    assert HeuristicPredictor().predict_actions(x2)[0][0] > p_col[0]


@pytest.mark.skipif(not PRED.exists(), reason="frozen predictor not present")
def test_g8_policy_variants_run_and_differ():
    sys.path.insert(0, str(ROOT / "experiments" / "G_architecture_selection" / "scripts"))
    import g_common as gc
    gc.init(str(PRED))
    base = {"kind": "regulated", "regulator": "static", "lam_c": 2.5, "lam_s": 1.5}
    runs = {k: gc.run_job({"spec": {**base, **extra}, "experiment": "G8", "partition": "P", "index": 0,
                           "n_episodes": 2})["actions"]
            for k, extra in (("full", {}), ("heur", {"predictor": "heuristic"}), ("single", {"predictor": "single"}),
                             ("nomem", {"no_memory": True}))}
    assert runs["heur"] != runs["full"] and runs["single"] != runs["full"]
    assert sum(runs["nomem"]) == sum(runs["full"])
