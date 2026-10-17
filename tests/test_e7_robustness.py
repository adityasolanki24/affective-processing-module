"""E7 criteria (2) and (3) as unit tests (MASTER_PREREGISTRATION.md §2 E7)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from apm_research.features import WIDTH
from apm_research.harness import Lifetime, run_lifetime
from apm_research.policies import B0, B1, B2
from apm_research.predictor import ActionPredictor
from apm_testbeds.wheeled_ctrl import GO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "F_behavioral_learning"))
import seeds as seedmod  # noqa: E402


def zero_predictor() -> ActionPredictor:
    p = ActionPredictor(WIDTH)
    p.weights = [(np.zeros((WIDTH + 4, 3), np.float32), np.zeros(3, np.float32))]
    return p


def test_model_failure_falls_back_to_go_in_the_same_decision_period():
    class Failing(B1):
        calls = 0

        def decide(self, t):
            self.calls += 1
            if self.calls % 3 == 0:
                raise RuntimeError("model failure")
            return 1  # SLOW otherwise, so a fallback is visible

    policy = Failing(zero_predictor(), 1.0, 0.5)
    out = run_lifetime(Lifetime(seedmod.world_seed("E7", "P", 0), n_episodes=1), policy)
    ep = out["episodes"][0]
    assert ep["fallbacks"] == policy.calls // 3
    assert ep["actions"][GO] == ep["fallbacks"]  # every failed period ran GO, never a stale action


def test_safety_supervisor_identical_with_apm_present_absent_or_crashed():
    class PresentButGo(B2):  # the full APM runs every step; the action is always GO
        def decide(self, t):
            super().decide(t)
            return GO

    class Crashed(B2):
        def observe(self, obs, t):
            if t > 3.0:
                raise RuntimeError("APM crashed")
            super().observe(obs, t)

        def decide(self, t):
            super().decide(t)
            return GO

    lt = Lifetime(seedmod.world_seed("E7", "P", 1), n_episodes=2)
    runs = [run_lifetime(lt, p) for p in (B0(), PresentButGo(zero_predictor(), 1.0, 0.5, 1.0, 0.25),
                                          Crashed(zero_predictor(), 1.0, 0.5, 1.0, 0.25))]
    for r in runs[1:]:
        assert r["interventions"] == runs[0]["interventions"]
        assert [e["collision_log"] for e in r["episodes"]] == [e["collision_log"] for e in runs[0]["episodes"]]
    assert runs[2]["episodes"][0]["policy_failed"]
