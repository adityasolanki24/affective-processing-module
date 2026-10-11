"""Target-specific advisory action tendencies (bounded 0..1), derived from appraisal frames and memory.

Advisory only: the host robot's planner may weigh them; they never command motion and never override safety.

Per target (APM design formulas):
  avoid           Σ relevance·max(−d,0)·likelihood·controllability          (adverse and avoidable)
  disengage       Σ relevance·max(−d,0)·likelihood·(1 − controllability)    (adverse, little control)
  approach        Σ relevance·max(d,0)·likelihood                           (beneficial)
  orient          Σ relevance·(1 − expectedness) over recent observed outcomes, plus novelty of a new entity
  explore         novelty · (1 − expected adverse probability)
  seek_assistance only for "self": strain · (1 − control), control = mean controllability of adverse frames,
                  or the stability margin when there are none (a fallen robot keeps asking)
Each Σ is a noisy-OR; outputs are exponentially smoothed with TAU_S so they do not chatter with the input rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from apm.core.affect import noisy_or
from apm.core.appraisal import Frame

TAU_S = 0.5
NAMES = ("approach", "avoid", "orient", "explore", "disengage", "seek_assistance")


def raw_tendencies(frames: list[Frame], novelty: dict[str, float], p_adverse: dict[str, float],
                   strain: float, stability: float | None = None) -> dict[str, dict[str, float]]:
    targets = {f.target or "self" for f in frames} | set(novelty) | {"self"}
    out = {}
    for t in targets:
        fs = [f for f in frames if (f.target or "self") == t]
        adverse = [f for f in fs if f.desirability < 0]
        out[t] = {
            "approach": noisy_or([f.relevance * f.desirability * f.likelihood for f in fs if f.desirability > 0]),
            "avoid": noisy_or([f.relevance * -f.desirability * f.likelihood * f.controllability for f in adverse]),
            "disengage": noisy_or([f.relevance * -f.desirability * f.likelihood * (1 - f.controllability)
                                   for f in adverse]),
            "orient": noisy_or([f.relevance * (1 - f.expectedness) for f in fs if f.reference_type == "outcome"]
                               + [novelty.get(t, 0.0)]),
            "explore": novelty.get(t, 0.0) * (1.0 - p_adverse.get(t, 0.0)),
            "seek_assistance": 0.0,
        }
    self_adverse = [f for f in frames if f.desirability < 0]
    if self_adverse:
        control = sum(f.controllability for f in self_adverse) / len(self_adverse)
    else:  # no live adverse appraisal: the body's own stability margin is the best evidence of control left
        control = stability if stability is not None else 1.0
    out["self"]["seek_assistance"] = min(1.0, max(0.0, strain * (1.0 - control)))
    return out


@dataclass
class TendencyFilter:
    state: dict[str, dict[str, float]] = field(default_factory=dict)
    t: float | None = None

    def update(self, raw: dict[str, dict[str, float]], now_s: float) -> dict[str, dict[str, float]]:
        alpha = 1.0 if self.t is None else 1.0 - math.exp(-max(0.0, now_s - self.t) / TAU_S)
        self.t = now_s
        for target in set(raw) | set(self.state):
            current = self.state.get(target, dict.fromkeys(NAMES, 0.0))
            new = raw.get(target, dict.fromkeys(NAMES, 0.0))
            self.state[target] = {k: current[k] + alpha * (new[k] - current[k]) for k in NAMES}
        for target in [t for t, v in self.state.items() if t not in raw and max(v.values()) < 1e-3]:
            del self.state[target]
        return self.state
