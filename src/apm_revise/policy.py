"""Decision-A rule (G) with a revising memory and optional information gathering.

`Revising` is G's `MemoryRegulated` with `RevisionMemory` in place of `BeliefMemory`. Rule, regulator, predictor and
safety supervisor are unchanged. Exploration modes (`explore["mode"]`):

- `none`.
- `uncertainty`, `random`: G2b's probes, inherited unchanged (budget per episode; eligibility: the rule would
  YIELD/DETOUR, the nearest person is within 1.5 m, nobody else is within 2 m).
- `voi`: same eligibility and budget. Instead of G2b's confidence threshold, probe (SLOW) when the expected value
  of the information exceeds what the rule gives up by probing:

      ν · [R(b) − E_c R(b'(c))]  >  U(rule action) − U(SLOW)

  where R(b) = 1 − max_k b_k is the chance of acting on the wrong type, and the expectation runs over the next
  encounter's outcome category c under the current belief b and the type model. ν converts information into
  utility units: an *exchange rate*, tuned on validation only.
- `thompson`: posterior sampling inside the memory (`RevisionMemory(thompson=True)`); no probe rule.

If the memory is not a `RevisionMemory`, `voi` falls back to no probing.
"""
from __future__ import annotations

import numpy as np

from apm_revise.belief import TypeModel
from apm_revise.memory import RevisionMemory
from apm_select.memory import RELIABLE_CONF
from apm_select.policies import PROBE_ALONE_M, PROBE_NEAR_M, MemoryRegulated
from apm_select.regulators import Regulated
from apm_testbeds.wheeled_ctrl import DETOUR, GO, SLOW, YIELD


def bayes_risk(b: np.ndarray) -> float:
    return 1.0 - float(np.max(b))


def expected_posterior_risk(b: np.ndarray, E: np.ndarray, w: float = 1.0) -> float:
    """E_c[R(b'(c))] for one more encounter, with outcome category c ~ Σ_k b_k E[k, c]."""
    pc = b @ E                                   # [C] predictive distribution of the category
    post = b[:, None] * E ** w                   # [K, C] unnormalised posteriors
    post = post / np.maximum(1e-300, post.sum(0, keepdims=True))
    return float(pc @ (1.0 - post.max(0)))


class Revising(MemoryRegulated):
    def __init__(self, predictor, lam_c: float, lam_s: float, regulator, model: TypeModel, lam: float,
                 explore: dict | None = None, trace=None, name: str = "I", confidence: str = "entropy"):
        explore = explore or {"mode": "none"}
        probe = explore if explore["mode"] in ("uncertainty", "random") else None
        super().__init__(predictor, lam_c, lam_s, regulator, {"probe": probe}, trace=trace, name=name)
        self.type_model, self.lam, self.explore, self.confidence = model, lam, explore, confidence

    def begin_lifetime(self) -> None:
        super().begin_lifetime()
        self.memory = RevisionMemory(self.type_model, self.lam,  # type: ignore[assignment]  # ObsMemory interface
                                     thompson=self.explore["mode"] == "thompson", seed=self.explore.get("seed", 0),
                                     confidence=self.confidence)
        self.voi_log: list[tuple[float, float]] = []  # (value of information, utility given up) at eligible moments

    def decide(self, t: float) -> int:
        if self.explore["mode"] != "voi":
            return super().decide(t)
        a = Regulated.decide(self, t)
        if a not in (YIELD, DETOUR) or self.probes_episode >= self.explore["budget"]:
            return a
        obs = self.state.last_obs
        people = sorted((e for e in obs.entities if e.kind == "person"), key=lambda e: e.distance) if obs else []
        if not people or people[0].distance > PROBE_NEAR_M or any(e.distance < PROBE_ALONE_M for e in people[1:]):
            return a
        e = people[0]
        if e.identity is None or e.identity_confidence < RELIABLE_CONF or not isinstance(self.memory, RevisionMemory):
            return a
        b = self.memory.posterior(e.identity)
        value = self.explore["nu"] * (bayes_risk(b) - expected_posterior_risk(b, self.type_model.E,
                                                                              e.identity_confidence))
        assert self.last_utility is not None
        cost = float(self.last_utility[a] - self.last_utility[SLOW])
        self.voi_log.append((value, cost))
        if value <= cost:
            return a
        self.probes += 1
        self.probes_episode += 1
        self.probe_log.append((self.probes_episode, a, SLOW))
        return SLOW


class KeepAway:
    """Memory-driven keep-away: the robot stops (YIELD) for a person it believes risky, before they get close.

    Wraps any policy that has `.memory` (with `p_collision`) and `.last_utility`. After the inner policy decides, if
    the nearest person is reliably identified, within `keep_m` and (if `approach_only`) approaching (closing speed
    > 0), and their memory p is at least `kappa` × the memory's population prior p, the robot YIELDs. A stopped
    robot hides how people behave (they walk around it), so this creates real censoring: the avoidance trap.

    Information gathering (`explore["mode"]`) decides, once per encounter and within a per-episode budget, whether
    to *probe* instead. A probed encounter is exempt from keep-away, so the inner rule acts normally, capped at SLOW.
    - `none`;
    - `uncertainty`: probe if the memory confidence is below `conf_max` (G2b's criterion);
    - `random`: probe with probability `p_random` (budget-matched control);
    - `voi`: probe if ν·[R(b) − E_c R(b'(c))] > U(YIELD) − U(SLOW) (RevisionMemory only; otherwise no probing);
    - `thompson`: no probe rule; the memory's per-encounter posterior sample drives the keep-away test itself.
    """

    def __init__(self, inner, keep_m: float = 2.0, kappa: float = 1.5, explore: dict | None = None,
                 approach_only: bool = True):
        self.inner, self.keep_m, self.kappa, self.approach_only = inner, keep_m, kappa, approach_only
        self.explore = explore or {"mode": "none"}
        self.name = getattr(inner, "name", "KA")

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def begin_lifetime(self) -> None:
        self.inner.begin_lifetime()
        self.ka_probes = 0
        self.keepaways = 0
        self.decided: dict[str, bool] = {}   # track → probe decision for this encounter
        self.rng = np.random.default_rng(self.explore.get("seed", 0))

    def begin_episode(self, t0: float) -> None:
        self.inner.begin_episode(t0)
        self.ka_probes_episode = 0
        self.decided = {}

    def observe(self, obs, t: float) -> None:
        self.inner.observe(obs, t)

    def end_episode(self, t: float) -> None:
        self.inner.end_episode(t)

    @property
    def probes(self) -> int:
        return self.ka_probes + getattr(self.inner, "probes", 0)

    def _pool_p(self) -> float:
        mem = self.inner.memory
        if isinstance(mem, RevisionMemory):
            return mem.posterior_p(None)[0]
        return mem.p_collision(None, "person")[0]

    def _probe(self, e, mem) -> bool:
        mode = self.explore["mode"]
        if mode in ("none", "thompson") or self.ka_probes_episode >= self.explore.get("budget", 2):
            return False
        if mode == "uncertainty":
            return mem.p_collision(e.identity, "person")[1] < self.explore["conf_max"]
        if mode == "random":
            return bool(self.rng.random() < self.explore["p_random"])
        if mode == "voi" and isinstance(mem, RevisionMemory):
            b = mem.posterior(e.identity)
            value = self.explore["nu"] * (bayes_risk(b) - expected_posterior_risk(b, mem.model.E,
                                                                                  e.identity_confidence))
            u = self.inner.last_utility
            return bool(value > float(u[YIELD] - u[SLOW]))
        return False

    def decide(self, t: float) -> int:
        a = self.inner.decide(t)
        obs = self.inner.state.last_obs
        people = sorted((e for e in obs.entities if e.kind == "person"), key=lambda e: e.distance) if obs else []
        if not people:
            return a
        e = people[0]
        if e.distance > self.keep_m or e.identity is None or e.identity_confidence < RELIABLE_CONF:
            return a
        mem = self.inner.memory
        if self.decided.get(e.track):
            return SLOW if a == GO else a  # probing this encounter: approach, at most at SLOW speed
        if mem.p_collision(e.identity, "person")[0] < self.kappa * self._pool_p():
            return a
        if self.approach_only and e.closing <= 0.0:
            return a
        if e.track not in self.decided:
            self.decided[e.track] = self._probe(e, mem)
            if self.decided[e.track]:
                self.ka_probes += 1
                self.ka_probes_episode += 1
                return SLOW if a == GO else a
        self.keepaways += 1
        return YIELD
