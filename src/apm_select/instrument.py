"""Evaluation instrumentation (G0): exposure, encounters, action switching, hesitation and decision latency.

`Instrumented` wraps any policy and forwards every call unchanged (the harness's fallback behaviour is preserved:
exceptions from the wrapped policy propagate exactly as before). It reads only the `Obs` stream that the policy
itself receives; nothing it computes is visible to the policy. Ground truth is not used here.

Per lifetime (sums over episodes):
  person_encounters      (episode, person track) pairs whose clearance fell below 1.0 m ("relevant encounters")
  close_encounters       … below 0.5 m
  person_exposure_s      seconds with at least one person within 1.0 m
  switches               decisions whose action differs from the previous decision in the episode
  oscillations           A → B → A within three consecutive decisions (A ≠ B)
  hesitant               SLOW or YIELD decisions with no person within 1.5 m and no obstacle within 0.5 m
  decide_us              total wall time spent in `decide` (µs), decisions
  observe_us             total wall time spent in `observe` (µs; the 20 Hz state updates, e.g. the APM core)
  max_decide_us          the slowest single decision
"""
from __future__ import annotations

import time

from apm_testbeds.wheeled_ctrl import SLOW, YIELD

ENCOUNTER_M, CLOSE_M, HESITATION_PERSON_M, HESITATION_OBSTACLE_M = 1.0, 0.5, 1.5, 0.5
KEYS = ("person_encounters", "close_encounters", "person_exposure_s", "switches", "oscillations", "hesitant",
        "decisions", "decide_us", "max_decide_us", "observe_us")


class Instrumented:
    def __init__(self, policy):
        self.policy = policy
        self.name = getattr(policy, "name", "policy")
        self.stats = dict.fromkeys(KEYS, 0.0)

    def __getattr__(self, item):  # readouts, memory, … of the wrapped policy (evaluation only)
        return getattr(self.policy, item)

    def begin_lifetime(self) -> None:
        self.stats = dict.fromkeys(KEYS, 0.0)
        self.policy.begin_lifetime()

    def begin_episode(self, t0: float) -> None:
        self.encountered: set[str] = set()
        self.close: set[str] = set()
        self.actions: list[int] = []
        self.last_obs = None
        self.t_prev: float | None = None
        self.policy.begin_episode(t0)

    def observe(self, obs, t: float) -> None:
        people = [e for e in obs.entities if e.kind == "person"]
        for e in people:
            if e.distance < ENCOUNTER_M:
                self.encountered.add(e.track)
            if e.distance < CLOSE_M:
                self.close.add(e.track)
        if self.t_prev is not None and any(e.distance < ENCOUNTER_M for e in people):
            self.stats["person_exposure_s"] += t - self.t_prev
        self.t_prev, self.last_obs = t, obs
        start = time.perf_counter()
        try:
            self.policy.observe(obs, t)
        finally:
            self.stats["observe_us"] += (time.perf_counter() - start) * 1e6

    def decide(self, t: float) -> int:
        start = time.perf_counter()
        try:
            a = self.policy.decide(t)
        finally:
            us = (time.perf_counter() - start) * 1e6
            self.stats["decide_us"] += us
            self.stats["max_decide_us"] = max(self.stats["max_decide_us"], us)
            self.stats["decisions"] += 1
        if isinstance(a, int) and self.actions:
            self.stats["switches"] += a != self.actions[-1]
            if len(self.actions) >= 2 and a == self.actions[-2] and a != self.actions[-1]:
                self.stats["oscillations"] += 1
        obs = self.last_obs
        if a in (SLOW, YIELD) and obs is not None:
            near_person = any(e.kind == "person" and e.distance < HESITATION_PERSON_M for e in obs.entities)
            near_obstacle = any(e.kind != "person" and e.distance < HESITATION_OBSTACLE_M for e in obs.entities)
            self.stats["hesitant"] += not (near_person or near_obstacle)
        self.actions.append(a)
        return a

    def end_episode(self, t: float) -> None:
        self.stats["person_encounters"] += len(self.encountered)
        self.stats["close_encounters"] += len(self.close)
        self.policy.end_episode(t)
