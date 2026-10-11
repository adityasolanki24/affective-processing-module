"""Decision-time features, identical for every learned or rule-based system (MASTER_PREREGISTRATION.md §1.1).

Computed only from `Obs` received up to the decision time plus the explicit memory. Width `WIDTH`:
  0–6    own speed/0.8, |turn rate|/2, traction margin, goal distance/8.5, cos/sin goal bearing, deadline left/30 s
  7–33   three nearest people (by clearance), 9 each: present, clearance/3, cos/sin bearing, closing/2,
         max TTC-estimator probability over the last decision period (high-rate summary), memory P(collision per
         encounter), memory confidence, reported identity confidence
  34–38  nearest obstacle: present, clearance/3, cos/sin bearing, closing/2
  39–40  exp(−time since last collision / 5 s), exp(−time since last slip / 5 s)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

K_PEOPLE = 3
PERSON_W = 9
WIDTH = 7 + K_PEOPLE * PERSON_W + 5 + 2
NAMES = (["speed", "turn", "traction", "goal_dist", "goal_cos", "goal_sin", "deadline"]
         + [f"p{i}_{n}" for i in range(K_PEOPLE) for n in ("present", "dist", "cos", "sin", "closing", "ttc_max",
                                                            "mem_p", "mem_conf", "id_conf")]
         + ["obst_present", "obst_dist", "obst_cos", "obst_sin", "obst_closing", "recent_collision", "recent_slip"])
MEM_P = [7 + i * PERSON_W + 6 for i in range(K_PEOPLE)]
MEM_CONF = [7 + i * PERSON_W + 7 for i in range(K_PEOPLE)]


class DecisionState:
    """High-rate summaries collected between decisions (same for every system)."""

    def __init__(self) -> None:
        self.last_obs: Any = None
        self.ttc_max: dict[str, float] = {}
        self.last_collision = -1e9
        self.last_slip = -1e9

    def observe(self, obs, t: float) -> None:
        self.last_obs = obs
        for e in obs.entities:
            self.ttc_max[e.track] = max(self.ttc_max.get(e.track, 0.0), e.ttc_p)
        for event in obs.events:
            if event[0] == "collision":
                self.last_collision = t
            elif event[0] == "slip":
                self.last_slip = t

    def features(self, t: float, memory_p) -> np.ndarray:
        """`memory_p(identity, kind) -> (p, confidence)`. Resets the per-period summaries."""
        x = np.zeros(WIDTH, dtype=np.float32)
        obs = self.last_obs
        if obs is None:
            return x
        x[0:7] = (obs.v / 0.8, abs(obs.omega) / 2.0, obs.traction_margin, obs.goal_distance / 8.5,
                  math.cos(obs.goal_bearing), math.sin(obs.goal_bearing), min(obs.goal_remaining_s, 30.0) / 30.0)
        people = sorted((e for e in obs.entities if e.kind == "person"), key=lambda e: e.distance)[:K_PEOPLE]
        for i, e in enumerate(people):
            p, conf = memory_p(e.identity, e.kind)
            b = 7 + i * PERSON_W
            x[b:b + PERSON_W] = (1.0, e.distance / 3.0, math.cos(e.bearing), math.sin(e.bearing),
                                 max(-1.0, min(1.0, e.closing / 2.0)), self.ttc_max.get(e.track, 0.0), p, conf,
                                 e.identity_confidence if e.identity else 0.0)
        obstacles = [e for e in obs.entities if e.kind != "person"]
        if obstacles:
            e = min(obstacles, key=lambda e: e.distance)
            x[34:39] = (1.0, e.distance / 3.0, math.cos(e.bearing), math.sin(e.bearing),
                        max(-1.0, min(1.0, e.closing / 2.0)))
        x[39] = math.exp(-(t - self.last_collision) / 5.0)
        x[40] = math.exp(-(t - self.last_slip) / 5.0)
        self.ttc_max = {}
        return x
