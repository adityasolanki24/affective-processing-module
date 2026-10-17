"""Episode/lifetime runner with common random numbers, fallback on policy failure and per-lifetime metrics.

A lifetime = a robot (one policy instance, persistent memory) living through `n_episodes` worlds that share one
population of identities. Exogenous randomness comes only from the lifetime seed (`seeds.py`), so every system sees
the same worlds, people, goals and slip noise. Policies get `Obs` only.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apm_testbeds.wheeled_ctrl import (
    DECISION_STEPS,
    GO,
    N_ACTIONS,
    Identity,
    WheeledCtrlWorld,
    WorldConfig,
    child,
)

EPISODE_GAP_S = 10.0
IDLE_SPEED, IDLE_CLEARANCE = 0.05, 1.0
J_WEIGHTS = {"collisions": -1.0, "severe": -2.0, "slips": -0.25, "goals_completed": 1.0}


def make_population(rng: np.random.Generator, n: int = 6, p_careless: float = 0.4) -> list[Identity]:
    return [Identity(f"ped-{i}", "careless" if rng.random() < p_careless else "careful") for i in range(n)]


@dataclass
class Lifetime:
    seed: np.random.SeedSequence
    n_episodes: int = 10
    config: WorldConfig = field(default_factory=WorldConfig)
    n_people: int = 6
    p_careless: float = 0.4
    # (episode index, base population) → population in force for that episode (E3 disposition changes)
    schedule: Callable[[int, list[Identity]], list[Identity]] | None = None

    def population(self) -> list[Identity]:
        return make_population(np.random.default_rng(child(self.seed, 0)), self.n_people, self.p_careless)

    def world(self, episode: int) -> WheeledCtrlWorld:
        base = self.population()
        pop = self.schedule(episode, base) if self.schedule else base
        return WheeledCtrlWorld(child(self.seed, 1 + episode), pop, self.config)


def _decide(policy, t: float) -> tuple[int, bool]:
    try:
        a = policy.decide(t)
        if isinstance(a, (int, np.integer)) and 0 <= int(a) < N_ACTIONS:
            return int(a), False
    except Exception:  # noqa: BLE001 - any policy failure must fall back, never stop the robot loop
        pass
    return GO, True


def run_episode(world: WheeledCtrlWorld, policy, t0: float, recorder=None) -> dict:
    failed = False
    try:
        policy.begin_episode(t0)
        policy.observe(world.observe_now(), t0)
    except Exception:  # noqa: BLE001
        failed = True
    log = world.log
    action, k = GO, 0
    period_speed, period_near, period_intervention = 0.0, False, False
    idle = periods = fallbacks = 0
    counts = np.zeros(N_ACTIONS, int)
    while not world.done:
        if k % DECISION_STEPS == 0:
            if k > 0:
                periods += 1
                idle += period_speed / DECISION_STEPS < IDLE_SPEED and not period_near and not period_intervention
            period_speed, period_near, period_intervention = 0.0, False, False
            if failed:
                action, fell_back = GO, True
            else:
                action, fell_back = _decide(policy, t0 + world.t)
            fallbacks += fell_back
            counts[action] += 1
            if recorder is not None:
                recorder.decision(world, policy, action, t0 + world.t)
        obs = world.step(action)
        if not failed:
            try:
                policy.observe(obs, t0 + obs.t)
            except Exception:  # noqa: BLE001
                failed = True
        period_speed += abs(obs.v)
        period_near |= any(e.distance < IDLE_CLEARANCE for e in obs.entities)
        period_intervention |= any(ev[0] == "intervention" for ev in obs.events)
        k += 1
    periods += 1
    idle += period_speed / DECISION_STEPS < IDLE_SPEED and not period_near and not period_intervention
    try:
        policy.end_episode(t0 + world.t)
    except Exception:  # noqa: BLE001
        pass
    if recorder is not None:
        recorder.end_episode(world)
    cols = log.collisions
    return {"minutes": world.t / 60.0, "goals_completed": log.goals_completed, "goals_issued": log.goals_issued,
            "timeouts": log.timeouts, "goal_time_sum": float(sum(log.goal_times)), "collisions": len(cols),
            "person_collisions": sum(c[2] == "person" for c in cols),
            "careless_collisions": sum(c[4] == "careless" for c in cols),
            "careful_collisions": sum(c[4] == "careful" for c in cols),
            "obstacle_collisions": sum(c[2] == "obstacle" for c in cols), "severe": sum(c[5] for c in cols),
            "slips": len(log.slips), "interventions": log.interventions, "distance": log.distance,
            "progress": log.progress, "idle": int(idle), "periods": periods, "fallbacks": int(fallbacks),
            "actions": counts.tolist(), "policy_failed": failed,
            "collision_log": [(round(c[0], 2), c[1], c[3], c[4], bool(c[5])) for c in cols]}


def run_lifetime(lifetime: Lifetime, policy, recorder=None, episodes: list[int] | None = None) -> dict:
    policy.begin_lifetime()
    rows = []
    for e in episodes if episodes is not None else range(lifetime.n_episodes):
        world = lifetime.world(e)
        t0 = e * (lifetime.config.episode_s + EPISODE_GAP_S)
        row = run_episode(world, policy, t0, recorder)
        row["episode"] = e
        rows.append(row)
    return summarize(rows)


SUM_KEYS = ("minutes", "goals_completed", "goals_issued", "timeouts", "goal_time_sum", "collisions",
            "person_collisions", "careless_collisions", "careful_collisions", "obstacle_collisions", "severe",
            "slips", "interventions", "distance", "progress", "idle", "periods", "fallbacks")


def summarize(rows: list[dict]) -> dict:
    total: dict[str, Any] = {k: float(sum(r[k] for r in rows)) for k in SUM_KEYS}
    total["actions"] = np.sum([r["actions"] for r in rows], axis=0).tolist()
    total["J"] = sum(w * total[k] for k, w in J_WEIGHTS.items())
    total["completion"] = total["goals_completed"] / max(1.0, total["goals_issued"])
    total["collision_rate"] = total["collisions"] / max(1e-9, total["minutes"])
    total["idle_fraction"] = total["idle"] / max(1.0, total["periods"])
    total["episodes"] = rows
    return total
