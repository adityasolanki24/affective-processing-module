"""Programme K: task-grounded expression with safe information gathering (robot-independent decision layer).

**Inputs.**
- A belief b over learned entity types: `TypeStateModel._at`, change-aware, relaxing at rate λ.
- The robot's own learned models:
  - `TimeModel`: arrival time as a function of intent;
  - emissions E[k, block, c] of the type model: IMU blocks per stopping-distance band, plus an optional camera block.
- A `Task` whose values are fixed a priori from task requirements (`experiments/K_recovery/PROTOCOL.md`).

**Task cost of an approach** with intent a and w sensing windows, under belief b:

    C(a, w | b) = T(a) + 4 (w − 1)                                   time (s)
                + price · w · Σ_k b_k P(strong | k, band(a))          perturbation cost
                + retry · (1 − P_success(d(a)))                       task failure cost

P_success is 1 when the stopping distance d ≤ d_task, falls linearly to 0 at d_task + 0.3 m, and is 1 for tasks without
a proximity requirement. It is a *model of reaching the interaction distance*, not of completing a task.

**Risk cap (audit A3, programme L).**
- Options whose *predicted* expected strong perturbations exceed `risk_cap` are excluded.
- If none is left, the result is `NO_SAFE_ACTION`. The robot then executes `HOLD`: it stops at the stand-off point
  (0.9 m, beyond every type's reach), does not attempt the task, and learns nothing from IMU silence there.
- K used the farthest approach even when it exceeded the cap; `legacy=True` reproduces that.
- The cap constrains *predicted* risk under the robot's own model. Physical safety stays with the independent safety
  monitor and the validated controller.

**Policies.**
- `greedy`: argmin over the grid of C(a, 1 | b). This is J6's decision-theoretic mapping with frozen task values.
- `voi`: argmin over (a, w) of C(a, w | b) − VOI(a, w | b).
  - VOI is non-myopic (knowledge-gradient*): the largest per-meeting reduction in the expected cost of the
    following task decisions about this entity, if the same observation were repeated for m = 1, 2, 3 or 5 meetings,
    times H future meetings. The belief is relaxed by the expected gap.
  - A single observation that cannot by itself flip a confident belief still has value when a few would. A myopic
    rule gives such observations zero value, which recreates the avoidance trap.
  - It is computed by sampling from the robot's own predictive distribution: type ~ b, then categories ~ E.
- `OBSERVE` action (only with a camera): stop at proximity 0 with a 0.3 m stand-off, i.e. 0.9 m from the entity,
  beyond every type's reach (0.8 m).
  - The task fails for that encounter (retry cost), and no perturbation risk is possible.
  - Only camera codes are evidence there. IMU silence out of reach is never treated as evidence of safety.
- `uncertain`: greedy intent, w = 3 when the belief's normalised entropy exceeds `u_trigger`.
- Oracle (evaluation only): `greedy` on the canonical belief of the true type.

Nothing here reads simulator truth. Lack of evidence is never treated as evidence: emissions are conditioned on the
band the robot actually stopped in, and unobservable windows are skipped by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from apm_transfer.expression_map import GRID, STRONG, TimeModel, band_of, stop_distance

WINDOW_S = 4.0
WINDOW_CHOICES = (1, 2, 3)


@dataclass(frozen=True)
class Task:
    name: str
    d_task: float | None        # required stopping distance (m); None: no proximity requirement
    retry_s: float              # cost of a failed task attempt (s)
    price_s: float = 20.0       # cost of one strong perturbation (s): a safety stop and check
    risk_cap: float = 0.30      # max expected strong perturbations per encounter (safety budget)
    min_sep: float | None = None  # tasks that require keeping at least this distance (m)
    # APM v1.0: OBSERVE and HOLD stand off and never attempt the task, so they never complete it. K and L let them
    # complete tasks with no proximity requirement (waypoint, separation): standing at the start "succeeded" (L §E
    # artefact). True reproduces that.
    hold_completes: bool = False

    def p_reach(self, d: float) -> float:
        """Model of *reaching the interaction distance* from the stopping distance d (audit A4). It is not task
        completion; the measured task outcome is the interaction hold (`hold_completed`)."""
        if self.min_sep is not None and d < self.min_sep - 1e-9:
            return 0.0
        if self.d_task is None:
            return 1.0
        return float(np.clip(1.0 - (d - self.d_task) / 0.3, 0.0, 1.0))

    p_success = p_reach  # programme K's name, kept for reproducibility

    def hold_completed(self, arrived: bool, hold_max_dist: float | None) -> bool:
        """Measured interaction task: the robot arrived and stayed within the interaction zone (d_task + 0.05 m) for
        the whole first sensing window, despite any pushes. It applies only to tasks with a proximity requirement."""
        if self.d_task is None or not arrived or hold_max_dist is None:
            return False
        return hold_max_dist <= self.d_task + 0.05


@dataclass
class Emissions:
    """E[k, band, c] for the IMU blocks and optionally Ecam[k, c], split from a flattened type model.

    `observe_imu=False` (programme L, vision only): IMU categories are not observed. The IMU block still predicts
    risk, i.e. the strong perturbations an approach would cause, but it carries no information."""
    imu: np.ndarray
    cam: np.ndarray | None
    observe_imu: bool = True

    @classmethod
    def from_model(cls, E: np.ndarray, bands: int = 3, C: int = 8, camera_block: bool = False) -> Emissions:
        K = E.shape[0]
        blocks = E.reshape(K, -1, C)
        return cls(blocks[:, :bands, :], blocks[:, bands, :] if camera_block else None)

    def strong(self) -> np.ndarray:
        return self.imu[:, :, STRONG:].sum(2)  # [K, bands]


def relax(b: np.ndarray, pi: np.ndarray, lam: float, gap: float) -> np.ndarray:
    return pi + (b - pi) * math.exp(-lam * gap)


def cost(i: int, w: int, b: np.ndarray, tm: TimeModel, em: Emissions, task: Task) -> tuple[float, float]:
    """(task cost, expected strong perturbations) of grid intent i with w windows."""
    it = GRID[i]
    band = band_of(it.proximity)
    risk = w * float(b @ em.strong()[:, band])
    c = tm.predict(it) + WINDOW_S * (w - 1) + task.price_s * risk
    c += task.retry_s * (1.0 - task.p_success(stop_distance(it.proximity)))
    return c, risk


OBSERVE = -1
HOLD = -2           # executed when there is NO_SAFE_ACTION: stand off beyond reach, task not attempted
NO_SAFE_ACTION = HOLD
STANDOFF_M = 0.3
OUT_OF_REACH = 3  # pseudo-band: no IMU evidence


def best_greedy(b: np.ndarray, tm: TimeModel, em: Emissions, task: Task, legacy: bool = False) -> tuple[int, float]:
    """Cheapest grid intent within the risk cap; `(NO_SAFE_ACTION, hold cost)` if there is none."""
    best, best_c, fallback = -1, math.inf, None
    for i in range(len(GRID)):
        c, risk = cost(i, 1, b, tm, em, task)
        if risk <= task.risk_cap and c < best_c:
            best, best_c = i, c
        far = (GRID[i].proximity, -GRID[i].tempo)
        if fallback is None or far < fallback[0]:
            fallback = (far, i, c)
    if best < 0:
        if legacy:  # programme K: farthest approach even when it exceeds the cap
            assert fallback is not None
            return fallback[1], fallback[2]
        return NO_SAFE_ACTION, action_cost(HOLD, 1, b, tm, em, task)[0]
    return best, best_c


def action_cost(a: int, w: int, b: np.ndarray, tm: TimeModel, em: Emissions, task: Task) -> tuple[float, float]:
    """Cost of a grid intent (a >= 0), OBSERVE (-1) or HOLD (-2) with w windows. OBSERVE and HOLD stop beyond reach:
    no predicted perturbation, and the task is not achieved there."""
    if a >= 0:
        return cost(a, w, b, tm, em, task)
    it = GRID[2]  # proximity 0, tempo 1, no hesitation
    p_done = task.p_success(stop_distance(0.0) + STANDOFF_M) if task.hold_completes else 0.0
    c = tm.predict(it) + WINDOW_S * (w - 1) + task.retry_s * (1.0 - p_done)
    return c, 0.0


def posterior(b: np.ndarray, imu_cats: list[int], band: int, cam_cats: list[int], em: Emissions) -> np.ndarray:
    lp = np.log(np.clip(b, 1e-300, None))
    if band != OUT_OF_REACH and em.observe_imu:
        for c in imu_cats:
            lp = lp + np.log(em.imu[:, band, c])
    if em.cam is not None:
        for c in cam_cats:
            lp = lp + np.log(em.cam[:, c])
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def improvement(n: int, band: int, b: np.ndarray, pi: np.ndarray, tm: TimeModel, em: Emissions, task: Task,
                lam: float, gap: float, rng: np.random.Generator, samples: int = 48) -> float:
    """Expected reduction in the next task decision's cost after n windows observed at `band`."""
    now = best_greedy(relax(b, pi, lam, gap), tm, em, task)[1]
    K = len(b)
    after = 0.0
    for _ in range(samples):
        k = int(rng.choice(K, p=b))
        imu = ([] if band == OUT_OF_REACH or not em.observe_imu else
               [int(rng.choice(em.imu.shape[2], p=em.imu[k, band])) for _ in range(n)])
        cam = [int(rng.choice(em.cam.shape[1], p=em.cam[k])) for _ in range(n)] if em.cam is not None else []
        after += best_greedy(relax(posterior(b, imu, band, cam, em), pi, lam, gap), tm, em, task)[1]
    return max(0.0, now - after / samples)


def voi(band: int, w: int, b: np.ndarray, pi: np.ndarray, tm: TimeModel, em: Emissions, task: Task, lam: float,
        horizon: float, gap: float, rng: np.random.Generator, samples: int = 48) -> float:
    """Knowledge-gradient* value of w windows at `band` per meeting (see the module docstring)."""
    best = 0.0
    for m in (1, 2, 3, 5):
        if m > horizon:
            break
        best = max(best, improvement(m * w, band, b, pi, tm, em, task, lam, gap, rng, samples) / m)
    return horizon * best


def choose(policy: str, b: np.ndarray, pi: np.ndarray, tm: TimeModel, em: Emissions, task: Task, lam: float = 0.02,
           horizon: float = 5.0, gap: float = 3.0, u_trigger: float = 0.6,
           rng: np.random.Generator | None = None, legacy: bool = False) -> tuple[int, int, dict]:
    """Return (grid index, OBSERVE or HOLD, windows, diagnostics incl. "standoff" and "no_safe_action")."""
    def hold(i: int, c: float, extra: dict) -> tuple[int, int, dict]:
        return i, 1, {"cost": c, "standoff": STANDOFF_M, "no_safe_action": True, **extra}

    if policy in ("greedy", "oracle"):
        i, c = best_greedy(b, tm, em, task, legacy)
        if i == NO_SAFE_ACTION:
            return hold(i, c, {})
        return i, 1, {"cost": c, "standoff": 0.0}
    if policy == "uncertain":
        i, c = best_greedy(b, tm, em, task, legacy)
        h = -float(np.sum(b * np.log(np.clip(b, 1e-12, 1.0)))) / math.log(len(b))
        if i == NO_SAFE_ACTION:
            return hold(i, c, {"entropy": h})
        w = 3 if h > u_trigger else 1
        if w > 1 and not legacy and cost(i, w, b, tm, em, task)[1] > task.risk_cap:
            w = 1  # the cap is per encounter: K checked one window and then executed three (audit A3)
        return i, w, {"cost": c, "entropy": h, "standoff": 0.0}
    if policy == "voi":
        assert rng is not None
        cache: dict[tuple[int, int], float] = {}  # VOI depends on the action only through its band
        actions = list(range(len(GRID))) + ([OBSERVE] if em.cam is not None else [])
        best = None
        for a in actions:
            band = OUT_OF_REACH if a == OBSERVE else band_of(GRID[a].proximity)
            for w in WINDOW_CHOICES:
                c, risk = action_cost(a, w, b, tm, em, task)
                if risk > task.risk_cap:
                    continue
                if (band, w) not in cache:
                    cache[(band, w)] = voi(band, w, b, pi, tm, em, task, lam, horizon, gap, rng)
                v = cache[(band, w)]
                if best is None or c - v < best[0] - 1e-9:
                    best = (c - v, a, w, c, v)
        if best is None:
            i, c = best_greedy(b, tm, em, task, legacy)
            if i == NO_SAFE_ACTION:
                return hold(i, c, {"voi": 0.0})
            return i, 1, {"cost": c, "voi": 0.0, "standoff": 0.0}
        return best[1], best[2], {"cost": best[3], "voi": best[4],
                                  "standoff": STANDOFF_M if best[1] == OBSERVE else 0.0}
    raise ValueError(policy)
