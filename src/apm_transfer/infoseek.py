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
a proximity requirement. **Risk cap:** options whose expected strong perturbations exceed `risk_cap` are excluded; if
none is left, the farthest, shortest option is used.

**Policies.**
- `greedy`: argmin over the grid of C(a, 1 | b). This is J6's decision-theoretic mapping with frozen task values.
- `voi`: argmin over (a, w) of C(a, w | b) − VOI(a, w | b).
  - VOI is the expected reduction in the cost of the *next* decision about this entity (H future meetings, belief
    relaxed by the expected gap) from what w windows at band(a) would show.
  - It is computed by sampling from the robot's own predictive distribution: type ~ b, then categories ~ E.
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

    def p_success(self, d: float) -> float:
        if self.min_sep is not None and d < self.min_sep - 1e-9:
            return 0.0
        if self.d_task is None:
            return 1.0
        return float(np.clip(1.0 - (d - self.d_task) / 0.3, 0.0, 1.0))


@dataclass
class Emissions:
    """E[k, band, c] for the IMU blocks and optionally Ecam[k, c], split from a flattened type model."""
    imu: np.ndarray
    cam: np.ndarray | None

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


def best_greedy(b: np.ndarray, tm: TimeModel, em: Emissions, task: Task) -> tuple[int, float]:
    best, best_c, fallback = -1, math.inf, None
    for i in range(len(GRID)):
        c, risk = cost(i, 1, b, tm, em, task)
        if risk <= task.risk_cap and c < best_c:
            best, best_c = i, c
        far = (GRID[i].proximity, -GRID[i].tempo)
        if fallback is None or far < fallback[0]:
            fallback = (far, i, c)
    if best < 0:
        assert fallback is not None
        return fallback[1], fallback[2]
    return best, best_c


def posterior(b: np.ndarray, imu_cats: list[int], band: int, cam_cats: list[int], em: Emissions) -> np.ndarray:
    lp = np.log(np.clip(b, 1e-300, None))
    for c in imu_cats:
        lp = lp + np.log(em.imu[:, band, c])
    if em.cam is not None:
        for c in cam_cats:
            lp = lp + np.log(em.cam[:, c])
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def voi(i: int, w: int, b: np.ndarray, pi: np.ndarray, tm: TimeModel, em: Emissions, task: Task, lam: float,
        horizon: float, gap: float, rng: np.random.Generator, samples: int = 48) -> float:
    """Expected reduction in the next decision's cost from observing w windows at band(i), times `horizon`."""
    band = band_of(GRID[i].proximity)
    now = best_greedy(relax(b, pi, lam, gap), tm, em, task)[1]
    K = len(b)
    after = 0.0
    for _ in range(samples):
        k = int(rng.choice(K, p=b))
        imu = [int(rng.choice(em.imu.shape[2], p=em.imu[k, band])) for _ in range(w)]
        cam = [int(rng.choice(em.cam.shape[1], p=em.cam[k])) for _ in range(w)] if em.cam is not None else []
        post = posterior(b, imu, band, cam, em)
        after += best_greedy(relax(post, pi, lam, gap), tm, em, task)[1]
    return horizon * max(0.0, now - after / samples)


def choose(policy: str, b: np.ndarray, pi: np.ndarray, tm: TimeModel, em: Emissions, task: Task, lam: float = 0.02,
           horizon: float = 5.0, gap: float = 3.0, u_trigger: float = 0.6,
           rng: np.random.Generator | None = None) -> tuple[int, int, dict]:
    """Return (grid index, windows, diagnostics)."""
    if policy in ("greedy", "oracle"):
        i, c = best_greedy(b, tm, em, task)
        return i, 1, {"cost": c}
    if policy == "uncertain":
        i, c = best_greedy(b, tm, em, task)
        h = -float(np.sum(b * np.log(np.clip(b, 1e-12, 1.0)))) / math.log(len(b))
        w = 3 if h > u_trigger else 1
        return i, w, {"cost": c, "entropy": h}
    if policy == "voi":
        assert rng is not None
        cache: dict[tuple[int, int], float] = {}  # VOI depends on the intent only through its band
        best = None
        for i in range(len(GRID)):
            band = band_of(GRID[i].proximity)
            for w in WINDOW_CHOICES:
                c, risk = cost(i, w, b, tm, em, task)
                if risk > task.risk_cap:
                    continue
                if (band, w) not in cache:
                    cache[(band, w)] = voi(i, w, b, pi, tm, em, task, lam, horizon, gap, rng)
                v = cache[(band, w)]
                if best is None or c - v < best[0]:
                    best = (c - v, i, w, c, v)
        if best is None:
            i, c = best_greedy(b, tm, em, task)
            return i, 1, {"cost": c, "voi": 0.0}
        return best[1], best[2], {"cost": best[3], "voi": best[4]}
    raise ValueError(policy)
