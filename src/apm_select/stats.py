"""Programme G statistics: paired cluster bootstrap over lifetimes for any ratio-of-totals metric.

A metric is (numerator key, denominator key), e.g. collisions per minute = ("collisions", "minutes"), collisions per
encounter = ("person_collisions", "person_encounters"). For systems A and B evaluated on the same lifetimes, each
bootstrap draw re-weights lifetimes (multinomial, shared by both systems), recomputes both ratios of totals and their
relative ("rel": A/B − 1) and absolute ("abs": A − B) differences. Percentile 95% CIs; one-sided p-values; one-sided
95% bounds for non-inferiority (equivalent to a one-sided test at α = 0.05).

With several training seeds per system, `two_level=True` resamples seeds within each draw as well (arrays
[seeds, lifetimes]); a deterministic system has one row.
"""
from __future__ import annotations

import numpy as np

from apm_research.stats import holm  # noqa: F401  (re-exported)

N_BOOT = 10_000


def _w(n_boot: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)])


def _k(s: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    if s == 1:
        return np.zeros((n_boot + 1, 1), int)
    return np.vstack([np.arange(s), rng.integers(0, s, size=(n_boot, s))])


class Paired:
    """Shared bootstrap weights for one comparison A vs B (dicts of per-lifetime totals; 1-D or [seeds, n])."""

    def __init__(self, a: dict, b: dict, n_boot: int = N_BOOT, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.a = {k: np.atleast_2d(np.asarray(v, float)) for k, v in a.items()}
        self.b = {k: np.atleast_2d(np.asarray(v, float)) for k, v in b.items()}
        n = next(iter(self.a.values())).shape[1]
        assert all(v.shape[1] == n for v in [*self.a.values(), *self.b.values()]), "lifetimes must be aligned"
        self.n = n
        self.w = _w(n_boot, n, rng)
        sa = next(iter(self.a.values())).shape[0]
        sb = next(iter(self.b.values())).shape[0]
        self.ka, self.kb = _k(sa, n_boot, rng), _k(sb, n_boot, rng)

    def _tot(self, d: dict, k: np.ndarray, key: str) -> np.ndarray:
        return np.einsum("bn,bn->b", self.w, d[key][k].mean(1))

    def ratio(self, side: str, num: str, den: str) -> np.ndarray:
        d, k = (self.a, self.ka) if side == "a" else (self.b, self.kb)
        return self._tot(d, k, num) / np.maximum(1e-12, self._tot(d, k, den))

    def compare(self, num: str, den: str) -> dict:
        ra, rb = self.ratio("a", num, den), self.ratio("b", num, den)
        out = {"a": float(ra[0]), "b": float(rb[0])}
        for name, draws in (("rel", ra / np.maximum(1e-12, rb) - 1.0), ("abs", ra - rb)):
            d = draws[1:]
            lo, hi = np.percentile(d, [2.5, 97.5])
            lo90, hi90 = np.percentile(d, [5.0, 95.0])
            out[name] = {"estimate": float(draws[0]), "ci95": [float(lo), float(hi)], "upper95_1s": float(hi90),
                         "lower95_1s": float(lo90), "se": float(d.std()), "p_not_lower": float(np.mean(d >= 0)),
                         "p_not_higher": float(np.mean(d <= 0)), "_draws": d}
        return out


METRICS = {
    "collision_rate": ("collisions", "minutes"),
    "severe_rate": ("severe", "minutes"),
    "completion": ("goals_completed", "goals_issued"),
    "idle_fraction": ("idle", "periods"),
    "collisions_per_100m": ("collisions_x100", "distance"),
    "person_collisions_per_encounter": ("person_collisions", "person_encounters"),
    "close_encounter_fraction": ("close_encounters", "person_encounters"),
    "hesitation_fraction": ("hesitant", "decisions"),
    "switch_rate": ("switches", "decisions"),
    "oscillation_rate": ("oscillations", "decisions"),
    "distance_per_min": ("distance", "minutes"),
    "interventions_per_min": ("interventions", "minutes"),
    "careless_collisions_per_min": ("careless_collisions", "minutes"),
    "decide_us": ("decide_us", "decisions"),
    "compute_us_per_decision": ("compute_us", "decisions"),
}


def compare_systems(a: dict, b: dict, metrics=METRICS, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def extra(d: dict) -> dict:
        d = {**d, "collisions_x100": np.asarray(d["collisions"], float) * 100.0}
        if "decide_us" in d and "observe_us" in d:
            d["compute_us"] = np.asarray(d["decide_us"], float) + np.asarray(d["observe_us"], float)
        return d
    a, b = extra(a), extra(b)
    p = Paired(a, b, n_boot, seed)
    return {name: p.compare(num, den) for name, (num, den) in metrics.items() if num in a and den in a}


def strip(obj):
    """Drop bootstrap draws (keys starting with "_") before writing results."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def p_beyond(entry: dict, threshold: float, direction: str) -> float:
    """Bootstrap one-sided p-value that the difference is not beyond `threshold`: direction "below" tests
    H1: diff < threshold (p = P(draw ≥ threshold)); "above" tests H1: diff > threshold."""
    d = entry["_draws"]
    return float(np.mean(d >= threshold)) if direction == "below" else float(np.mean(d <= threshold))


def e2_rule(c: dict, p_a: float, p_b: float, alpha: float = 0.025) -> dict:
    """Programme F's superiority rule (MASTER_PREREGISTRATION §2 E2), on a `compare_systems` result."""
    col, comp, idle = c["collision_rate"]["rel"], c["completion"]["abs"], c["idle_fraction"]["abs"]
    a = col["estimate"] <= -0.10 and p_a < alpha and comp["ci95"][0] >= -0.03
    b = comp["estimate"] >= 0.05 and p_b < alpha and col["ci95"][1] <= 0.05
    no_freeze = idle["estimate"] <= 0.05
    return {"criterion_A": bool(a), "criterion_B": bool(b), "no_degenerate_freezing": bool(no_freeze),
            "beats": bool((a or b) and no_freeze)}


def non_inferior(c: dict, collision_margin: float, completion_margin: float, alpha_bound: str = "upper95_1s") -> dict:
    """A is non-inferior to B iff the one-sided 95% upper bound of the relative collision-rate difference is below
    `collision_margin` AND the one-sided 95% lower bound of the completion difference is above −completion_margin."""
    col, comp = c["collision_rate"]["rel"], c["completion"]["abs"]
    ok_col = col[alpha_bound] < collision_margin
    ok_comp = comp["lower95_1s"] > -completion_margin
    p = max(p_beyond(col, collision_margin, "below"), p_beyond(comp, -completion_margin, "above"))
    return {"collision_ok": bool(ok_col), "completion_ok": bool(ok_comp), "non_inferior": bool(ok_col and ok_comp),
            "collision_upper95": col[alpha_bound], "completion_lower95": comp["lower95_1s"], "p": p}
