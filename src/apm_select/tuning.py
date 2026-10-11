"""G0/G1 equal-budget tuning: every family gets exactly K configurations, drawn by a seeded Latin hypercube from a
preregistered space, evaluated on the same tuning lifetimes, selected by the same utility.

Space entries: ("log", lo, hi) log-uniform, ("lin", lo, hi) uniform. R5 (learned scalar) is not sampled here: it is
trained by evolution strategies with the same number of lifetime evaluations (scripts/g1_es.py).
"""
from __future__ import annotations

import math

import numpy as np

LAM = {"lam_c": ("log", 0.1, 10.0), "lam_s": ("lin", 0.0, 4.0)}
FAMILIES: dict[str, dict] = {
    "R1": {"kind": "regulated", "regulator": "static", "space": {**LAM}},
    "R2": {"kind": "regulated", "regulator": "ema",
           "space": {**LAM, "tau": ("log", 0.5, 60.0), "k": ("log", 0.25, 8.0), "c": ("lin", 0.0, 1.0)}},
    "R3": {"kind": "regulated", "regulator": "hysteresis",
           "space": {**LAM, "tau_up": ("log", 0.2, 10.0), "tau_down": ("log", 1.0, 120.0), "k": ("log", 0.25, 8.0),
                     "c": ("lin", 0.0, 1.0)}},
    "R4": {"kind": "regulated", "regulator": "bayes",
           "space": {**LAM, "tau_f": ("log", 10.0, 600.0), "z": ("lin", 0.0, 3.0), "k": ("log", 0.05, 4.0)}},
    "R6": {"kind": "b2v", "mode": "pad", "space": {**LAM, "k": ("log", 0.25, 8.0), "k_t": ("log", 0.05, 1.0)}},
    "R7": {"kind": "b2v", "mode": "appraisal", "space": {**LAM, "k": ("log", 0.25, 8.0), "k_t": ("log", 0.05, 1.0)}},
    "R8": {"kind": "b2v", "mode": "filter",
           "space": {**LAM, "tau": ("log", 0.5, 60.0), "k": ("log", 0.25, 8.0), "c": ("lin", 0.0, 1.0),
                     "k_t": ("log", 0.05, 1.0)}},
}
REGULATOR_PARAMS = {"static": (), "ema": ("tau", "k", "c"), "hysteresis": ("tau_up", "tau_down", "k", "c"),
                    "bayes": ("tau_f", "z", "k")}


def lhs(space: dict, k: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    names = sorted(space)
    cols = {}
    for n in names:
        u = (rng.permutation(k) + rng.random(k)) / k
        kind, lo, hi = space[n]
        cols[n] = np.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if kind == "log" else lo + u * (hi - lo)
    return [{n: float(round(cols[n][i], 6)) for n in names} for i in range(k)]


def to_spec(family: str, values: dict, config_id: int | str) -> dict:
    f = FAMILIES[family]
    spec = {"kind": f["kind"], "family": family, "config_id": f"{family}:{config_id}", "label": family,
            "lam_c": values["lam_c"], "lam_s": values["lam_s"]}
    if f["kind"] == "regulated":
        spec["regulator"] = f["regulator"]
        spec["params"] = {p: values[p] for p in REGULATOR_PARAMS[f["regulator"]]}
    else:
        spec |= {"mode": f["mode"], "k": values["k"], "k_t": values["k_t"], "tau": values.get("tau", 1.0),
                 "c": values.get("c", 0.0)}
    return spec


def family_configs(family: str, k: int, seed: int) -> list[dict]:
    """The K tuning specs of one family (seed offset by family so families do not share draws)."""
    offset = sorted(FAMILIES).index(family)
    return [to_spec(family, v, i) for i, v in enumerate(lhs(FAMILIES[family]["space"], k, seed + 1000 * offset))]
