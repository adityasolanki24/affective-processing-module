"""Writes the I1 and I2 system files (MASTER_PREREGISTRATION.md §3.4) from the fitted type models and tuned values.

    python i_systems.py --types configs/types_T [--lam-dense J] [--lam-sparse J] [--explore J]
        --out configs/I_systems.json

Values not yet tuned are filled with placeholders (λ = 1/356.35 s⁻¹, ν = 1, conf_max = 0.5, p_random = 0.1).
Tuning overrides them, so a file written before tuning is only valid for the tuning stage itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

A0 = {"kind": "regulated", "regulator": "ema", "params": {"tau": 0.621595, "k": 0.449887, "c": 0.017277},
      "lam_c": 3.946054, "lam_s": 0.993398}
G2B_MEM = {"model": "M1t", "params": {"tau_s": 356.352646}, "evidence": "exposed"}
PLACEHOLDER_LAM = 1 / 356.352646


def ka(explore: dict | None = None) -> dict:
    return {"keep_m": 3.0, "kappa": 1.5, "approach_only": False, "explore": explore or {"mode": "none"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", required=True, help="path stem written by i_fit.py, relative to the repo root")
    ap.add_argument("--lam-dense", default="")
    ap.add_argument("--lam-sparse", default="")
    ap.add_argument("--explore", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sel = lambda p, key="selected": json.loads(Path(p).read_text())[key] if p else None  # noqa: E731
    lam_d = sel(a.lam_dense) or PLACEHOLDER_LAM
    lam_s = sel(a.lam_sparse) or PLACEHOLDER_LAM
    ex = json.loads(Path(a.explore).read_text()) if a.explore else None
    nu = ex["nu"]["selected"] if ex else 1.0
    conf = ex["conf_max"]["selected"] if ex else 0.5
    p_rand = ex["p_random"]["selected"] if ex else 0.1
    dense, sparse = f"{a.types}_dense.json", f"{a.types}_sparse_K2.json"
    R = {"kind": "revising", "type_model": dense, "lam": lam_d}
    i1 = {"A0": {**A0, "memory": {"model": "M0", "evidence": "all"}},
          "G2b": {**A0, "memory": {**G2B_MEM, "probe": {"mode": "uncertainty", "budget": 2, "conf_max": 0.5}}},
          "R_sparse": {"kind": "revising", "type_model": sparse, "lam": lam_s},
          "R_static": {**R, "lam": 0.0},
          "R_dense": R,
          "Oracle": {"kind": "oracle"}}
    i2 = {"KA_A0": {**A0, "memory": {"model": "M0", "evidence": "all"}, "keep_away": ka()},
          "KA_G2b": {**A0, "memory": G2B_MEM, "keep_away": ka({"mode": "uncertainty", "budget": 2, "conf_max": 0.5})},
          "KA_R": {**R, "keep_away": ka()},
          "KA_R_voi": {**R, "keep_away": ka({"mode": "voi", "budget": 2, "nu": nu})},
          "KA_R_unc": {**R, "keep_away": ka({"mode": "uncertainty", "budget": 2, "conf_max": conf})},
          "KA_R_rand": {**R, "keep_away": ka({"mode": "random", "budget": 2, "p_random": p_rand, "seed": 0})},
          "KA_R_ts": {**R, "explore": {"mode": "thompson", "seed": 0}, "keep_away": ka({"mode": "thompson"})},
          "KA_Oracle": {"kind": "oracle", "keep_away": ka()}}
    a.out.write_text(json.dumps({"I1": {"systems": i1}, "I2": {"systems": i2},
                                 "tuned": {"lam_dense": lam_d, "lam_sparse": lam_s, "nu": nu, "conf_max": conf,
                                           "p_random": p_rand, "final": bool(a.lam_dense and a.lam_sparse and ex)}},
                                indent=1), encoding="utf-8")
    for k in ("I1", "I2"):  # the format i_run.py reads
        a.out.with_name(a.out.stem + f"_{k}.json").write_text(
            json.dumps({"systems": i1 if k == "I1" else i2}, indent=1), encoding="utf-8")
    print("wrote", a.out, "final" if a.lam_dense and a.lam_sparse and ex else "(placeholders for untuned values)")


if __name__ == "__main__":
    main()
