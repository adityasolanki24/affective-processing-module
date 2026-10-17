"""Writes configs/G1_selected.json from the tuning table and the R5 ES result by the preregistered rule
(prereg/G1_ADDENDUM.md): per family the configuration with the highest mean validation J; R_best = the family among
R2–R5 with the highest validation J. R0 = programme F's tuned B1 (λc 1, λs 2), untuned. B0 = always GO.

    python scripts/g1_select.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
KEEP = ("kind", "regulator", "params", "mode", "lam_c", "lam_s", "k", "k_t", "tau", "c")


def main() -> None:
    tab = json.loads((HERE / "results" / "G1" / "tuning_table.json").read_text())
    es = json.loads((HERE / "results" / "G1" / "es_R5.json").read_text())["selected"]
    rows = tab["table"] + [es | {"family": "R5"}]
    best = {}
    for row in rows:
        f = row["family"]
        if f not in best or row["J_mean"] > best[f]["J_mean"]:
            best[f] = row
    simple = max(("R2", "R3", "R4", "R5"), key=lambda f: best[f]["J_mean"])
    systems = {"B0": {"kind": "b0"}, "R0": {"kind": "regulated", "regulator": "static", "lam_c": 1.0, "lam_s": 2.0}}
    for f in sorted(best):
        systems[f] = {k: best[f][k] for k in KEEP if k in best[f]} | {"config_id": best[f]["config_id"]}
    out = {"rule": "max mean validation J per family (G1-V 0..N-1, 10 episodes); best_simple = argmax over R2-R5",
           "best_simple": simple, "validation": {f: {k: best[f][k] for k in ("config_id", "J_mean", "J_se",
                                                                              "collision_rate", "completion")}
                                                 for f in sorted(best)},
           "systems": systems}
    (HERE / "configs" / "G1_selected.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"best_simple": simple, "validation": out["validation"]}, indent=1))


if __name__ == "__main__":
    main()
