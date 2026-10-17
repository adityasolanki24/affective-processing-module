"""Writes configs/G2_test_systems.json from configs/G2_selected.json (prereg/G2_ADDENDUM.md): every validated memory,
the Oracle reference and, if a candidate M* was selected, its evidence and probing variants.

    python scripts/g2_systems.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROBE_U = {"mode": "uncertainty", "budget": 2, "conf_max": 0.5}
PROBE_R = {"mode": "random", "budget": 2, "p_random": 0.1, "seed": 0}


def main() -> None:
    sel = json.loads((HERE / "configs" / "G2_selected.json").read_text())
    reg, systems = sel["regulator"], dict(sel["systems"])
    systems["Oracle"] = {**reg, "memory": {"model": "oracle"}}
    star = sel["candidate"]
    if star:
        mem = systems[star]["memory"]
        systems[f"{star}_all"] = {**reg, "memory": {**mem, "evidence": "all"}}
        systems[f"{star}_probeU"] = {**reg, "memory": {**mem, "probe": PROBE_U}}
        systems[f"{star}_probeR"] = {**reg, "memory": {**mem, "probe": PROBE_R}}
    out = {"candidate": star, "systems": systems}
    (HERE / "configs" / "G2_test_systems.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(star, sorted(systems))


if __name__ == "__main__":
    main()
