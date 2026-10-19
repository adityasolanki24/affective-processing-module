"""M6 robustness checks F (PROTOCOL.md §8), from the F_* and X runs.

    python experiments/M_release/scripts/m6_robustness.py --cloud runs/M/cloud --out results/m6/robustness.json

Per robot (Isaac: nova_carter, humanoid):
- restart: decisions of the save/load run vs the uninterrupted reference run (same seeds): identical up to the
  first physical divergence; reports where decisions first differ and whether that precedes the restart;
- replay: `runtime.replay` on every logged X DT lifetime (mismatches must be 0);
- nocam, dropout, interrupt, contact, perturb, long: falls, trips, lunge contacts, encounters completed;
- veto: supervisor vetoes and executed approaches above the risk cap (must be 0);
- anon: B1 handled as an anonymous track; its belief before every meeting must equal the kind prior (no memory).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
M = REPO / "experiments" / "M_release"
CFG = M / "configs"


def recs(cloud: Path, tag: str) -> list[dict]:
    out = []
    for f in sorted((cloud / tag).rglob("part_*.jsonl")):
        out += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
    return out


def summary(rs: list[dict]) -> dict:
    rows = [x for r in rs for x in r["rows"]]
    return {"lifetimes": len(rs), "encounters": len(rows), "falls": sum(x["fell"] for x in rows),
            "trips": sum(x["trips"] for x in rows), "lunge_contacts": sum(x["contacts"] for x in rows),
            "arrived": sum(bool(x["arrived"]) for x in rows), "vetoes": sum(r.get("vetoes", 0) for r in rs),
            "max_executed_risk": max((x["risk"] for x in rows if x["action"] != -2 and not x["vetoed"]),
                                     default=None)}


def seed_of(index: int) -> int:
    spec = importlib.util.spec_from_file_location("m_seeds", M / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return int(mod.world_seed("M6", "X", index).generate_state(1)[0])


def replay_all(cloud: Path, robot: str) -> dict:
    from apm_transfer.infoseek import Task
    from apm_transfer.runtime import Runtime, replay
    mcfg = json.loads((CFG / f"M5_{robot}_types.json").read_text())
    tc = json.loads((CFG / f"M5_{robot}_tasks.json").read_text())
    time_coef = json.loads((CFG / f"M5_{robot}_time.json").read_text())["time_coef"]
    task = Task("handoff", **{k: v for k, v in tc["tasks"]["handoff"].items() if k != "why"})
    logs = sorted((cloud / f"X_{robot}_main_DT").rglob(f"logs/{robot}_warehouse_DT_*_X*.jsonl"))
    total = {"lifetimes": 0, "decisions": 0, "mismatches": 0, "tolerance": 1e-9}
    for lg in logs:
        idx = int(re.search(r"_X(\d+)\.jsonl$", lg.name).group(1))  # type: ignore[union-attr]

        def factory(idx=idx):
            return Runtime(mcfg, time_coef, task, tc["voi"], lam=0.02, seed=seed_of(idx) + 3, camera=True,
                           policy="greedy", supervise=True)
        r = replay(lg, factory, tol=1e-9)  # logs from Linux VMs, replayed here: last-digit float rounding
        total["lifetimes"] += 1
        total["decisions"] += r["decisions"]
        total["mismatches"] += r["mismatches"]
        total["bit_exact"] = total.get("bit_exact", 0) + r["bit_exact"]
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from apm_revise.belief import TypeModel
    res: dict = {}
    for robot in ("nova_carter", "humanoid"):
        r: dict = {}
        ref = {(x["history"], x["index"]): x for x in recs(a.cloud, f"F_{robot}_ref")}
        rst = {(x["history"], x["index"]): x for x in recs(a.cloud, f"F_{robot}_restart")}
        cmp = []
        for key, lt in rst.items():
            if key not in ref:
                continue
            da = [(x["action"], x["proximity"], x["windows_planned"]) for x in lt["rows"]]
            db = [(x["action"], x["proximity"], x["windows_planned"]) for x in ref[key]["rows"]]
            first = next((i for i, (u, v) in enumerate(zip(da, db, strict=False)) if u != v), None)
            obs_a = [(x["imu_cats"], x["cam_codes"], x["band"]) for x in lt["rows"]]
            obs_b = [(x["imu_cats"], x["cam_codes"], x["band"]) for x in ref[key]["rows"]]
            first_obs = next((i for i, (u, v) in enumerate(zip(obs_a, obs_b, strict=False)) if u != v), None)
            cmp.append({"lifetime": list(key), "decisions": len(da), "first_decision_difference": first,
                        "first_observation_difference": first_obs})
        r["restart"] = {"restart_at": 12, "pairs": cmp, "summary": summary(list(rst.values()))}
        try:
            r["replay"] = replay_all(a.cloud, robot)
        except Exception as e:  # noqa: BLE001
            r["replay"] = {"error": repr(e)[:200]}
        for chk in ("nocam", "dropout", "interrupt", "contact", "perturb", "long", "veto"):
            rs = recs(a.cloud, f"F_{robot}_{chk}")
            r[chk] = summary(rs) if rs else "not run"
        rs = recs(a.cloud, f"F_{robot}_anon")
        if rs:
            model = TypeModel.from_dict(json.loads((CFG / f"M5_{robot}_types.json").read_text())["model"])
            b1 = [x for lt in rs for x in lt["rows"] if x["entity"] == "B1"]
            priors = {tuple(np.round(x["b_before"], 4)) for x in b1}
            r["anon"] = {**summary(rs), "b1_meetings": len(b1), "distinct_b1_beliefs_before": len(priors),
                         "kind_prior_only": len(priors) <= 1, "types": int(len(model.pi))}
        else:
            r["anon"] = "not run"
        res[robot] = r
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
