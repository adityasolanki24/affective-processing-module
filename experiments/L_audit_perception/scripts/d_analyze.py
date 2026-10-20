"""L protocol §D: analysis of the stress scenarios.

    python d_analyze.py --runs <D.jsonl> --types-model configs/LC_wheeled_vision_types.json --out results/D_stress.json

Per lifetime (changing identities B1/B2 only):
- **task cost per meeting:** approach time + 4 s × (windows − 1) + 20 s × strong perturbations
  + 12.6 s × (1 − measured success), measured as in LB4;
- **contacts per risky meeting**;
- **anticipation accuracy:** whether the belief *before* the meeting put the risk above the calm/pushy midpoint
  exactly when the entity was risky;
- **excess avoidance:** the share of safe meetings stopped at ≥ 0.85 m (hold or observe).

Paired bootstrap CIs over (history, index). The criteria are in `PROTOCOL.md` §D.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
K = REPO / "experiments" / "K_recovery"
sys.path.insert(0, str(K / "scripts"))
import k3b_analyze as K3B  # noqa: E402
import k_analyze as KA  # noqa: E402


def per_lifetime(rows: list[dict], spec: dict, r: np.ndarray, mid: float) -> dict:
    K3B.MEASURED = True
    b = [x for x in rows if x["entity"].startswith("B")]
    risky = [x for x in b if x["risky"]]
    safe = [x for x in b if not x["risky"]]
    return {"cost": float(np.mean([K3B.enc_cost(x, spec) for x in b])) if b else np.nan,
            "contacts_risky": float(np.mean([x["contacts"] for x in risky])) if risky else np.nan,
            "anticipation": float(np.mean([(float(np.asarray(x["b_before"]) @ r) > mid) == x["risky"] for x in b]))
            if b else np.nan,
            "excess_avoidance": float(np.mean([x["d_stop"] >= 0.85 for x in safe])) if safe else np.nan,
            "falls": int(sum(x["fell"] for x in rows))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--types-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mcfg = json.loads(a.types_model.read_text())
    r = np.asarray(mcfg["model"]["r"], float)
    anchors = {t: float(np.asarray(v) @ r) for t, v in mcfg["oracle"].items()}
    mid = (anchors["calm"] + anchors["pushy"]) / 2
    tasks = json.loads((K / "configs" / "K_tasks.json").read_text())["tasks"]
    d: dict = defaultdict(lambda: defaultdict(dict))
    for p in a.runs:
        for line in p.read_text(encoding="utf-8").splitlines():
            lt = json.loads(line)
            d[lt["scenario"]][lt["system"]][(lt["history"], lt["index"])] = per_lifetime(
                lt["rows"], tasks[lt["task"]], r, mid)
    out: dict = {"scenarios": {}}
    effective, m2_better, m2_worse, m3_better, m3_worse = [], [], [], [], []
    m1_fail, m2_fail = [], []
    for scn, sy in d.items():
        o: dict = {"systems": {s: {m: float(np.nanmean([v[m] for v in lts.values()])) for m in
                                   ("cost", "contacts_risky", "anticipation", "excess_avoidance")}
                               | {"falls": int(sum(v["falls"] for v in lts.values())), "lifetimes": len(lts)}
                               for s, lts in sy.items()}}

        def diff(x: str, y: str, m: str, sy: dict = sy) -> dict:
            keys = sorted(set(sy.get(x, {})) & set(sy.get(y, {})))
            return KA.boot([sy[x][k][m] - sy[y][k][m] for k in keys])
        o["M1_minus_M0"] = {m: diff("M1_bayes", "M0_prior", m) for m in ("cost", "contacts_risky")}
        o["M2_minus_M0"] = {m: diff("M2_leaky", "M0_prior", m) for m in ("cost", "contacts_risky")}
        o["M2_minus_M1"] = {m: diff("M2_leaky", "M1_bayes", m) for m in ("cost", "contacts_risky", "anticipation")}
        o["M3_minus_M1"] = {m: diff("M3_context", "M1_bayes", m) for m in ("cost", "contacts_risky", "anticipation")}
        o["M1_minus_ORACLE"] = {"cost": diff("M1_bayes", "ORACLE", "cost")}

        def eff(x: dict) -> bool:
            return (x["cost"]["ci95"][1] is not None and x["cost"]["ci95"][1] < 0
                    and x["contacts_risky"]["ci95"][1] is not None and x["contacts_risky"]["ci95"][1] <= 0.05)
        o["D1_effective_M1"] = eff(o["M1_minus_M0"])
        o["D1_effective_M2"] = eff(o["M2_minus_M0"])
        (effective if o["D1_effective_M1"] else m1_fail).append(scn)
        if not o["D1_effective_M2"]:
            m2_fail.append(scn)
        c2, c3 = o["M2_minus_M1"]["cost"]["ci95"], o["M3_minus_M1"]["cost"]["ci95"]
        if c2[1] is not None and c2[1] < 0:
            m2_better.append(scn)
        if c2[0] is not None and c2[0] > 0:
            m2_worse.append(scn)
        if c3[1] is not None and c3[1] < 0:
            m3_better.append(scn)
        if c3[0] is not None and c3[0] > 0:
            m3_worse.append(scn)
        out["scenarios"][scn] = o
    n = len(d)
    out["summary"] = {
        "scenarios": n, "M1_effective": effective, "M1_not_effective": m1_fail,
        "answer_effective_under_change": ("yes" if len(effective) >= 12 else "partly" if len(effective) >= 8
                                          else "no") if n == 15 else f"{len(effective)} of {n}",
        "M2_better_than_M1": m2_better, "M2_worse_than_M1": m2_worse,
        "M2_replaces_M1": len(m2_better) >= 3 and not m2_worse,
        "M3_better_than_M1": m3_better, "M3_worse_than_M1": m3_worse,
        "M3_recommended_as_context_extension": "context" in m3_better and len(m3_worse) <= 1,
        "predictive_model_justified": len(set(m1_fail) & set(m2_fail)) >= 2}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for scn, o in out["scenarios"].items():
        s = o["systems"]

        def g(sys_: str, m: str, s: dict = s) -> float:
            return s.get(sys_, {}).get(m, np.nan)
        print(f"{scn:16} cost M0 {g('M0_prior', 'cost'):5.1f} M1 {g('M1_bayes', 'cost'):5.1f} "
              f"M2 {g('M2_leaky', 'cost'):5.1f} M3 {g('M3_context', 'cost'):5.1f} OR {g('ORACLE', 'cost'):5.1f} | "
              f"contacts M0 {g('M0_prior', 'contacts_risky'):.2f} M1 {g('M1_bayes', 'contacts_risky'):.2f} | "
              f"antic M1 {g('M1_bayes', 'anticipation'):.2f} M2 {g('M2_leaky', 'anticipation'):.2f} | "
              f"eff M1 {o['D1_effective_M1']}")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
