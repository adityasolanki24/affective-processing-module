"""J4 analysis: experience-dependent, revisable memory on NERVA (MASTER_PREREGISTRATION §4.2, J4 addendum).

    python j4_analyze.py calibrate --runs <V lifetimes.jsonl> --out configs/J4_calibration.json
    python j4_analyze.py test --runs <X lifetimes.jsonl> --calibration configs/J4_calibration.json --out <json>

Probe columns (j4_lifetimes): 0 = B1 plain, 1 = A1 plain, 2 = B1 marked, 3 = B3 plain (novel). Each holds
[p, u, g, caution]. Checkpoints come every 6 encounters (8 per lifetime). The switch in change_B/late_B is at encounter
24, i.e. after checkpoint index 3.

Per memory:
- **history effect:** end-of-lifetime B1 p and caution, risky_B − safe (paired, bootstrap); history classification
  AUROC from end B1 p;
- **generalisation:** B3 (novel B-kind) p, risky_B − safe;
- **revision:** change_B (pushy → calm) and late_B (calm → pushy). Decisive crossing of the calibrated midpoint
  between μ_pushy and μ_calm (from V), beyond the midpoint of θ and the post-switch μ, for 2 consecutive checkpoints.
  Kaplan–Meier median in checkpoint blocks of 6 encounters (horizon 4);
- **false attribution:** zone_B. The marked zone, not the entity, is risky, so B1 plain p in zone_B minus safe,
  normalised by the risky_B − safe effect;
- **restart:** maximum |difference| between `type_kind` and `type_kind_restart` over all probes and checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from apm.learning.evaluation import auroc  # noqa: E402
from apm_select.survival import km_bootstrap  # noqa: E402

MEMS = ("H_A0", "count", "type", "type_kind")
SWITCH_CP, HORIZON = 4, 4  # first post-switch checkpoint index; checkpoints after the switch


def load(path: Path) -> dict:
    d: dict = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d[r["history"]][r["index"]] = r
    return d


def end(r: dict, mem: str, probe: int = 0, col: int = 0) -> float:
    return r["probes"][mem][-1][probe][col]


def paired_diff(a: list[float], b: list[float], n_boot: int = 5000) -> dict:
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"estimate": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def calibrate(data: dict) -> dict:
    out = {}
    for m in MEMS:
        mu_p = float(np.mean([end(r, m) for r in data["risky_B"].values()]))
        mu_c = float(np.mean([end(r, m) for r in data["safe"].values()]))
        out[m] = {"mu_pushy": mu_p, "mu_calm": mu_c, "threshold": (mu_p + mu_c) / 2, "gap": mu_p - mu_c}
    return out


def revision(rows: list[dict], mem: str, cal: dict, to: str) -> dict:
    mu_post = cal["mu_calm"] if to == "calm" else cal["mu_pushy"]
    deep = (cal["threshold"] + mu_post) / 2
    sign = 1.0 if mu_post >= cal["threshold"] else -1.0
    groups = []
    for r in rows:
        traj = [cp[0][0] for cp in r["probes"][mem]]
        pre = traj[SWITCH_CP - 1]
        pre_side = (1.0 if cal["gap"] >= 0 else -1.0) * (pre - cal["threshold"]) > 0  # True = pushy side
        if pre_side != (to == "calm"):
            groups.append([])  # not established before the switch
            continue
        t = None
        for j in range(SWITCH_CP, len(traj)):
            ok = sign * (traj[j] - deep) > 0
            nxt = j == len(traj) - 1 or sign * (traj[j + 1] - deep) > 0
            if ok and nxt:
                t = j - SWITCH_CP + 1
                break
        groups.append([(float(t), True)] if t is not None else [(float(HORIZON), False)])
    res = km_bootstrap(groups, HORIZON, 2000)
    res["established"] = sum(len(g) for g in groups)
    return res


def cmd_test(a) -> None:
    data = load(a.runs)
    cal = json.loads(a.calibration.read_text())
    keys = sorted(set(data["risky_B"]) & set(data["safe"]))
    res: dict = {"memories": {}}
    for m in MEMS:
        c = cal[m]
        risky = [end(data["risky_B"][i], m) for i in keys]
        safe = [end(data["safe"][i], m) for i in keys]
        eff = paired_diff(risky, safe)
        zone = paired_diff([end(data["zone_B"][i], m) for i in keys], safe)
        s = {"history_effect_p": eff,
             "history_effect_caution": paired_diff([end(data["risky_B"][i], m, 0, 3) for i in keys],
                                                   [end(data["safe"][i], m, 0, 3) for i in keys]),
             "history_auroc": auroc(risky + safe, [True] * len(risky) + [False] * len(safe)),
             "generalisation_B3": paired_diff([end(data["risky_B"][i], m, 3) for i in keys],
                                             [end(data["safe"][i], m, 3) for i in keys]),
             "revision_change_B": revision(list(data["change_B"].values()), m, c, "calm"),
             "revision_late_B": revision(list(data["late_B"].values()), m, c, "pushy"),
             "false_attribution_zone_B": zone,
             "false_attribution_normalised": zone["estimate"] / eff["estimate"] if eff["estimate"] else None,
             "risky_A_specificity_B1": paired_diff([end(data["risky_A"][i], m) for i in keys], safe)}
        res["memories"][m] = s
    diffs: list[float] = []
    for h in data:
        for r in data[h].values():
            for cp_a, cp_b in zip(r["probes"]["type_kind"], r["probes"]["type_kind_restart"], strict=True):
                for pa, pb in zip(cp_a, cp_b, strict=True):
                    diffs += [abs(x - y) for x, y in zip(pa, pb, strict=True)]
    res["restart_max_abs_diff"] = float(max(diffs))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for m, s in res["memories"].items():
        print(f"{m:10} effect p {s['history_effect_p']['estimate']:+.3f} caution "
              f"{s['history_effect_caution']['estimate']:+.3f} AUROC {s['history_auroc']:.2f} B3 "
              f"{s['generalisation_B3']['estimate']:+.3f} rev change {s['revision_change_B']['median']} late "
              f"{s['revision_late_B']['median']} zone-norm {s['false_attribution_normalised']}")
    print("restart max |diff|", res["restart_max_abs_diff"])


def cmd_calibrate(a) -> None:
    cal = calibrate(load(a.runs))
    a.out.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print({m: (round(c["threshold"], 4), round(c["gap"], 4)) for m, c in cal.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "test"))
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cmd_calibrate(a) if a.mode == "calibrate" else cmd_test(a)


if __name__ == "__main__":
    main()
