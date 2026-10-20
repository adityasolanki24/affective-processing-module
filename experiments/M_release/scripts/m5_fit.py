"""M5 fitting (PROTOCOL.md §2): each robot's own models from its exploration lifetimes (partition T).

    python experiments/M_release/scripts/m5_fit.py --robot nova_carter --runs <T dirs or jsonl ...>

Writes, in experiments/M_release/configs/:
- `M5_<robot>_types.json`: type model (EM, K by BIC; IMU blocks per distance band + camera block), via `k_fit.py`;
- `M5_<robot>_edges.json`: its own acc8 edges (pooled quantiles of its window peaks);
- `M5_<robot>_time.json`: its own time model (least squares of arrival time on intent);
- `M5_<robot>_tasks.json`: K's task values with `retry_s` = the mean predicted approach time over the intent grid
  (the frozen rule).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
CFG = REPO / "experiments" / "M_release" / "configs"
LABEL = {"nova_carter": "nova_carter", "humanoid": "t1_isaac", "nerva": "nerva_isaac"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=tuple(LABEL), required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    lines = []
    for p in a.runs:
        files = sorted(p.rglob("part_*.jsonl")) if p.is_dir() else [p]
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("robot") == a.robot and rec.get("partition") == "T" and rec.get("system") == "RANDINTENT":
                    lines.append(line)
    if not lines:
        sys.exit("no T exploration lifetimes for this robot")
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td) / "T.jsonl"
        runs.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = Path(td) / "types.json"
        r = subprocess.run([sys.executable, str(REPO / "experiments" / "K_recovery" / "scripts" / "k_fit.py"),
                            "--runs", str(runs), "--out", str(out), "--robot", LABEL[a.robot]],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)
        lab = LABEL[a.robot]
        CFG.mkdir(parents=True, exist_ok=True)
        meta = {"robot": a.robot, "lifetimes": len(lines), "source": "M5 partition T (RANDINTENT; risky_B, safe; "
                "warehouse and home)"}
        types = json.loads(out.read_text())
        (CFG / f"M5_{a.robot}_types.json").write_text(json.dumps({**types, "meta": meta}, indent=1))
        edges = json.loads((Path(td) / f"K_{lab}_edges.json").read_text())
        (CFG / f"M5_{a.robot}_edges.json").write_text(json.dumps({**edges, "meta": meta}, indent=1))
        time_m = json.loads((Path(td) / f"K_{lab}_time.json").read_text())
        (CFG / f"M5_{a.robot}_time.json").write_text(json.dumps({**time_m, "meta": meta}, indent=1))
    # task values: K's, with retry_s from this robot's own time model (K's frozen rule)
    from apm_transfer.expression_map import GRID, TimeModel
    tm = TimeModel(time_m["time_coef"])
    retry = float(np.mean([tm.predict(g) for g in GRID]))
    tasks = json.loads((REPO / "experiments" / "K_recovery" / "configs" / "K_tasks.json").read_text())
    for t in tasks["tasks"].values():
        t["retry_s"] = round(retry, 2)
    tasks["note"] = (f"APM v1.0, {a.robot}: K's task values; retry_s = this robot's mean predicted approach time "
                     f"over the intent grid ({retry:.2f} s, M5 time model, R2 {time_m['r2']:.3f}).")
    (CFG / f"M5_{a.robot}_tasks.json").write_text(json.dumps(tasks, indent=1))
    print(a.robot, "lifetimes", len(lines), "retry_s", round(retry, 2), "time R2", round(time_m["r2"], 3))


if __name__ == "__main__":
    main()
