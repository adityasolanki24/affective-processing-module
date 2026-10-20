"""Analysis for the APM v1.0 perception study (PROTOCOL.md §5). It runs only on a real response file collected
under ethics approval. It never generates, imputes or simulates ratings.

    python experiments/M_release/human_study/analyze.py --responses responses.csv --key key.json --approval REF

The response file needs the columns participant, left_id, right_id, choice (left/right), descriptor_left,
descriptor_right, confidence, attention (0/1) and seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--approval", required=True, help="the ethics approval or exemption reference")
    a = ap.parse_args()
    if not a.responses.exists():
        sys.exit("no response file: this study has not been run (PROTOCOL.md §1)")
    rows = list(csv.DictReader(a.responses.open(encoding="utf-8")))
    if not rows:
        sys.exit("empty response file")
    key = json.loads(a.key.read_text())
    from scipy.stats import binomtest
    by_robot: dict[str, list[int]] = {}
    for r in rows:
        if r.get("attention") == "0":
            continue
        left, right = key[r["left_id"]], key[r["right_id"]]
        if left["robot"] != right["robot"] or left["condition"].startswith("pushed") == right["condition"].startswith(
                "pushed"):
            continue  # attention pairs and malformed pairs carry no history judgement
        pushed_left = left["condition"].startswith("pushed")
        correct = (r["choice"] == "left") == pushed_left
        by_robot.setdefault(left["robot"], []).append(int(correct))
    out = {"approval": a.approval, "per_robot": {}}
    for robot, c in by_robot.items():
        t = binomtest(sum(c), len(c), 0.5, alternative="greater")
        out["per_robot"][robot] = {"n": len(c), "correct": sum(c) / len(c), "p_binomial": t.pvalue}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
