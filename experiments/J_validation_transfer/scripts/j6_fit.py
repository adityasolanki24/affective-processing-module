"""J6: fit the self-supervised time model, and train the GRU mapping, from RANDINTENT exploration (J6-T).

    python j6_fit.py --runs <T_explore.jsonl> --types-model <J5_types_distance.json> --price 10 --out-dir <dir>

1. **Time model.** Least squares of the robot's own arrival time on (1, proximity, tempo, hesitation,
   tempo·hesitation) over every exploration encounter → `J6_model.json` = {time_coef, price}. Sensitivity files are
   also written for prices 5 and 20.
2. **GRU mapping** (torch, CPU). For each entity in each exploration lifetime, the input sequence is that entity's
   own evidence history (`expression_map.gru_input`: acc8 category, distance band, detected; unobservable flag). The
   target at each meeting is `utility_intent(posterior before the meeting)`, i.e. what the explicit memory plus
   decision-theoretic mapping would choose.
   - Hidden 16; cross-entropy; Adam 3e-3; 60 epochs; lifetimes split 80/20 (seed 0); the best validation epoch is
     kept.
   - Weights are exported for the numpy `GruMapping` → `J6_gru.json` (with validation accuracy).

The GRU sees only raw evidence, never the explicit memory. It tests whether a recurrent state can replace explicit
memory plus mapping.

3. **Value mapping (DEVIATIONS D3).** With price 10 and no proximity value, `utility_intent` chooses the same intent
   for every belief, so the GRU's targets are constant. `J6_model_v.json` adds `--v-price` and `--v-value` (the task's
   proximity value), and a second GRU is trained on that mapping's targets → `J6_gru_v.json`.
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
from apm_express.intent import ExpressionIntent  # noqa: E402
from apm_express.styles import NEUTRAL  # noqa: E402
from apm_revise.belief import TypeModel  # noqa: E402
from apm_transfer.expression_map import TimeModel, gru_input, strong_prob, utility_intent  # noqa: E402

MIN_QUIET = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--types-model", type=Path, required=True)
    ap.add_argument("--price", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--v-price", type=float, default=40.0)
    ap.add_argument("--v-value", type=float, default=5.0)
    a = ap.parse_args()
    lifetimes = [json.loads(x) for x in a.runs.read_text(encoding="utf-8").splitlines()]
    base = NEUTRAL.as_dict()
    intents, times = [], []
    for lt in lifetimes:
        for r in lt["rows"]:
            if r["t_arrive"] is not None:
                intents.append(ExpressionIntent(**{**base, "proximity": r["proximity"], "tempo": r["tempo"],
                                                   "hesitation": r["hesitation"]}))
                times.append(r["t_arrive"])
    tm = TimeModel.fit(intents, times)
    pred = np.array([tm.predict(i) for i in intents])
    r2 = 1 - np.sum((np.asarray(times) - pred) ** 2) / np.sum((np.asarray(times) - np.mean(times)) ** 2)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for price, name in ((a.price, "J6_model.json"), (5.0, "J6_model_price5.json"), (20.0, "J6_model_price20.json")):
        (a.out_dir / name).write_text(json.dumps({"time_coef": tm.coef, "price": price, "r2": float(r2),
                                                  "encounters": len(times)}, indent=1), encoding="utf-8")
    (a.out_dir / "J6_model_v.json").write_text(json.dumps({"time_coef": tm.coef, "price": a.v_price,
                                                         "value": a.v_value, "r2": float(r2),
                                                         "encounters": len(times)}, indent=1), encoding="utf-8")
    print("time model", np.round(tm.coef, 3), "R2", round(float(r2), 3), "n", len(times))
    model = TypeModel.from_dict(json.loads(a.types_model.read_text()))
    for price, value, name in ((a.price, 0.0, "J6_gru.json"), (a.v_price, a.v_value, "J6_gru_v.json")):
        train_gru(lifetimes, tm, model, price, value, a.out_dir / name)


def train_gru(lifetimes: list[dict], tm: TimeModel, model: TypeModel, price: float, value: float,
              out_path: Path) -> None:
    ps = strong_prob(model)
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    split = []
    rng = np.random.default_rng(0)
    for lt in lifetimes:
        val = rng.random() < 0.2
        per: dict = defaultdict(list)
        for r in lt["rows"]:
            per[r["entity"]].append(r)
        for rows in per.values():
            xs, ys = [np.zeros(13)], []
            for r in rows:
                ys.append(utility_intent(np.asarray(r["post"]), tm, ps, price, value)[0])
                obs = r.get("observable", True)
                xs.append(gru_input(r["category"] if obs else None, r["band"], r["detected"]))
            # input at step t = evidence from meetings before t (a zero vector before the first meeting)
            seqs.append((np.stack(xs[:-1]), np.array(ys)))
            split.append(val)
    import torch
    torch.manual_seed(0)
    H, C = 16, 20
    cell, out = torch.nn.GRUCell(13, H), torch.nn.Linear(H, C)
    opt = torch.optim.Adam(list(cell.parameters()) + list(out.parameters()), lr=3e-3)

    def run(batch, train: bool) -> tuple[float, float]:
        tot, correct, n = 0.0, 0, 0
        for x, y in batch:
            h = torch.zeros(1, H)
            loss = torch.zeros(())
            for t in range(len(y)):
                if t > 0:  # as at run time: the first choice is from h = 0, then one step per earlier meeting
                    h = cell(torch.tensor(x[t:t + 1], dtype=torch.float32), h)
                logits = out(h)
                loss = loss + torch.nn.functional.cross_entropy(logits, torch.tensor([int(y[t])]))
                correct += int(logits.argmax().item() == int(y[t]))
                n += 1
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += float(loss.item())
        return tot / max(1, n), correct / max(1, n)
    tr = [s for s, v in zip(seqs, split, strict=True) if not v]
    va = [s for s, v in zip(seqs, split, strict=True) if v]
    best = (-1.0, None)
    for _ep in range(60):
        order = rng.permutation(len(tr))
        run([tr[i] for i in order], True)
        with torch.no_grad():
            _, acc = run(va, False)
        if acc > best[0]:
            best = (acc, {"weight_ih": cell.weight_ih.detach().numpy().tolist(),
                          "weight_hh": cell.weight_hh.detach().numpy().tolist(),
                          "bias_ih": cell.bias_ih.detach().numpy().tolist(),
                          "bias_hh": cell.bias_hh.detach().numpy().tolist(),
                          "out_w": out.weight.detach().numpy().tolist(), "out_b": out.bias.detach().numpy().tolist()})
    assert best[1] is not None
    w = {**best[1], "val_accuracy": best[0], "sequences": len(seqs), "hidden": H}
    w["majority_baseline"] = base_rate = float(np.mean([np.mean(y == np.bincount(np.concatenate([s[1] for s in tr]))
                                                               .argmax()) for _, y in va]))
    w["target_classes"] = sorted({int(c) for _, y in seqs for c in y})
    out_path.write_text(json.dumps(w), encoding="utf-8")
    print(out_path.name, "GRU val accuracy", round(best[0], 3), "majority-class baseline", round(base_rate, 3),
          "classes", w["target_classes"], "sequences", len(seqs))


if __name__ == "__main__":
    main()
