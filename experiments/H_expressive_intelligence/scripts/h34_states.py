"""H3/H4: tune and train the internal-state models, then evaluate them on experience streams. Intent level.

Every model sees the same physically measured experience streams (`h34_experience.py`). After every encounter, each
model is probed with fixed evaluation situations. The probe produces:
- the readout (p, u, g);
- caution κ;
- the `ExpressionIntent` (L0 lookup, or the PAD model's own mapping).

Ground truth for evaluation only: the world's hidden jolt probability for the probe at that point in the lifetime.

    python scripts/h34_states.py tune  --root results/H34                       # V-partition tuning (equal budget)
    python scripts/h34_states.py train --root results/H34 --seeds 5             # learned S5-S7 on T, select on V
    python scripts/h34_states.py eval  --root results/H34 --partition X         # readouts, intents, controls
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from apm_express.develop import N_ENCOUNTERS, jolt_probability
from apm_express.states import (
    BASE,
    Constant,
    EntityMemory,
    FixedStyle,
    Obs,
    OracleLookup,
    PADState,
    Probe,
    ScalarEMA,
    SmallState,
    caution,
    intent_of,
)

PROBES = {
    "B1_plain": Probe("B1", "B", "plain"),
    "B1_marked": Probe("B1", "B", "marked"),
    "A1_plain": Probe("A1", "A", "plain"),
    "C1_plain": Probe("C1", "C", "plain"),
    "B3_novel": Probe("B3", "B", "plain"),
    "D1_unknown": Probe("D1", "D", "plain"),
}
TUNE_GRID = {  # equal budget: 8 configurations per tuned family, selected by V log loss of p (world learning)
    "S2_scalar": [{"alpha": a} for a in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)],
    "S3_small": [{"alpha": a} for a in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)],
    "S4_pad": [{"alpha": a} for a in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)],
    "M_id": [{"alpha": a} for a in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)],
    "M_ctx": [{"alpha": a} for a in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)],
    "M_id_cp": [{"alpha": a} for a in (0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45)],
}
LEARNED_GRID = [{"hidden": h, "lr": lr, "epochs": e} for h in (16, 32) for lr in (1e-3, 3e-3) for e in (40, 80)]


def make(name: str, cfg: dict | None = None):
    cfg = cfg or {}
    if name == "S0_none":
        return Constant()
    if name == "S1_fixed":
        return FixedStyle()
    if name == "S2_scalar":
        return ScalarEMA(alpha=cfg.get("alpha", 0.15))
    if name == "S3_small":
        return SmallState(alpha=cfg.get("alpha", 0.15))
    if name == "S4_pad":
        return PADState(alpha=cfg.get("alpha", 0.15))
    if name == "M_id":
        return EntityMemory(alpha=cfg.get("alpha", 0.15))
    if name == "M_ctx":
        return EntityMemory(context=True, alpha=cfg.get("alpha", 0.15))
    if name == "M_id_cp":
        return EntityMemory(changepoint=True, alpha=cfg.get("alpha", 0.15))
    raise ValueError(name)


def load_streams(path: Path, threshold: float) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for lt in doc["lifetimes"]:
        evs = sorted(lt["events"], key=lambda e: e["k"])
        obs = [  # observed outcome from the measured disturbance (addendum §1): peak base speed
            Obs(e["identity"], e["kind"], e["zone"], float(e["peak_speed"] >= threshold), float(e["peak_speed"] / 0.1))
            for e in evs
        ]
        out.append({"history": lt["history"], "index": lt["index"], "obs": obs, "events": evs})
    return out


def logloss(model, streams) -> float:
    """Mean log loss of the readout p for each encounter's observed outcome, before observing it."""
    total, n = 0.0, 0
    for s in streams:
        model.reset()
        for ob in s["obs"]:
            p = min(max(model.readout(Probe(ob.identity, ob.kind, ob.zone)).p, 1e-4), 1 - 1e-4)
            total -= ob.y * math.log(p) + (1 - ob.y) * math.log(1 - p)
            n += 1
            model.observe(ob)
    return total / max(n, 1)


def tune(root: Path, threshold: float) -> dict:
    v = load_streams(root / "V" / "experience.json", threshold)
    out = {}
    for fam, grid in TUNE_GRID.items():
        scores = [logloss(make(fam, cfg), v) for cfg in grid]
        best = int(np.argmin(scores))
        out[fam] = {"grid": grid, "val_logloss": scores, "selected": grid[best]}
    return out


def train_learned(root: Path, threshold: float, seeds: int, kinds=("gru", "fastslow", "gru_mem")) -> dict:
    import torch

    torch.set_num_threads(1)  # tiny recurrent steps: threading overhead dominates; one process per family instead

    from apm_express.states_learned import lifetime_tensors, train

    t = load_streams(root / "T" / "experience.json", threshold)
    v = load_streams(root / "V" / "experience.json", threshold)
    rng = np.random.default_rng(0)
    tt = [lifetime_tensors(s["obs"], rng) for s in t]
    vt = [lifetime_tensors(s["obs"], rng) for s in v]
    summary = {}
    for kind in kinds:
        trials = []
        for cfg in LEARNED_GRID:  # equal budget: 8 configurations, seed 0, selected by V loss
            _, info = train(kind, tt, vt, seed=0, **cfg)
            trials.append({"cfg": cfg, "val_loss": info["val_loss"], "params": info["params"]})
        best = min(trials, key=lambda r: r["val_loss"])["cfg"]
        nets = []
        for seed in range(seeds):
            net, info = train(kind, tt, vt, seed=seed, **best)
            nets.append(net)
            torch.save(net.state_dict(), root / "models" / f"{kind}_seed{seed}.pt")
        summary[kind] = {"trials": trials, "selected": best, "params": info["params"]}
    return summary


def learned_model(root: Path, kind: str, cfg: dict, seeds: int, name: str):
    import torch

    from apm_express.states_learned import LearnedState, Net

    nets = []
    for seed in range(seeds):
        net = Net(kind, cfg["hidden"])
        net.load_state_dict(torch.load(root / "models" / f"{kind}_seed{seed}.pt"))
        net.eval()
        nets.append(net)
    return LearnedState(nets, name)


def control_stream(obs: list[Obs], control: str, rng: np.random.Generator) -> list[Obs]:
    """Causal controls on the experience stream (MASTER §5): same encounters, altered evidence."""
    if control == "shuffled_outcomes":  # outcomes permuted across encounters: breaks entity-outcome association
        perm = rng.permutation(len(obs))
        return [Obs(o.identity, o.kind, o.zone, obs[j].y, obs[j].magnitude) for o, j in zip(obs, perm, strict=True)]
    if control == "random_identity":  # identity tags randomised (kind kept): entity-specific learning impossible
        ids = {k: [i for i in ("A1", "A2", "B1", "B2", "C1", "C2") if i[0] == k] for k in "ABC"}
        return [Obs(str(rng.choice(ids[o.kind])), o.kind, o.zone, o.y, o.magnitude) for o in obs]
    return obs


def evaluate(root: Path, partition: str, threshold: float, systems: dict, controls: tuple[str, ...]) -> dict:
    streams = load_streams(root / partition / "experience.json", threshold)
    rows = []
    rng = np.random.default_rng(12345)
    for name, factory in systems.items():
        for control in ("none",) + controls:
            for s in streams:
                model = factory(s["history"])
                model.reset()
                obs = control_stream(s["obs"], control, rng)
                course = []
                for k, ob in enumerate(obs):
                    model.observe(ob)
                    if (k + 1) % 4 == 0 or k + 1 == len(obs):
                        snap = {}
                        for pname, probe in PROBES.items():
                            it = intent_of(model, probe)
                            r = model.readout(probe)
                            snap[pname] = {
                                "kappa": caution(r) if model.fixed_intent is None else float("nan"),
                                "p": r.p,
                                "u": r.u,
                                "g": r.g,
                                "intent": it.as_dict(),
                                "truth": jolt_probability(
                                    s["history"], probe.identity, probe.zone, min(k + 1, N_ENCOUNTERS - 1)
                                )
                                if probe.kind != "D"
                                else None,
                            }
                        course.append({"k": k + 1, "probes": snap})
                reset_snap = None
                if control == "none":  # state reset before evaluation (MASTER §5 control)
                    model.reset()
                    reset_snap = {pn: intent_of(model, pr).as_dict() for pn, pr in PROBES.items()}
                rows.append(
                    {
                        "system": name,
                        "control": control,
                        "history": s["history"],
                        "index": s["index"],
                        "course": course,
                        "reset_intents": reset_snap,
                    }
                )
    return {"partition": partition, "threshold": threshold, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "train", "eval"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True, help="outcome threshold, peak speed m/s (0.04)")
    ap.add_argument("--partition", default="X")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kinds", default="gru,fastslow,gru_mem", help="train these learned families (one per process)")
    a = ap.parse_args()
    start = time.time()
    if a.mode == "tune":
        res = tune(a.root, a.threshold)
        (a.root / "tuning.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    elif a.mode == "train":
        (a.root / "models").mkdir(parents=True, exist_ok=True)
        kinds = tuple(a.kinds.split(","))
        res = train_learned(a.root, a.threshold, a.seeds, kinds)
        (a.root / f"learned_{'_'.join(kinds)}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    else:
        from apm_express.provenance import confirmatory_lock

        lock = confirmatory_lock(Path(__file__).resolve().parents[1] / "prereg" / "H34_ADDENDUM.md")
        import torch

        torch.set_num_threads(1)
        tuning = json.loads((a.root / "tuning.json").read_text(encoding="utf-8"))
        learned = {}
        for f in sorted(a.root.glob("learned_*.json")):
            learned.update(json.loads(f.read_text(encoding="utf-8")))
        systems = {n: (lambda h, n=n: make(n)) for n in ("S0_none", "S1_fixed")}
        for fam in TUNE_GRID:
            systems[fam] = lambda h, fam=fam: make(fam, tuning[fam]["selected"])
        for kind, name in (("gru", "S5_gru"), ("fastslow", "S6_fastslow"), ("gru_mem", "S7_gru_mem")):
            cfg = learned[kind]["selected"]
            systems[name] = lambda h, kind=kind, cfg=cfg, name=name: learned_model(a.root, kind, cfg, a.seeds, name)
        systems["oracle_lookup"] = lambda h: OracleLookup(history=h)
        res = evaluate(a.root, a.partition, a.threshold, systems, ("shuffled_outcomes", "random_identity"))
        res["base"] = BASE
        res["lock"] = lock
        (a.root / a.partition / "states_eval.json").write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({"mode": a.mode, "wall_s": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
