"""Milestone C: experience-grounded outcome prediction on NERVA experience (protocol: PREREGISTRATION.md).

    python experiments/C_learning/run_learning.py --experience runs/nerva_experience --out <results dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from apm import APM
from apm.adapters.base import Emitter
from apm.adapters.nerva import NervaAdapter
from apm.learning import baselines
from apm.learning.dataset import Episode, episode_from_stream, samples
from apm.learning.evaluation import bootstrap_difference, brier, summary
from apm.learning.predictor import GRUPredictor
from apm.learning.replay import ShadowLearner

STRIDE, HISTORY, HORIZON, FALL_HORIZON = 0.1, 20, 0.5, 1.0
TURNS = ([0.0, 0.0, 0.6], [0.0, 0.0, -0.6])
EARLY = ([0.0, 0.0, 0.0], [0.074, 0.0, 0.0], [-0.074, 0.0, 0.0])


def build_episode(trial: dict, a) -> Episode:
    """Replay one stored trial through the NERVA adapter; an APM runs alongside to supply the M5 affect inputs."""
    adapter = NervaAdapter(Emitter(f"nerva-{trial['trial']}"))
    apm = APM(profile=adapter.profile())
    command = tuple(trial["command"])

    def stream():
        yield 0.0, [adapter.profile()]
        for i in range(len(a["t"])):
            t = float(a["t"][i])
            msgs = adapter.step(t, a["base_quat"][i], a["base_linvel"][i], a["base_angvel"][i],
                                float(a["tau_sq"][i]), command)
            for m in msgs:
                apm.ingest(m)
            yield t, msgs

    def affect(t):
        out = apm.step(t)
        return np.concatenate([out.affect_vector(), apm.affect.u])

    return episode_from_stream(str(trial["trial"]), adapter.profile(), stream(), STRIDE, meta=trial,
                               on_sample=affect)


def build_episodes(experience: Path, limit: int | None = None) -> list[Episode]:
    manifest = json.loads((experience / "manifest.json").read_text(encoding="utf-8"))
    return [build_episode(trial, np.load(experience / f"trial_{trial['trial']:04d}.npz"))
            for trial in manifest["trials"][:limit]]


def split(episodes, seed=0):
    ids = np.random.default_rng(seed).permutation(len(episodes))
    n = len(ids)
    return ([episodes[i] for i in ids[:int(.6 * n)]], [episodes[i] for i in ids[int(.6 * n):int(.8 * n)]],
            [episodes[i] for i in ids[int(.8 * n):]])


def fit_all(train, val, kind, horizon):
    X, y, _, _ = samples(train, kind, horizon, HISTORY)
    Xv, yv, _, _ = samples(val, kind, horizon, HISTORY)
    models = {"M0_goal_rate": baselines.GoalRate().fit(X, y), "M1_margin_bins": baselines.MarginBins().fit(X, y),
              "M2_logistic": baselines.Logistic().fit(X, y), "M3_tilt_extrapolation": baselines.TiltExtrapolation()}
    t0 = time.monotonic()
    gru = GRUPredictor(X.shape[2], hidden=64, seed=0).fit(X, y, Xv, yv)
    models["M4_gru"] = gru
    train_seconds = {"M4_gru": time.monotonic() - t0}
    return models, train_seconds


def evaluate(models, test, kind, horizon, extra_model=None):
    X, y, g, _ = samples(test, kind, horizon, HISTORY)
    preds = {name: m.predict(X) for name, m in models.items()}
    if extra_model is not None:
        Xe, ye, ge, _ = samples(test, kind, horizon, HISTORY, use_extra=True)
        assert np.array_equal(ye, y) and np.array_equal(ge, g)
        preds["M5_gru_affect"] = extra_model.predict(Xe)
    no_push = np.isin(g, [i for i, e in enumerate(test) if e.meta["push"] is None and not e.events])
    metrics = {}
    for name, p in preds.items():
        metrics[name] = summary(p, y)
        metrics[name]["false_alarm_rate_no_push"] = float(np.mean(p[no_push] >= 0.5)) if no_push.any() else None
    return preds, y, g, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experience", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/C_learning/results"))
    ap.add_argument("--limit", type=int, help="smoke test on the first N trials only (not a result)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    episodes = build_episodes(args.experience, args.limit)
    counts = {k: sum(len(e.events.get(k, [])) for e in episodes) for k in ("stability_loss", "fall", "recovery",
                                                                           "task_interrupted")}
    print("episodes", len(episodes), "events", counts, f"{time.monotonic() - t_start:.0f} s", flush=True)
    train, val, test = split(episodes)
    results: dict = {"episodes": len(episodes), "event_counts": counts,
                     "splits": {"train": len(train), "val": len(val), "test": len(test)}}

    # ---------------- held-out episodes, primary target
    models, train_seconds = fit_all(train, val, "stability_loss", HORIZON)
    Xe, ye, _, _ = samples(train, "stability_loss", HORIZON, HISTORY, use_extra=True)
    Xev, yev, _, _ = samples(val, "stability_loss", HORIZON, HISTORY, use_extra=True)
    t0 = time.monotonic()
    m5 = GRUPredictor(Xe.shape[2], hidden=64, seed=0).fit(Xe, ye, Xev, yev)
    train_seconds["M5_gru_affect"] = time.monotonic() - t0
    preds, y, g, metrics = evaluate(models, test, "stability_loss", HORIZON, extra_model=m5)
    best_baseline = min(("M1_margin_bins", "M2_logistic"), key=lambda k: metrics[k]["brier"])
    c1 = bootstrap_difference(preds["M4_gru"], preds[best_baseline], y, g)
    c3 = bootstrap_difference(preds["M5_gru_affect"], preds["M4_gru"], y, g)
    results["held_out_episodes"] = {"metrics": metrics, "train_seconds": train_seconds,
                                    "best_statistical_baseline": best_baseline,
                                    "M4_minus_best_baseline_brier": c1, "M5_minus_M4_brier": c3,
                                    "train_samples": int(len(ye)), "train_positives": int(ye.sum())}
    criteria = {
        "C1_gru_beats_statistical": bool(metrics["M4_gru"]["brier"] <= 0.9 * metrics[best_baseline]["brier"]
                                         and c1["ci95"][1] < 0),
        "C2_calibration": bool(metrics["M4_gru"]["ece"] <= 0.05),
        "C3_affect_adds_value": bool(c3["ci95"][1] < 0)}
    print(json.dumps({m: {"brier": round(v["brier"], 5), "ece": round(v["ece"], 4), "auroc": round(v["auroc"], 3)}
                      for m, v in metrics.items()}, indent=1), flush=True)

    # ---------------- secondary target: fall within 1 s (reported only)
    fall_models, _ = fit_all(train, val, "fall", FALL_HORIZON)
    _, _, _, fall_metrics = evaluate({k: fall_models[k] for k in ("M1_margin_bins", "M2_logistic", "M4_gru")},
                                     test, "fall", FALL_HORIZON)
    results["secondary_fall_1s"] = fall_metrics

    # ---------------- held-out context: train non-turn, test turns
    def is_turn(e):
        return list(e.meta["command"]) in TURNS
    ctx_models, _ = fit_all([e for e in train if not is_turn(e)], [e for e in val if not is_turn(e)],
                            "stability_loss", HORIZON)
    _, _, _, ctx_metrics = evaluate(ctx_models, [e for e in test + val if is_turn(e)], "stability_loss", HORIZON)
    results["held_out_context_turns"] = ctx_metrics
    criteria["C4_context_generalization"] = bool(ctx_metrics["M4_gru"]["brier"] <= ctx_metrics["M2_logistic"]["brier"])

    # ---------------- incremental learning with shadow promotion
    def early(e):
        return (not e.meta["latency"]) and list(e.meta["command"]) in EARLY
    early_train = [e for e in train if early(e)]
    later = [e for e in train if not early(e)]
    rng = np.random.default_rng(1)
    later = [later[i] for i in rng.permutation(len(later))]
    batches = np.array_split(np.arange(len(later)), 4)
    Xv, yv, gv, _ = samples(val, "stability_loss", HORIZON, HISTORY)
    protected = np.isin(gv, [i for i, e in enumerate(val) if early(e)])
    X0, y0, _, _ = samples(early_train, "stability_loss", HORIZON, HISTORY)
    stage0 = GRUPredictor(X0.shape[2], hidden=64, seed=0).fit(X0, y0, Xv[protected], yv[protected])
    Xt, yt, gt, _ = samples(test, "stability_loss", HORIZON, HISTORY)
    early_test = np.isin(gt, [i for i, e in enumerate(test) if early(e)])

    def test_scores(model):
        p = model.predict(Xt)
        return {"all": brier(p, yt), "early_life": brier(p[early_test], yt[early_test])}

    incremental = {"stage0": test_scores(stage0), "early_train_episodes": len(early_train)}
    for variant, old in (("replay_50pct", 0.5), ("no_replay", 0.0)):
        learner = ShadowLearner(validated=stage0.clone(), X_val=Xv, y_val=yv, protected=protected, old_fraction=old)
        learner.buffer.add(X0, y0, 0)
        records = []
        for k, idx in enumerate(batches, start=1):
            Xb, yb, _, _ = samples([later[i] for i in idx], "stability_loss", HORIZON, HISTORY)
            if variant == "replay_50pct" and k == 2:
                records.append(learner.offer(Xb, yb, 100 + k, corrupt=True))  # must be rejected
            records.append(learner.offer(Xb, yb, k))
        incremental[variant] = {"arrivals": records, "final_test": test_scores(learner.validated),
                                "promotions": int(sum(r.get("promoted", False) for r in records))}
        if variant == "replay_50pct":
            before = learner.validated
            rolled = learner.rollback()
            incremental["rollback_works"] = bool(rolled and learner.validated is not before)
    final = incremental["replay_50pct"]["final_test"]
    criteria["C5_forgetting_with_replay"] = bool(final["early_life"] <= 1.1 * incremental["stage0"]["early_life"]
                                                 and final["all"] < incremental["stage0"]["all"])
    corrupt = [r for r in incremental["replay_50pct"]["arrivals"] if r.get("corrupted")]
    criteria["corrupted_shadow_rejected"] = bool(corrupt and not any(r["promoted"] for r in corrupt))
    results["incremental"] = incremental

    # ---------------- costs
    gru = models["M4_gru"]
    sample = Xt[:1000]
    t0 = time.perf_counter()
    for i in range(len(sample)):
        gru.predict(sample[i:i + 1])
    results["costs"] = {"M4_parameters": gru.parameters, "M4_state_bytes": gru.state_bytes(),
                        "M4_cpu_inference_ms_per_sample_batch1": 1000 * (time.perf_counter() - t0) / len(sample),
                        "logistic_parameters": baselines.parameter_count(models["M2_logistic"]),
                        "margin_bins_parameters": baselines.parameter_count(models["M1_margin_bins"]),
                        "M4_train_seconds": train_seconds["M4_gru"],
                        "M4_epochs_run": len(gru.history)}
    results["criteria"] = criteria
    results["wall_seconds"] = round(time.monotonic() - t_start, 1)
    (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({"criteria": criteria, "costs": results["costs"]}, indent=1))


if __name__ == "__main__":
    main()
