"""Programme J behavioural comparison figures (`figures/`), from committed results and the run logs.

    python j_plots.py --j4 <J4-X lifetimes.jsonl> --j5 <J5-X lifetimes.jsonl> --j6 <J6-X lifetimes.jsonl>

Figures:
1. `J1_safety.png`: avoidable-collision change vs A0 per programme-I system (J1-X), with 95% CIs. R_decay is the
   decay-only control.
2. `J2_stress.png`: EU_I and EC per stress condition: avoidable-collision change vs A0, careful misclassification and
   history AUROC; failed criteria marked.
3. `J4_beliefs.png`: NERVA memory, B1 readout p by checkpoint for each memory and history (mean ± 95% CI).
4. `J5_J6_beliefs.png`: closed loop, B1's belief before each meeting (mean by encounter, risky_B / change_B / late_B /
   safe) for CL and LEARNED_V.
5. `J5_J6_expression.png`: stopping distance against belief over B1 encounters, per system.
6. `J5_J6_outcomes.png`: per system (J6-X): mean arrival time, contacts and max push per B1 encounter (risky_B,
   k ≥ 10), and strong-evidence rate.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES, FIG = HERE / "results", HERE / "figures"
HIST_COL = {"risky_B": "#c0392b", "change_B": "#e67e22", "late_B": "#2980b9", "safe": "#27ae60",
            "zone_B": "#8e44ad", "risky_A": "#7f8c8d", "unpredictable": "#16a085"}


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def fig_j1() -> None:
    r = json.loads((RES / "J1" / "J1_X.json").read_text())
    names = [k.removesuffix("_vs_A0") for k in r["comparisons"]]
    est = [r["comparisons"][f"{n}_vs_A0"]["avoidable_collision_rate"]["rel"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    y = np.arange(len(names))
    ax.errorbar([100 * e["estimate"] for e in est], y, fmt="o", color="k",
                xerr=[[100 * (e["estimate"] - e["ci95"][0]) for e in est],
                      [100 * (e["ci95"][1] - e["estimate"]) for e in est]])
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvline(10, color="red", lw=0.8, ls="--", label="+10% non-inferiority margin")
    ax.set_yticks(y, names)
    ax.set_xlabel("avoidable collisions vs A0 (%)")
    ax.set_title("J1: wheeled safety (frozen calibration, new test seeds)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "J1_safety.png", dpi=150)
    plt.close(fig)


def fig_j2() -> None:
    r = json.loads((RES / "J2" / "J2_X.json").read_text())
    conds = list(r["closed"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for j, s in enumerate(("EU_I", "EC")):
        off = -0.15 + 0.3 * j
        y = np.arange(len(conds)) + off
        rows = [r["closed"][c].get(s) for c in conds]
        for ax, key, scale in ((axes[0], "avoidable_rel", 100), (axes[1], "careful_misclassification", 1),
                               (axes[2], "auroc_S3", 1)):
            vals = [(row["vs_A0"][key] if key == "avoidable_rel" else row[key]) if row else None for row in rows]
            e = np.array([v["estimate"] * scale if v else np.nan for v in vals])
            lo = np.array([v["ci95"][0] * scale if v else np.nan for v in vals])
            hi = np.array([v["ci95"][1] * scale if v else np.nan for v in vals])
            ax.errorbar(e, y, xerr=[e - lo, hi - e], fmt="o", ms=4, label=s)
    surv = r["survival"]
    labels = []
    for c in conds:
        bad = [s for s in ("EU_I", "EC") if c in surv[s]["failing"]]  # the analysis's own verdict per condition
        labels.append(c + (f"  ✗ {','.join(bad)}" if bad else ""))
    axes[0].set_yticks(np.arange(len(conds)), labels, fontsize=8)
    axes[0].axvline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("avoidable collisions vs A0 (%)")
    axes[1].axvline(0.20, color="red", ls="--", lw=0.8)
    axes[1].set_xlabel("careful misclassification (criterion ≤ 0.20)")
    axes[2].axvline(0.80, color="red", ls="--", lw=0.8)
    axes[2].set_xlabel("history AUROC S3 (criterion ≥ 0.80)")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("J2: near-miss evidence memory under stress (✗ = a survival criterion failed)")
    fig.tight_layout()
    fig.savefig(FIG / "J2_stress.png", dpi=150)
    plt.close(fig)


def fig_j4(path: Path) -> None:
    lts = load(path)
    mems = ("H_A0", "count", "type_kind", "gru")
    hists = ("risky_B", "safe", "change_B", "late_B", "zone_B")
    fig, axes = plt.subplots(1, len(mems), figsize=(14, 3.4), sharey=True)
    for ax, m in zip(axes, mems, strict=True):
        for h in hists:
            tr = np.array([[cp[0][0] for cp in r["probes"][m]] for r in lts if r["history"] == h and m in r["probes"]])
            if not len(tr):
                continue
            stats = np.array([ci(tr[:, j]) for j in range(tr.shape[1])])
            x = 6 * (np.arange(tr.shape[1]) + 1)
            ax.plot(x, stats[:, 0], color=HIST_COL[h], label=h)
            ax.fill_between(x, stats[:, 1], stats[:, 2], color=HIST_COL[h], alpha=0.2)
        ax.axvline(24, color="grey", ls=":", lw=0.8)
        ax.set_title(m)
        ax.set_xlabel("encounter")
    axes[0].set_ylabel("B1 readout p")
    axes[0].legend(fontsize=7)
    fig.suptitle("J4-X: NERVA memories from IMU evidence (switch at encounter 24 in change_B / late_B)")
    fig.tight_layout()
    fig.savefig(FIG / "J4_beliefs.png", dpi=150)
    plt.close(fig)


def by_system(lts: list[dict]) -> dict:
    d: dict = defaultdict(lambda: defaultdict(list))
    for r in lts:
        d[r["system"]][r["history"]].append(r)
    return d


def fig_loop_beliefs(j5: list[dict], j6: list[dict]) -> None:
    sets = [("J5-X CL (L0)", by_system(j5)["CL"]), ("J6-X LEARNED_V", by_system(j6)["LEARNED_V"]),
            ("J5-X H_MEM", by_system(j5)["H_MEM"])]
    fig, axes = plt.subplots(1, len(sets), figsize=(13, 3.4), sharey=False)
    for ax, (title, hist) in zip(axes, sets, strict=True):
        for h, rs in hist.items():
            per = defaultdict(list)
            for r in rs:
                for x in r["rows"]:
                    if x["entity"] == "B1":
                        per[x["k"]].append(x["p_before"])
            ks = sorted(per)
            st = np.array([ci(per[k]) for k in ks])
            ax.plot(ks, st[:, 0], color=HIST_COL[h], label=h)
            ax.fill_between(ks, st[:, 1], st[:, 2], color=HIST_COL[h], alpha=0.2)
        ax.axvline(15, color="grey", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("encounter")
    axes[0].set_ylabel("B1 belief p before meeting")
    axes[0].legend(fontsize=7)
    fig.suptitle("Closed loop on NERVA: B1 belief by history (switch at encounter 15)")
    fig.tight_layout()
    fig.savefig(FIG / "J5_J6_beliefs.png", dpi=150)
    plt.close(fig)


def fig_loop_expression(j6: list[dict]) -> None:
    d = by_system(j6)
    systems = [s for s in ("CL", "LEARNED", "LEARNED_V", "RESIDUAL_V", "GRU_V", "FROZEN") if s in d]
    fig, axes = plt.subplots(1, len(systems), figsize=(2.6 * len(systems), 3.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, s in zip(np.atleast_1d(axes), systems, strict=True):
        for h, rs in d[s].items():
            xs = [x for r in rs for x in r["rows"] if x["entity"] == "B1"]
            p = np.array([x["p_before"] for x in xs])
            dd = np.array([x["d_stop"] for x in xs]) + rng.normal(0, 0.003, len(xs))
            ax.scatter(p, dd, s=2, alpha=0.25, color=HIST_COL[h], label=h)
        ax.set_title(s)
        ax.set_xlabel("belief p")
    np.atleast_1d(axes)[0].set_ylabel("stopping distance (m)")
    np.atleast_1d(axes)[0].legend(fontsize=6, markerscale=4)
    fig.suptitle("J6-X: belief → stopping distance (B1 encounters)")
    fig.tight_layout()
    fig.savefig(FIG / "J5_J6_expression.png", dpi=150)
    plt.close(fig)


def fig_loop_outcomes(j6: list[dict]) -> None:
    d = by_system(j6)
    systems = [s for s in ("FROZEN", "CL", "LEARNED", "LEARNED_V", "RESIDUAL_V", "GRU_V") if s in d]
    metrics = [("arrival time (s)", lambda r: [x["t_arrive"] for x in r["rows"] if x["t_arrive"] is not None]),
               ("contacts / B1 meeting\n(risky_B, k ≥ 10)",
                lambda r: [x["contacts"] for x in r["rows"] if x["entity"] == "B1" and x["k"] >= 10]),
               ("max push / B1 meeting (m/s)\n(risky_B, k ≥ 10)",
                lambda r: [x["max_push"] for x in r["rows"] if x["entity"] == "B1" and x["k"] >= 10]),
               ("strong-evidence rate", lambda r: [x["category"] >= 6 for x in r["rows"] if x.get("observable", True)])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.2))
    for ax, (name, f) in zip(axes, metrics, strict=True):
        hs = ("risky_B",) if "risky_B" in name else tuple(d[systems[0]])
        st = [ci([np.mean(f(r)) for h in hs for r in d[s][h] if len(f(r))]) for s in systems]
        err = [[m - lo for m, lo, _ in st], [hi - m for m, _, hi in st]]
        ax.bar(range(len(systems)), [m for m, _, _ in st], yerr=err, color="#95a5a6", capsize=3)
        ax.set_xticks(range(len(systems)), systems, rotation=45, ha="right", fontsize=8)
        ax.set_title(name, fontsize=9)
    fig.suptitle("J6-X: task and physical outcomes per expression mapping (mean per lifetime, 95% CI)")
    fig.tight_layout()
    fig.savefig(FIG / "J5_J6_outcomes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--j4", type=Path, required=True)
    ap.add_argument("--j5", type=Path, required=True)
    ap.add_argument("--j6", type=Path, required=True)
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    fig_j1()
    fig_j2()
    fig_j4(a.j4)
    j5, j6 = load(a.j5), load(a.j6)
    fig_loop_beliefs(j5, j6)
    fig_loop_expression(j6)
    fig_loop_outcomes(j6)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
