"""Programme K figures (`figures/`), from committed results and run logs.

    python k_plots.py --k3 <K3_X.jsonl> --k5 <K5_X.jsonl> [--demo <trace.json>] [--controls <controls.jsonl>]

1. `K1_information.png`: evidence per sensing window, calm vs pushy, by distance and sensor.
2. `K3_recovery_nerva.png`, `K5_recovery_wheeled.png`: change_B (pushy → calm at 16) and late_B (calm → pushy).
   B1's belief P(disruptive) and stopping distance per B1 meeting, mean ± 95% CI, per system.
3. `K4_demo.png`: the two continuous robots: belief, stopping distance, informative observations and contacts per
   encounter, with stage boundaries. Control conditions as means when `--controls` is given.
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
FIG = HERE / "figures"
COL = {"L0": "#7f8c8d", "DT_STATIC": "#bdc3c7", "DT": "#2980b9", "DT_VOI": "#16a085", "DT_UNC": "#8e44ad",
       "ORACLE": "#000000", "DT_CAM": "#e67e22", "DT_CAM_VOI": "#c0392b"}
SWITCH = 16


def ci(x) -> tuple[float, float, float]:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)


def fig_info() -> None:
    r = json.loads((HERE / "results" / "K1_information.json").read_text())
    ds = sorted(r["imu"], key=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(ds))
    ax.bar(x - 0.25, [r["imu"][d]["detector_kl_calm_pushy"] for d in ds], 0.25, label="IMU detector")
    ax.bar(x, [r["imu"][d]["acc8_kl_calm_pushy"] for d in ds], 0.25, label="IMU graded (acc8)")
    ax.bar(x + 0.25, [r["imu"][d]["touch_kl_calm_pushy"] for d in ds], 0.25, label="ideal touch (reference)")
    ax.axhline(r["camera"]["kl_calm_pushy"], color="#c0392b", ls="--", label="camera extension (any distance)")
    ax.set_xticks(x, [f"{float(d):.1f} m" for d in ds])
    ax.set_ylabel("nats per window, calm meeting\n(evidence that it is calm)")
    ax.set_title("K1: what a calm meeting tells NERVA, by distance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "K1_information.png", dpi=150)
    plt.close(fig)


def load(path: Path) -> dict:
    d: dict = defaultdict(lambda: defaultdict(list))
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d[r["system"]][r["history"]].append(r["rows"])
    return d


def fig_recovery(path: Path, name: str, title: str, systems: list[str]) -> None:
    d = load(path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True)
    for col, h in enumerate(("change_B", "late_B")):
        for s in systems:
            if s not in d or h not in d[s]:
                continue
            per_b, per_d = defaultdict(list), defaultdict(list)
            for rows in d[s][h]:
                b1 = [x for x in rows if x["entity"] == "B1"]
                post = [x for x in b1 if x["k"] >= SWITCH]
                pre = [x for x in b1 if x["k"] < SWITCH][-4:]
                for j, x in enumerate(pre):
                    per_b[j - len(pre)].append(x["b_after"][-1])
                    per_d[j - len(pre)].append(x["d_stop"])
                for j, x in enumerate(post[:14]):
                    per_b[j + 1].append(x["b_after"][-1])
                    per_d[j + 1].append(x["d_stop"])
            ks = sorted(per_b)
            for row, per in ((0, per_b), (1, per_d)):
                st = np.array([ci(per[k]) for k in ks])
                ax = axes[row, col]
                ax.plot(ks, st[:, 0], color=COL.get(s), label=s, lw=2 if "VOI" in s or s == "DT_CAM" else 1.2)
                ax.fill_between(ks, st[:, 1], st[:, 2], color=COL.get(s), alpha=0.12)
        for row in (0, 1):
            axes[row, col].axvline(0.5, color="grey", ls=":", lw=1)
        axes[0, col].set_title(f"{h}: {'pushy → calm' if h == 'change_B' else 'calm → pushy'} (switch at dotted line)")
        axes[1, col].set_xlabel("B1 meetings relative to the switch")
    axes[0, 0].set_ylabel("belief P(disruptive) after meeting")
    axes[1, 0].set_ylabel("stopping distance (m)")
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_demo(trace: Path, controls: Path | None) -> None:
    res = json.loads(trace.read_text())
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    cols = {"A": "#c0392b", "B": "#2980b9"}
    for r in res:
        rows = r["rows"]
        k = [x["k"] + 1 for x in rows]
        axes[0].plot(k, [x["p_disruptive"] for x in rows], "-o", ms=3, color=cols[r["robot"]],
                     label=f"Robot {r['robot']}")
        axes[1].plot(k, [x["d_stop"] for x in rows], "-o", ms=3, color=cols[r["robot"]])
        axes[2].bar(np.array(k) + (-0.2 if r["robot"] == "A" else 0.2), [x["informative"] for x in rows], 0.4,
                    color=cols[r["robot"]])
        axes[3].bar(np.array(k) + (-0.2 if r["robot"] == "A" else 0.2), [x["contacts"] for x in rows], 0.4,
                    color=cols[r["robot"]])
    if controls is not None:
        cd: dict = defaultdict(lambda: defaultdict(list))
        for line in controls.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            for x in r["rows"]:
                cd[(r["condition"], r["robot"])][x["k"] + 1].append(x["d_stop"])
        ls = {"NO_MEMORY_UPDATE": ":", "NO_EXPRESSION": "-.", "SHUFFLED": "--", "NO_INFO": (0, (1, 3))}
        for (cond, robot), per in sorted(cd.items()):
            if cond == "FULL":
                continue
            ks = sorted(per)
            axes[1].plot(ks, [np.mean(per[k]) for k in ks], ls=ls.get(cond, "-"), color=cols[robot], lw=0.9,
                         alpha=0.7, label=f"{cond} ({robot}, mean)")
    for ax in axes:
        for b in (10.5, 16.5):
            ax.axvline(b, color="grey", ls=":", lw=1)
    axes[0].set_ylabel("belief\nP(disruptive)")
    axes[1].set_ylabel("stopping distance (m)")
    axes[2].set_ylabel("informative\nobservations")
    axes[3].set_ylabel("contacts\n(simulator log)")
    axes[3].set_xlabel("encounter (one continuous lifetime per robot)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=6, ncol=2)
    axes[0].set_title("K4: stage 1 A meets a disruptive entity, B a reliable one | stage 2-3 both meet the disruptive "
                      "entity | stage 4-6 it becomes reliable", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "K4_demo.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k3", type=Path)
    ap.add_argument("--k5", type=Path)
    ap.add_argument("--demo", type=Path)
    ap.add_argument("--controls", type=Path)
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    fig_info()
    if a.k3:
        fig_recovery(a.k3, "K3_recovery_nerva.png", "K3-X NERVA: recovery after the entity changes",
                     ["L0", "DT_STATIC", "DT", "DT_VOI", "DT_UNC", "ORACLE", "DT_CAM", "DT_CAM_VOI"])
    if a.k5:
        fig_recovery(a.k5, "K5_recovery_wheeled.png", "K5-X wheeled robot: recovery after the entity changes",
                     ["L0", "DT", "DT_VOI", "ORACLE", "DT_CAM", "DT_CAM_VOI"])
    if a.demo:
        fig_demo(a.demo, a.controls)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
