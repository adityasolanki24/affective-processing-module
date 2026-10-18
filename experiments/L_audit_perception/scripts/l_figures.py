"""Programme L figures (read only from committed results and the raw traces they hash).

    python l_figures.py depth-frames --out figures/C_depth_frames.npz      (NERVA venv: renders one encounter)
    python l_figures.py plots --out figures/                               (matplotlib: every figure whose inputs exist)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
L = REPO / "experiments" / "L_audit_perception"
sys.path.insert(0, str(REPO / "src"))


def depth_frames(out: Path) -> None:
    """One NERVA encounter with a pushy entity, rendered with the robot's own depth eye; keeps the frames of the
    first window with a perceived event."""
    from apm_transfer.expression_map import GRID
    from apm_transfer.nerva_world import closed_loop_encounter
    from apm_transfer.vision import VisionCamera
    channels = json.loads((REPO / "experiments" / "J_validation_transfer" / "configs" /
                           "H_admitted_interface.json").read_text())["interface"]
    for seed in range(1, 40):
        _, o, _ = closed_loop_encounter(seed, "pushy", GRID[2], channels, windows=1, camera=VisionCamera(keep_frames=True),
                                        abort_on_detect=False)
        ev = o["vision"]["events"][0]
        if ev and o["vision"]["truth"][0]:
            frames = np.array(o["vision"]["frames"], np.float32)
            np.savez_compressed(out, frames=frames, events=json.dumps(ev), truth=json.dumps(o["vision"]["truth"][0]),
                                d_stop=o["final_distance"], seed=seed)
            print("seed", seed, "events", ev, "->", out)
            return
    raise SystemExit("no encounter with a perceived lunge")


def plots(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    res = L / "results"

    # B: corrected recovery per body and system
    if (res / "B_corrective.json").exists():
        b = json.loads((res / "B_corrective.json").read_text())["bodies"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, (exp, o) in zip(axes, b.items(), strict=False):
            systems = [s for s in ("L0", "DT", "DT_VOI", "DT_CAM", "DT_CAM_VOI", "DT_K", "DT_CAM_VOI_K")
                       if s in o["summary"]]
            vals = [o["summary"][s]["recovery_task"] for s in systems]
            vals = [14.5 if (v is None or not np.isfinite(v)) else v for v in vals]
            cols = ["#2a9d8f" if o["criteria"].get(s, {}).get("milestone") else "#bbbbbb" for s in systems]
            ax.barh(systems, vals, color=cols)
            ax.axvline(8, color="k", ls="--", lw=0.8)
            ax.set_title(f"{exp} {o['robot']}")
            ax.set_xlabel("KM median B1 meetings to recover\n(task criterion; 14.5 = not reached)")
        fig.suptitle("Phase B: recovery with every audit correction (green = milestone met)")
        fig.tight_layout()
        fig.savefig(out / "B_recovery.png", dpi=130)
        plt.close(fig)

    # C.1: information per window, vision vs abstract camera
    if (res / "C1_perception.json").exists():
        c = json.loads((res / "C1_perception.json").read_text())
        fig, axes = plt.subplots(1, len(c), figsize=(4.2 * len(c), 3.4), sharey=True)
        for ax, (robot, o) in zip(np.atleast_1d(axes), c.items(), strict=False):
            stops = list(o["information_nats_per_window"])
            x = np.arange(len(stops))
            ax.bar(x - 0.2, [o["information_nats_per_window"][s]["vision"]["kl_calm_pushy"] for s in stops], 0.4,
                   label="rendered vision")
            ax.bar(x + 0.2, [o["information_nats_per_window"][s]["abstract"]["kl_calm_pushy"] for s in stops], 0.4,
                   label="abstract camera (K)")
            ax.axhline(0.14, color="k", ls=":", lw=0.8)
            ax.set_xticks(x, [f"{s} m" for s in stops])
            ax.set_title(f"{robot}: recall {o['recall']:.2f}, false/window {o['false_events_per_window']:.2f}")
            ax.set_ylabel("KL(calm || pushy), nats per window")
        np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.suptitle("Phase C: evidence per sensing window (dotted: NERVA IMU at 0.6 m, K1)")
        fig.tight_layout()
        fig.savefig(out / "C_information.png", dpi=130)
        plt.close(fig)

    # C: depth frames strip
    f = out / "C_depth_frames.npz"
    if f.exists():
        z = np.load(f)
        fr = z["frames"]
        ev = json.loads(str(z["events"]))
        sel = np.linspace(0, len(fr) - 1, 10).astype(int)
        fig, axes = plt.subplots(1, len(sel), figsize=(16, 1.9))
        for ax, i in zip(axes, sel, strict=True):
            ax.imshow(np.clip(fr[i], 0, 1.5), cmap="magma_r", vmin=0, vmax=1.5)
            ax.set_axis_off()
            ax.set_title(f"frame {i}", fontsize=7)
        fig.suptitle(f"NERVA's own depth eye (64x48) during one window; perceived: {ev}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "C_depth_frames.png", dpi=130)
        plt.close(fig)

    # D: scenario x memory task cost
    if (res / "D_stress.json").exists():
        d = json.loads((res / "D_stress.json").read_text())["scenarios"]
        scn = list(d)
        mems = ("M0_prior", "M1_bayes", "M2_leaky", "M3_context", "ORACLE")
        fig, ax = plt.subplots(figsize=(13, 3.8))
        x = np.arange(len(scn))
        for j, m in enumerate(mems):
            ax.bar(x + (j - 2) * 0.16, [d[s]["systems"].get(m, {}).get("cost", np.nan) for s in scn], 0.16, label=m)
        ax.set_xticks(x, scn, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("task cost per B meeting (s)")
        ax.legend(fontsize=8, ncol=5)
        ax.set_title("Phase D: stress scenarios (wheeled, IMU + rendered vision)")
        fig.tight_layout()
        fig.savefig(out / "D_stress.png", dpi=130)
        plt.close(fig)

    # G: the demo lifetime
    tr = REPO / "runs" / "L" / "G_demo" / "trace.json"
    if tr.exists():
        t = json.loads(tr.read_text())
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        for lt in t:
            ks = [x["k"] + 1 for x in lt["rows"]]
            axes[0].plot(ks, [x["d_stop"] for x in lt["rows"]], "o-", ms=3, label=f"robot {lt['robot']}")
            axes[1].plot(ks, [x["p_risk"] for x in lt["rows"]], "o-", ms=3, label=f"robot {lt['robot']}")
        for ax in axes:
            ax.axvspan(0.5, 10.5, color="#f4a261", alpha=0.15)
            ax.axvspan(10.5, 16.5, color="#e76f51", alpha=0.15)
            ax.axvspan(16.5, 36.5, color="#2a9d8f", alpha=0.12)
        axes[0].set_ylabel("stopping distance (m)")
        axes[1].set_ylabel("expected disruption (belief)")
        axes[1].set_xlabel("encounter (stage 1: A pushy / B calm; 2: both pushy; 3: both calm)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Phase G: two identical NERVA robots, continuous lifetimes, IMU + rendered vision")
        fig.tight_layout()
        fig.savefig(out / "G_demo.png", dpi=130)
        plt.close(fig)
    print("figures ->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("depth-frames", "plots"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    depth_frames(a.out) if a.mode == "depth-frames" else plots(a.out)


if __name__ == "__main__":
    main()
