"""Publication figures for the remediation results, generated only from committed result files.

    python experiments/M_release/scripts/make_figures.py [--out experiments/M_release/figures/remediation]

Each figure is written twice, `<name>_light.png` and `<name>_dark.png` (same data, colours for a light or dark page),
plus `figures.json` with each figure's title, sample sizes, interpretation and source files.

Sources (all in results/remediation/):
- A, B: corrected_{nova_carter,humanoid,nerva_mujoco}.json (scripts/r_analyze.py): paired differences between APM
  (DT) and each control, with 95% bootstrap CIs and Holm-corrected one-sided sign-flip tests;
- C: demo_traces.json (scripts/demo_traces.py): one learning-and-recovery lifetime per robot;
- D: expression10_humanoid.json, expression11_nerva.json, expression7_nova_carter.json (scripts/expression_measures.py);
- E: gait_battery_v1_vs_candidate.json (scripts/gait_battery_analyze.py), 520 runs.
Nothing is edited by hand: rerunning the script reproduces the files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "remediation"

# Okabe-Ito (colour-blind safe)
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00", "sky": "#56B4E9",
      "purple": "#CC79A7", "yellow": "#F0E442", "grey": "#999999"}
THEMES = {
    "light": {"bg": "#ffffff", "fg": "#2b2f32", "muted": "#6b7277", "grid": "#e6e8e9", "zero": "#494e52"},
    "dark": {"bg": "#2c2f32", "fg": "#f1f3f4", "muted": "#a3a9ae", "grid": "#3a3e41", "zero": "#d9dcdf"},
}
ROBOTS = [("nova_carter", "Nova Carter", "Isaac Sim (PhysX)"), ("humanoid", "Booster T1", "Isaac Sim (PhysX)"),
          ("nerva_mujoco", "NERVA", "MuJoCo")]
CORRECTED = {"nova_carter": "corrected_nova_carter.json", "humanoid": "corrected_humanoid.json",
             "nerva_mujoco": "corrected_nerva_mujoco.json"}


def style(theme: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["bg"], "axes.facecolor": theme["bg"], "savefig.facecolor": theme["bg"],
        "text.color": theme["fg"], "axes.labelcolor": theme["fg"], "axes.edgecolor": theme["muted"],
        "xtick.color": theme["fg"], "ytick.color": theme["fg"], "axes.titlecolor": theme["fg"],
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "grid.color": theme["grid"], "axes.grid": False})


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def save(fig, out: Path, name: str, theme_name: str) -> None:
    fig.savefig(out / f"{name}_{theme_name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def diff_bars(ax, theme, robot_key: str, tests: dict, controls: list[tuple[str, str]], colours: list[str],
              highlight: str | None = None) -> None:
    """Horizontal bars: control's cost minus APM's (s per planned encounter), 95% CI; > 0 means APM is cheaper."""
    ys = np.arange(len(controls))[::-1]
    for y, (key, _label), col in zip(ys, controls, colours, strict=True):
        t = tests[key]
        d = -t["mean_diff_DT_minus_other"]
        lo, hi = -t["ci95"][1], -t["ci95"][0]
        if abs(d) < 1e-9 and abs(hi - lo) < 1e-9:
            txt, face, hatch = "0.0 s: identical behaviour", "none", None
        elif t.get("holm_pass"):
            txt, face, hatch = f"{d:+.1f} s", col, None
        elif hi < 0:
            txt, face, hatch = f"{d:+.1f} s: APM worse", col, None
        else:
            txt, face, hatch = f"{d:+.1f} s (n.s.)", "none", "///"
        ax.barh(y, d, height=0.6, color=face, edgecolor=col, linewidth=1.6, hatch=hatch)
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="none", ecolor=theme["fg"], elinewidth=1.2, capsize=3)
        ax.text(max(hi, 0) + 0.35, y, txt, va="center", fontsize=9.5, color=theme["fg"])
    ax.axvline(0, color=theme["zero"], linewidth=1)
    ax.set_yticks(ys, [lab for _, lab in controls])
    ax.tick_params(axis="y", length=0)


def figure_a(out: Path, theme_name: str) -> dict:
    theme = THEMES[theme_name]
    style(theme)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    controls = [("identity/DT_vs_POOLED", "one shared memory\n(pooled)"),
                ("identity/DT_vs_SWAPPED", "another person's memory\n(swapped)"),
                ("identity/DT_vs_SHUFFLED", "shuffled identities")]
    ns = {}
    for ax, (key, name, engine) in zip(axes, ROBOTS, strict=True):
        r = load(CORRECTED[key])["robots"][key]
        diff_bars(ax, theme, key, r["identity_tests"], controls, [OI["blue"], OI["vermillion"], OI["purple"]])
        ax.set_title(f"{name}  ·  {engine}", loc="left")
        ns[name] = r["identity_tests"]["identity/DT_vs_POOLED"]["n_pairs"]
        if ax is not axes[0]:
            ax.set_yticklabels([])
    for ax in axes:
        ax.set_xlim(-2, 22)
    fig.supxlabel("extra task cost compared with APM's correct per-person memory (s per planned encounter; "
                  "bars = mean paired difference, whiskers = 95% bootstrap CI)", fontsize=10.5, y=-0.06)
    fig.suptitle("Remembering the right person matters: removing or mixing up person-specific memory raises cost "
                 "on every robot", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    save(fig, out, "A_person_memory", theme_name)
    return {"title": "Person-specific memory against pooled, swapped and shuffled memory",
            "n": f"{ns['Nova Carter']} paired lifetimes per comparison per robot (history risky_B, R/Y seeds 0-14)",
            "interpretation": "Bars show how much more each control costs than APM's correct per-person memory "
                              "(positive = APM better). All nine comparisons pass Holm-corrected one-sided tests. "
                              "APM also takes fewer contacts from the pushy person than pooled memory on all three "
                              "robots. Identities are provided to the memory; visual re-identification is not tested.",
            "sources": [CORRECTED[k] for k, _, _ in ROBOTS]}


def figure_b(out: Path, theme_name: str) -> dict:
    theme = THEMES[theme_name]
    style(theme)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharex=True)
    controls = [("main/DT_vs_NEUTRAL", "neutral motion\n+ same supervisor"),
                ("main/DT_vs_MONOTONE", "calibrated simple rule\n(MONOTONE*)"),
                ("main/DT_vs_NOMEM", "no memory"),
                ("main/DT_vs_FIXED", "fixed style"),
                ("main/DT_vs_LOOKUP", "lookup table")]
    cols = [OI["grey"], OI["orange"], OI["sky"], OI["green"], OI["blue"]]
    for ax, (key, name, engine) in zip(axes, ROBOTS, strict=True):
        r = load(CORRECTED[key])["robots"][key]
        diff_bars(ax, theme, key, r["tests"], controls, cols)
        ax.set_title(f"{name}  ·  {engine}", loc="left")
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_xlim(-3, 22)
    fig.supxlabel("baseline cost minus APM cost (s per planned encounter; > 0 = APM cheaper; whiskers = 95% "
                  "bootstrap CI; hatched = not significant)", fontsize=10.5, y=-0.05)
    fig.suptitle("APM beats the rule, lookup, fixed-style and no-memory baselines, but not neutral motion with the "
                 "same supervisor", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    save(fig, out, "B_fair_comparisons", theme_name)
    return {"title": "Fair controller comparisons under one safety supervisor",
            "n": "30 paired lifetimes per comparison per robot (histories change_B and late_B, R/Y seeds 0-14); "
                 "0 falls in all 720 lifetimes",
            "interpretation": "Positive bars: the baseline is costlier than APM (decision-theoretic expression, "
                              "DT). Hatched bars are not significant. Against neutral motion with the same "
                              "supervisor APM is not better: identical on Nova Carter, +0.38 s (n.s.) on T1 and "
                              "+0.78 s worse on NERVA. The gains come from per-person memory acting through the "
                              "supervisor, not from the choice of expressive style.",
            "sources": [CORRECTED[k] for k, _, _ in ROBOTS]}


def figure_c(out: Path, theme_name: str) -> dict:
    theme = THEMES[theme_name]
    style(theme)
    d = load("demo_traces.json")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    names = {"humanoid": ("Booster T1", OI["blue"], "o"), "nova_carter": ("Nova Carter", OI["orange"], "s"),
             "nerva": ("NERVA", OI["green"], "^")}
    pushy_n = d["robots"]["humanoid"]["pushy_n"]
    total = pushy_n + d["robots"]["humanoid"]["calm_n"]
    ax.axvspan(0.5, pushy_n + 0.5, color=OI["vermillion"], alpha=0.10, linewidth=0)
    ax.axvspan(pushy_n + 0.5, total + 0.5, color=OI["sky"], alpha=0.10, linewidth=0)
    ax.text(0.65, 0.515, "the person pushes", color=theme["muted"], fontsize=10)
    ax.text(pushy_n + 0.65, 0.515, "the same person is now calm", color=theme["muted"], fontsize=10)
    for key, (label, col, mk) in names.items():
        t = d["robots"][key]
        m = t["meetings"]
        xs = [0] + [r["meeting"] for r in m]
        ys = [m[0]["wariness_before"]] + [r["wariness_after"] for r in m]
        ax.plot(xs, ys, color=col, marker=mk, linewidth=2, markersize=5, label=label)
        ax.text(xs[-1] + 0.25, ys[-1], f"{label} {ys[-1]:.2f}", color=col, va="center", fontsize=9.5)
    ax.set_xlim(-0.3, total + 2.6)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(range(0, total + 1))
    ax.set_xlabel("meeting with the same person (0 = prior, before any meeting)")
    ax.set_ylabel("learned wariness of that person")
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Learning is fast, recovery is slow: T1 recovers part-way; Nova Carter and NERVA barely move "
                 "within seven calm meetings", loc="left", fontsize=12.5, pad=12)
    save(fig, out, "C_learning_recovery", theme_name)
    return {"title": "Learning and recovery across one lifetime per robot",
            "n": "one lifetime per robot (seed 73401): 6 pushy meetings, then 7 calm meetings with the same person",
            "interpretation": "Wariness is the robot's own memory readout, learned from its IMU and depth camera "
                              "without labels. It rises within the first meeting. After the switch, T1 falls from "
                              "0.97 to 0.61 and its guarded posture visibly weakens; Nova Carter (0.79) and NERVA "
                              "(0.71) stay high, because memory retention is slow and occasional calm touches still "
                              "count as disturbance. Recovery is incomplete in this horizon. Single lifetimes, so "
                              "no statistical claim; T1 needed three attempts (earlier ones fell).",
            "sources": ["demo_traces.json"]}


EXPR = {
    "humanoid": ("expression10_humanoid.json", "Booster T1 (v1 controller)",
                 [("observe_s", "watching before\nengaging (s)"), ("guard_posture_rad", "guarded arms\n(rad)"),
                  ("torso_turn_rad", "torso turned\nside-on (rad)"), ("retreat_m", "backs away\n(m)"),
                  ("turn_s", "time to turn\naway (s)")]),
    "nerva": ("expression11_nerva.json", "NERVA",
              [("observe_s", "watching before\nengaging (s)"), ("head_pitch_rad", "head pitch\n(rad, + up)"),
               ("neck_pitch_rad", "neck pitch\n(rad, + up)"), ("retreat_m", "backs away\n(m)")]),
    "nova_carter": ("expression7_nova_carter.json", "Nova Carter",
                    [("observe_s", "watching before\nengaging (s)"), ("retreat_m", "backs away\n(m)"),
                     ("turn_s", "time to turn\naway (s)"), ("heading_sweep_rad", "looks around while\n"
                                                                                "watching (rad)")]),
}


def figure_d(out: Path, theme_name: str, robot: str) -> dict:
    theme = THEMES[theme_name]
    style(theme)
    fname, label, measures = EXPR[robot]
    r = load(fname)[robot]
    conds = [("neutral", "neutral", OI["grey"]), ("memory_after_calm", "after\ncalm", OI["sky"]),
             ("memory_after_pushy", "after\npushy", OI["vermillion"])]
    fig, axes = plt.subplots(1, len(measures), figsize=(3.0 * len(measures), 3.4), gridspec_kw={"wspace": 0.45})
    for ax, (key, mlabel) in zip(axes, measures, strict=True):
        for i, (c, _clabel, col) in enumerate(conds):
            vals = [m[key] for m in r["meetings"][c] if m.get(key) is not None]
            mean = float(np.mean(vals)) if vals else 0.0
            ax.bar(i, mean, color=col, width=0.65, alpha=0.9)
            ax.scatter([i] * len(vals), vals, color=theme["fg"], s=14, zorder=3)
        ax.axhline(0, color=theme["zero"], linewidth=0.8)
        ax.set_xticks(range(3), [c[1] for c in conds], fontsize=8.5)
        ax.set_title(mlabel, fontsize=10.5)
    lw = r["learned"]
    fig.suptitle(f"{label}: measured movement in the same calm meeting after different histories "
                 f"(learned wariness {lw['pushy']['wariness']:.2f} after pushy, "
                 f"{lw['calm']['wariness']:.2f} after calm)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.06)
    save(fig, out, f"D_expression_{robot}", theme_name)
    n = {c: len(r["meetings"][c]) for c, _, _ in conds}
    return {"title": f"Physical expression, {label}",
            "n": f"meetings: neutral {n['neutral']}, after calm {n['memory_after_calm']}, after pushy "
                 f"{n['memory_after_pushy']} (dots = individual meetings; bars = means). Descriptive, not a test.",
            "interpretation": "Measured from the recorded joint and pose traces in a matched present task (identical "
                              "physics seeds; approach speed matched within 0.005 m/s). The experience state is "
                              "learned; the mapping from it to these movements is designed.",
            "sources": [fname]}


def figure_e(out: Path, theme_name: str) -> dict:
    theme = THEMES[theme_name]
    style(theme)
    g = load("gait_battery_v1_vs_candidate.json")
    runs = g["per_run"]
    def vals(ctrl: str, key: str) -> list[float]:
        return [v[key] for k, v in runs.items() if k.endswith("/" + ctrl) and v.get(key) is not None]
    panels = [("stance_slip_m_s", "foot slip in stance\n(m/s)"),
              ("joint_accel_rms", "joint acceleration RMS\n(rad/s², lower = smoother)"),
              ("tilt_p95_deg", "torso tilt, 95th pct.\n(degrees)"), ("swing_clearance_m", "swing foot clearance\n(m)"),
              ("settle_s", "time to settle\nafter stopping (s)")]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(16, 3.6))
    cols = {"v1": OI["blue"], "candidate": OI["orange"]}
    rng = np.random.default_rng(0)
    for ax, (key, lab) in zip(axes, panels, strict=False):
        for i, c in enumerate(("v1", "candidate")):
            v = np.asarray(vals(c, key), float)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, alpha=0.35, color=cols[c], linewidths=0)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=theme["fg"], linewidth=2.2)
        ax.set_xticks([0, 1], ["v1\n(kept)", "Booster\ncandidate"], fontsize=9)
        ax.set_title(lab, fontsize=10)
    # falls under 0.5 m/s walking pushes
    ax = axes[-1]
    fv = sum(v["fell"] for k, v in runs.items() if "/push_walk_0.5/" in k and k.endswith("/v1"))
    fc = sum(v["fell"] for k, v in runs.items() if "/push_walk_0.5/" in k and k.endswith("/candidate"))
    nv = sum(1 for k in runs if "/push_walk_0.5/" in k and k.endswith("/v1"))
    ax.bar([0, 1], [fv, fc], color=[cols["v1"], cols["candidate"]], width=0.6)
    for i, (f, n) in enumerate(((fv, nv), (fc, nv))):
        ax.text(i, f + 0.1, f"{f}/{n}", ha="center", fontsize=10)
    ax.set_ylim(0, 4)
    ax.set_xticks([0, 1], ["v1\n(kept)", "Booster\ncandidate"], fontsize=9)
    ax.set_title("falls under 0.5 m/s\nwalking pushes", fontsize=10)
    fig.suptitle("The official Booster controller walks more smoothly but falls under hard pushes and settles more "
                 "slowly, so the original controller was kept", x=0.01, ha="left", fontsize=13, fontweight="bold",
                 y=1.08)
    save(fig, out, "E_t1_gait", theme_name)
    return {"title": "T1 gait: original controller vs the official Booster controller",
            "n": f"frozen battery, {g['pairs']} matched pairs (4 environments x 13 scenarios x 5 seeds); dots = "
                 "runs, bars = medians",
            "interpretation": "The candidate halves foot slip, joint acceleration and tilt, but falls in 3 of 20 "
                              "runs with 0.5 m/s walking pushes (v1: 0) and takes 42% longer to settle after "
                              "stopping. It failed two of the four promotion rules frozen before the data, so the "
                              "original Isaac Lab PPO controller remains.",
            "sources": ["gait_battery_v1_vs_candidate.json"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "figures" / "remediation")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for theme_name in THEMES:
        meta["A_person_memory"] = figure_a(a.out, theme_name)
        meta["B_fair_comparisons"] = figure_b(a.out, theme_name)
        meta["C_learning_recovery"] = figure_c(a.out, theme_name)
        for robot in EXPR:
            meta[f"D_expression_{robot}"] = figure_d(a.out, theme_name, robot)
        meta["E_t1_gait"] = figure_e(a.out, theme_name)
    (a.out / "figures.json").write_text(json.dumps(meta, indent=1))
    print("wrote", sorted(p.name for p in a.out.glob("*.png")))


if __name__ == "__main__":
    main()
