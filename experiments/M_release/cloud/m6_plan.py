"""Print the jcloud submit commands for the PROTOCOL.md runs (V calibration, X main study, X environments, F
robustness, D demonstration lifetimes), so every run follows the frozen design. Nothing is submitted by this script.

    python experiments/M_release/cloud/m6_plan.py --stage V|X|D|F --vms apm-m-gpu9,apm-m-gpu10,...

Each job runs `m5_par.sh` (3 Isaac processes per VM) on one robot × scene × system set.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CFG = "experiments/M_release/configs"
T1 = "--policy-dir experiments/M_release/policies/t1_isaaclab_v1"
ROBOTS = ("nova_carter", "humanoid", "nerva")
MAIN = "DT,NOMEM,FROZEN,SHUFFLED,FIXED,NEUTRAL,LOOKUP,{mono}"


def model_args(robot: str) -> str:
    return (f"--types-model {CFG}/M5_{robot}_types.json --edges {CFG}/M5_{robot}_edges.json "
            f"--time-model {CFG}/M5_{robot}_time.json --task-config {CFG}/M5_{robot}_tasks.json "
            + (T1 if robot == "humanoid" else ""))


def mono(robot: str) -> str:
    t = json.loads((REPO / CFG / "M6_tuned.json").read_text())
    return t[robot]


GRID = ["MONOTONE:lo=0.1/width=0.2", "MONOTONE:lo=0.1/width=0.4", "MONOTONE:lo=0.2/width=0.2",
        "MONOTONE:lo=0.2/width=0.4"]
R_MAIN = ("DT", "MONO", "FIXED", "NEUTRAL", "LOOKUP", "NOMEM")
R_IDENTITY = ("DT", "POOLED", "SWAPPED", "SHUFFLED")


def mono_r(robot: str) -> str:
    """MONOTONE* re-tuned under matched supervision (remediation RV), from configs/M6_tuned_R.json."""
    return json.loads((REPO / CFG / "M6_tuned_R.json").read_text())[robot]


def r_jobs(stage: str, robots: tuple[str, ...], t1_policy: str) -> list[tuple[str, str, int]]:
    """Remediation corrected comparisons (REMEDIATION_PROTOCOL.md): every system under matched supervision."""
    out = []
    for r in robots:
        margs = model_args(r).replace("experiments/M_release/policies/t1_isaaclab_v1", t1_policy)
        base = f"--robot {r} --scene warehouse --experiment R --supervision matched {margs}"
        if stage == "RV":
            out.append((f"RV_{r}", f"3 0 3 out/RV_{r} -- {base} --partition V --systems {','.join(GRID)} "
                        f"--histories change_B,late_B", 14400))
        else:
            for sysname in R_MAIN:
                sys_ = mono_r(r) if sysname == "MONO" else sysname
                out.append((f"RY_{r}_main_{sysname}", f"3 0 15 out/RY_{r}_main_{sysname} -- {base} --partition Y "
                            f"--systems {sys_} --histories change_B,late_B", 21600))
            for sysname in R_IDENTITY:
                out.append((f"RY_{r}_id_{sysname}", f"3 0 15 out/RY_{r}_id_{sysname} -- {base} --partition Y "
                            f"--systems {sysname} --histories risky_B", 21600))
    return out


def jobs(stage: str) -> list[tuple[str, str, int]]:
    """(tag, m5_par command, timeout s)."""
    out = []
    if stage == "V":
        grid = ["MONOTONE:lo=0.1/width=0.2", "MONOTONE:lo=0.1/width=0.4", "MONOTONE:lo=0.2/width=0.2",
                "MONOTONE:lo=0.2/width=0.4"]
        for r in ROBOTS:
            out.append((f"V_{r}", f"3 0 3 out/V_{r} -- --robot {r} --scene warehouse --experiment M6 --partition V "
                        f"--systems {','.join(grid)} --histories change_B,late_B {model_args(r)}", 9000))
    elif stage == "X":
        for r in ROBOTS:
            systems = MAIN.format(mono=mono(r))
            for half, (first, n) in enumerate(((0, 5), (5, 5))):
                out.append((f"X_{r}_main{half}", f"3 {first} {n} out/X_{r}_main{half} -- --robot {r} --scene "
                            f"warehouse --experiment M6 --partition X --systems {systems} --histories "
                            f"change_B,late_B {model_args(r)}", 21600))
            for scene in ("home", "hospital"):
                out.append((f"X_{r}_{scene}", f"3 10 6 out/X_{r}_{scene} -- --robot {r} --scene {scene} "
                            f"--experiment M6 --partition X --systems DT,{mono(r)},REACTIVE --histories "
                            f"change_B,late_B {model_args(r)}", 14400))
    elif stage == "D":
        for r, scene in itertools.product(ROBOTS, ("warehouse", "hospital")):
            eye = "--record-eye" if r == "nerva" else ""
            out.append((f"D_{r}_{scene}", f"3 0 3 out/D_{r}_{scene} -- --robot {r} --scene {scene} --experiment M7 "
                        f"--partition D --systems DT --histories change_B,late_B,safe --video {eye} "
                        f"{model_args(r)}", 14400))
    elif stage == "F":
        base = "--scene warehouse --experiment M6 --partition X --systems DT --histories change_B"
        checks = {
            "restart": "--restart-at 12", "nocam": "--no-camera", "anon": "--anon-b1",
            "interrupt": "--interrupt 3 2", "dropout": "--camera-dropout 0.3", "veto": "--fault-inject",
            "contact": "--unexpected-contact 0.15", "perturb": "--perturb 0.15",
        }
        for r in ROBOTS:
            for name, flag in checks.items():
                out.append((f"F_{r}_{name}", f"1 40 2 out/F_{r}_{name} -- --robot {r} {base} {flag} "
                            f"{model_args(r)}", 7200))
            out.append((f"F_{r}_long", f"1 50 1 out/F_{r}_long -- --robot {r} {base} --n-enc 100 "
                        f"{model_args(r)}", 14400))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("V", "X", "D", "F", "RV", "RY"), required=True)
    ap.add_argument("--vms", required=True)
    ap.add_argument("--robots", default="nova_carter,humanoid", help="RV/RY: Isaac robots")
    ap.add_argument("--t1-policy", default="experiments/M_release/policies/t1_isaaclab_v1",
                    help="RV/RY: the T1 controller selected by the gait battery")
    a = ap.parse_args()
    vms = a.vms.split(",")
    todo = (r_jobs(a.stage, tuple(a.robots.split(",")), a.t1_policy) if a.stage.startswith("R")
            else jobs(a.stage))
    for i, (tag, cmd, tmo) in enumerate(todo):
        vm = vms[i % len(vms)]
        print(f'JC submit {vm} "$E {tag} {tmo} bash experiments/M_release/cloud/m5_par.sh {cmd}" --tag {tag}')


if __name__ == "__main__":
    main()
