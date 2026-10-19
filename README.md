# APM: experience-driven expressive behaviour for robots

APM is a research system for robots that:
- learn beliefs about the people they meet from their own sensor experience;
- keep those beliefs as explicit, revisable memory;
- express them through bounded changes in how they move, on top of a validated motion controller and independent
  safety layers.

It is **research code, simulation only.** Its beliefs are estimates of expected physical perturbation from specific
people, not emotions. It makes no claim that robots have feelings, and no human has yet rated its expression.

**v1.0 (this branch):**
- one integrated runtime;
- three robots in NVIDIA Isaac Sim 5.1: NVIDIA Nova Carter, Booster T1, and NERVA (Open Duck Mini v2);
- realistic scenes with animated, physically present people;
- rendered depth vision;
- preregistered experiments A–F;
- four videos.

The release report, with every claim linked to evidence, is
[experiments/M_release/REPORT.md](experiments/M_release/REPORT.md). Read [docs/STATE.md](docs/STATE.md) first for
what works and what does not.

## What works (simulation)

- The same runtime (memory, decision-theoretic expression, risk supervisor, persistence, replay, AffectLink) drives
  all three robots. Each robot keeps its own validated controller and physical safety layer.
- **Isaac Sim (PhysX):**
  - Nova Carter and T1 meet people in a warehouse, a home and a held-out hospital;
  - they sense them with an IMU and a rendered depth eye (no simulator state), learn who pushes, and change how
    they approach.
- **NERVA:**
  - imported into Isaac Sim with kinematics matching MuJoCo exactly; its frozen controller walks there and passes
    its gate battery;
  - it cannot walk in the home scene, and it had no valid Isaac model. Its comparative data therefore come from its
    validated MuJoCo backend, clearly labelled (DEVIATIONS D3).
- Results, with confidence intervals: [REPORT.md](experiments/M_release/REPORT.md).

## Install

```bash
git clone https://github.com/adityasolanki24/affective-processing-module && cd affective-processing-module
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,sim]"                          # sim: MuJoCo 3.3.0
pytest -q && ruff check . && mypy src
```

**Isaac Sim** (Linux, an RTX GPU such as an L4; pinned Isaac Sim 5.1.0 + Isaac Lab v2.3.2):
`experiments/M_release/cloud/isaac_env.sh` builds or restores the environment and exports `ISAAC_PY`.

**NERVA** needs the NERVA repository and Open Duck Playground at the pinned commit
(`experiments/M_release/cloud/setup_nerva_inputs.sh`, `nerva_mujoco_env.sh`).

## Run

| what | command |
|---|---|
| integration check (runtime, sensors, memory, veto, restart, replay, AffectLink; MuJoCo) | `python experiments/M_release/scripts/m1_integration.py` |
| one robot meeting people in an Isaac scene | `$ISAAC_PY experiments/M_release/isaac/m4_encounter.py --body nova_carter --scene warehouse --video --out out/m4` |
| an APM lifetime in Isaac (any system, partition, scene) | `$ISAAC_PY experiments/M_release/isaac/m5_lifetime.py --robot humanoid --policy-dir experiments/M_release/policies/t1_isaaclab_v1 --scene hospital --experiment M6 --partition P --n 1 --systems DT --histories change_B --types-model experiments/M_release/configs/M5_humanoid_types.json --edges experiments/M_release/configs/M5_humanoid_edges.json --time-model experiments/M_release/configs/M5_humanoid_time.json --task-config experiments/M_release/configs/M5_humanoid_tasks.json --out out/demo.jsonl` |
| NERVA lifetimes on its MuJoCo backend | `python experiments/M_release/scripts/m6_nerva_mujoco.py --partition P --n 1 --systems DT --out out/nerva.jsonl` |
| gate batteries | `isaac/m3_nerva_battery.py`, `isaac/m3_lab_battery.py`, `isaac/m3_wheeled.py` |
| analysis | `python experiments/M_release/scripts/m6_analyze.py --runs <dirs> --tasks <robot>=<tasks.json> ... --out <json>` |

## Where the code lives

| package | role |
|---|---|
| `src/apm_transfer` | integrated runtime (`runtime`), memory, expression mapping (`infoseek`), rendered vision (`vision`), MuJoCo worlds |
| `src/apm_isaac` | Isaac Sim integration: robots (Nova Carter, T1 Isaac Lab policy, reference wheeled), NERVA import and controller (`nerva`), people (`agents`), scenes and encounters (`world`), runtime body (`body`), Isaac Lab T1 task (`lab`) |
| `src/apm_express` | `ExpressionIntent`, expression adapters, safety monitor |
| `src/apm_revise` | learned entity types, change-aware beliefs |
| `src/apm` | AffectLink protocol (the runtime's message contract) |
| `experiments/<programme>/` | preregistrations, scripts, results and reports per programme; v1.0 is `experiments/M_release` |

## Rules this project follows

- Preregistered confirmatory tests on fresh seeds; deviations logged; every failure kept.
- Robot-observable evidence only; simulator truth is never an input.
- APM never bypasses a robot's own controller or safety layer.

Licence: none chosen yet; all rights reserved.
