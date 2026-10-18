# APM: experience-driven expressive behaviour for robots

APM is a research system for robots that:
- learn beliefs about the entities they meet from their own sensor experience;
- keep those beliefs as explicit, revisable memory;
- express them through bounded changes in how they move, on top of a validated motion controller and an independent
  safety layer.

It is **research code, simulation only.** It makes no claim that robots have emotions.

**Read first:** [docs/STATE.md](docs/STATE.md), which says what works, what does not, and what is current. All docs:
[docs/README.md](docs/README.md).

## What works today (simulation)

- A NERVA biped (MuJoCo) learns, from its own IMU only, which entities push it. It changes its approach (stopping
  distance, path, timing) accordingly. Identical robots with different histories move differently.
- On a wheeled testbed, the same memory revises beliefs about individual people within 2–3 episodes and halves
  avoidable collisions.
- **What does not work yet:** after bad experience, NERVA does not recover when the entity becomes harmless (the
  avoidance trap). Programme K addresses this. Human legibility is untested.

## Install

```bash
git clone https://github.com/adityasolanki24/affective-processing-module && cd affective-processing-module
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # ".[learning]" adds PyTorch (CPU)
pytest -q && ruff check . && mypy src
```

NERVA experiments also need the NERVA repository and its simulation venv (MuJoCo, ONNX Runtime, Open Duck
Playground). Set `NERVA_ROOT=<NERVA checkout>` and `PYTHONPATH="src;<NERVA>"` (`:` on Linux).

## Run a demonstration

| what | command | needs |
|---|---|---|
| synthetic robot, no simulator | `python examples/standalone/scenario.py` | APM only |
| wheeled robot with learned per-person expectations | `python examples/wheeled_robot/run.py` | APM only |
| NERVA closed loop: memory → expression → physical outcomes (J5/J6 protocol) | `python experiments/J_validation_transfer/scripts/j5_closed_loop.py --partition P --n 2 --types-model experiments/J_validation_transfer/configs/J5_types_distance.json --edges experiments/J_validation_transfer/configs/J4_edges.json --lam 0.02 --systems CL --out runs/demo.jsonl` | NERVA |
| paired NERVA videos (risky vs safe history) | `python experiments/J_validation_transfer/scripts/j_videos.py --runs <J6-X lifetimes> --out <dir>` | NERVA, ffmpeg |

Recorded videos are in `experiments/J_validation_transfer/videos/`.

## Where the code lives

| package | role |
|---|---|
| `src/apm_revise` | experience memory: EM-learned entity types, change-aware beliefs (robot-independent) |
| `src/apm_transfer` | NERVA observation model, NERVA memory, decision-theoretic expression mapping, stress tests |
| `src/apm_express` | `ExpressionIntent`, L0 mapping, NERVA and wheeled expression adapters, safety monitor |
| `src/apm_select`, `src/apm_research` | consequence predictors, regulators, statistics (programmes F/G) |
| `src/apm_testbeds` | wheeled testbed with pedestrians and a safety supervisor |
| `src/apm` | v0.1 runtime: AffectLink protocol (current contract) and PAD core (superseded) |
| `experiments/<programme>/` | preregistrations, scripts, results and reports per programme (historical records) |

## Rules this project follows

- Preregistered confirmatory tests, run once from a committed tree.
- Every failure is kept.
- Robot-observable evidence only; simulator truth is never an input.
- APM never bypasses the robot's safety layer.

Licence: none chosen yet; all rights reserved.
