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

- A NERVA biped (MuJoCo) learns from its own sensors which entities disrupt it, and changes how it approaches.
- **It revises those expectations when the entity changes, and its behaviour recovers.** A wheeled robot and a
  humanoid (Booster T1) do the same with the same code.
- With a (modelled) camera, it gathers information safely, by watching from beyond reach before approaching again.
- Expression is chosen by a small decision-theoretic mapping with task values fixed from requirements. It beats the
  earlier hand-written mapping in 4/4 tasks.
- **Not yet:** a real camera on NERVA, hardware, human legibility, worlds with overlapping entity behaviours.

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
| **best demo:** two identical NERVA robots, different histories, one continuous lifetime each (experience → belief → expression → outcome → revision → recovery) | `python experiments/K_recovery/scripts/k_demo.py --system DT_CAM_VOI --render --out runs/demo` | NERVA, ffmpeg |
| recovery study on the wheeled robot (fast) | `python experiments/K_recovery/scripts/k_run.py --robot wheeled --experiment K5 --partition P --n 2 --systems DT,DT_CAM_VOI --transition-fix --types-model experiments/K_recovery/configs/K_wheeled_types.json --edges experiments/K_recovery/configs/K_wheeled_edges.json --time-model experiments/K_recovery/configs/K_wheeled_time.json --out runs/wheeled.jsonl` | MuJoCo |

Recorded video: `experiments/K_recovery/videos/K4_two_robots_continuous.mp4`.

## Where the code lives

| package | role |
|---|---|
| `src/apm_revise` | experience memory: EM-learned entity types, change-aware beliefs (robot-independent) |
| `src/apm_transfer` | memory (`nerva_memory`), expression mapping and information gathering (`expression_map`, `infoseek`), worlds: NERVA (`nerva_world`, `nerva_continuous`), wheeled (`wheeled_world`), humanoid (`humanoid_world`) |
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
