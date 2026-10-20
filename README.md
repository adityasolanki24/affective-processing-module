# APM: Affective Processing Module

**Experience-driven memory, adaptive behaviour and expressive motion across robot embodiments.**

APM is an independent, simulation-based robotics research project. It asks whether a robot can build lasting,
person-specific expectations from its own sensor experience, use them to act more safely and efficiently, and show
what it has learned through how it moves, on top of each robot's own validated motion controller and safety layer.

> **Scope.** All results are in simulation (NVIDIA Isaac Sim 5.1 with PhysX, and MuJoCo). APM's "beliefs" are
> estimates of the physical perturbation to expect from specific people. They are not emotions, and nothing here
> claims feelings, consciousness, production safety or hardware deployment. The expressive mapping is designed, not
> learned end to end, and no human study has tested how people read the movement.

**Snapshot:** `v1.0-sim-snapshot`, the APM v1.0 simulation research snapshot including the October 2026
remediation. Project page: <https://aditya-solanki.vercel.app/projects/apm>. Videos: the
[release assets](https://github.com/adityasolanki24/affective-processing-module/releases/tag/v1.0-sim-snapshot).

## Where it came from

APM began as the affect layer of [NERVA](https://github.com/adityasolanki24/NERVA), an expressive biped built on
Open Duck Mini. NERVA's first design used a continuous PAD (pleasure–arousal–dominance) affective state.

APM's controlled experiments showed that PAD did not improve behaviour; in programme G it made it worse. The
project then moved to what the evidence supported:
- an explicit, revisable memory of *who* caused *what*;
- a decision layer that uses that memory under a hard risk cap;
- a bounded expressive channel that each robot realises with its own controller.

NERVA remains the embodied robot platform. APM is the robot-independent architecture that grew out of it.

## Architecture

```
 sensors (IMU, rendered depth camera; robot-specific)
   │  only robot-observable signals, never simulator state
   ▼
 observation model ──────► encounter evidence: unintended self-displacement, camera looming, quiet-period thresholds
   ▼
 experience memory  ─────► per-person belief over learned behaviour types; Bayesian, change-aware, revisable
   ▼
 decision / regulation ──► decision-theoretic choice of an abstract ExpressionIntent under an enforced risk cap
   ▼                       (proximity, tempo, hesitation, gaze, posture: bounded channels)
 independent risk supervisor (re-checks every decision; can veto → hold)
   ▼
 embodiment adapter ─────► Nova Carter wheels │ T1 arms, waist, head │ NERVA neck and head
   ▼
 the robot's own controller + physical safety monitor (never bypassed by APM)
   ▼
 physical world ─► sensors
```

| part | how it is made |
|---|---|
| per-person memory, behaviour types, sensor thresholds | **learned** from the robot's own experience (EM-fitted types, online Bayesian updates, self-calibrated thresholds) |
| decision layer, risk cap, supervisor | **designed** (decision-theoretic, with a 5-parameter time model fitted from the robot's own exploration) |
| expressive mapping (state → posture, gaze, timing) | **designed**, bounded and rate-limited |
| controllers and physical safety | each robot's own: stock Nova Carter drives; an Isaac Lab PPO policy for T1; NERVA's frozen ONNX policy and SafetyMonitor |

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Robots

| robot | engine | controller | role |
|---|---|---|---|
| NVIDIA Nova Carter (wheeled) | Isaac Sim 5.1 / PhysX | stock wheel drives | comparative experiments and demonstrations |
| Booster T1 (humanoid) | Isaac Sim 5.1 / PhysX | Isaac Lab PPO velocity policy (`policies/t1_isaaclab_v1`) | comparative experiments and demonstrations |
| NERVA / Open Duck Mini v2 (small biped) | **MuJoCo** for comparative results; Isaac Sim for demonstrations | frozen neutral ONNX policy + SafetyMonitor | NERVA cannot reliably tolerate contact in Isaac, so its comparative data come from its validated MuJoCo backend |

## Research contributions

- **One runtime across three morphologies.** The same memory, decision layer and supervisor drive a wheeled robot,
  a humanoid and a small biped through robot-specific adapters.
- **Person-specific memory that pays off.** On fresh seeds with controlled baselines, correct per-person memory
  beats pooled, swapped and shuffled memory on all three robots.
- **Experience-dependent movement.** The same calm meeting looks different after a pushy or a calm history, and the
  differences are measured from joint and pose traces.
- **A research record that changed direction.** Every confirmatory test was preregistered, run once and reported,
  including failures. Negative results retired PAD, a lookup expression mapping, a memory variant and a
  smoother-looking T1 controller (see [docs/RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md)).

## Key results (corrected, October 2026)

Source: [experiments/M_release/REMEDIATION_REPORT.md](experiments/M_release/REMEDIATION_REPORT.md). The figures
are generated from committed results by `experiments/M_release/scripts/make_figures.py`
([figures](experiments/M_release/figures/remediation)).

| question | Nova Carter (Isaac) | Booster T1 (Isaac) | NERVA (MuJoCo) |
|---|---|---|---|
| person-specific memory beats pooled / swapped / shuffled memory (s per encounter, Holm) | +4.1 / +7.5 / +6.4 | +5.0 / +4.8 / +7.2 | +2.5 / +15.0 / +7.9 |
| APM beats the calibrated rule, lookup, fixed style and no memory | yes, all four | yes, all four | yes, all four |
| **APM beats neutral motion with the same supervisor** | **no** (identical behaviour) | **no** (+0.38 s, n.s.) | **no** (0.78 s worse) |
| falls / ejections in 240 lifetimes | 0 / 0 | 0 / 0 | 0 / – |

- **What the comparisons mean.** The task-cost gains come from per-person memory acting through the safety
  supervisor, not from the choice of expressive style.
- **Expression after a pushy vs a calm history** (matched speed and distance):
  - T1 watches longer, guards with its arms, turns side-on and backs away;
  - NERVA lowers or raises its head and neck and backs away;
  - Nova Carter changes its timing and backs away.
- **Learning is fast; recovery is slow.** T1 recovers part-way within 7 calm meetings; Nova Carter and NERVA barely
  do.
- **T1 controller.** The official Booster controller walks more smoothly, but it fell under 0.5 m/s pushes and
  settles more slowly, so it failed the frozen promotion rule and the original controller was kept.

**Results superseded by the remediation.** A simulator defect in the people's lunge contacts (DEVIATIONS D6)
affected programme M's original Isaac results. Those results are preserved in
[experiments/M_release/REPORT.md](experiments/M_release/REPORT.md), and the corrected results above govern. The
corrected comparisons cover a narrower scope (one scene, two histories plus an identity history) than the original
experiments.

## Known limitations

- Simulation only, with no hardware.
- Perception is rendered depth plus motion heuristics, not a deployed detector or tracker.
- In the identity experiments the memory receives the person's identity. Visual re-identification is not tested.
- The expressive mapping is designed, and experience-dependent movement is shown in a limited set of conditions
  (one present-task seed per robot).
- No human evaluation of how the movement is perceived.
- Recovery after a person changes behaviour is incomplete within the tested horizon.
- NERVA cannot reliably take contact in Isaac. Its expression memory is near-miss learning.

## Install and run

```bash
git clone https://github.com/adityasolanki24/affective-processing-module && cd affective-processing-module
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,sim]"                          # sim: MuJoCo 3.3.0
pytest -q && ruff check . && mypy src
```

- **Isaac Sim** (Linux, an RTX GPU such as an NVIDIA L4; pinned Isaac Sim 5.1.0 + Isaac Lab v2.3.2):
  `source experiments/M_release/cloud/isaac_env.sh` installs the pinned environment and exports `ISAAC_PY`.
- **NERVA** needs the NERVA repository and Open Duck Playground at the pinned commits
  (`experiments/M_release/cloud/setup_nerva_inputs.sh`).
- **The Booster T1 candidate controller** (not adopted) is fetched at a pinned commit, with hashes checked, by
  `python experiments/M_release/scripts/prepare_booster.py`.

| what | command |
|---|---|
| integration check (runtime, memory, veto, restart, replay; MuJoCo) | `python experiments/M_release/scripts/m1_integration.py` |
| NERVA lifetimes on MuJoCo | `python experiments/M_release/scripts/m6_nerva_mujoco.py --partition P --n 1 --systems DT --out out/nerva.jsonl` |
| one robot meeting people in Isaac | `$ISAAC_PY experiments/M_release/isaac/m4_encounter.py --body nova_carter --scene warehouse --video --out out/m4` |
| expression run (both histories) | `HISTORIES="pushy calm" SEED=73302 bash experiments/M_release/cloud/expression_v2.sh nerva` |
| corrected comparisons | `python experiments/M_release/cloud/m6_plan.py --stage RY --vms <vm> --tuned M6_tuned_R6.json`, then `scripts/r_analyze.py` |
| figures | `python experiments/M_release/scripts/make_figures.py` |

More commands are in [REMEDIATION_REPORT.md §7](experiments/M_release/REMEDIATION_REPORT.md).

## Documentation

| document | contents |
|---|---|
| [docs/STATE.md](docs/STATE.md) | what works today and what does not |
| [docs/RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md) | programmes A–M: questions, decisive results, retired ideas |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components, learning, safety boundaries |
| [experiments/M_release/REMEDIATION_REPORT.md](experiments/M_release/REMEDIATION_REPORT.md) | the corrected v1.0 results (current) |
| [experiments/M_release/REPORT.md](experiments/M_release/REPORT.md) | the original v1.0 release report (partly superseded) |
| [experiments/M_release/DEVIATIONS.md](experiments/M_release/DEVIATIONS.md) | every deviation from the preregistered protocols |

## Code layout

| package | role |
|---|---|
| `src/apm_transfer` | integrated runtime, memory, experience state, expressive program, rendered vision, MuJoCo worlds |
| `src/apm_isaac` | Isaac Sim integration: robots, NERVA import, people, scenes and encounters, the Isaac Lab T1 task |
| `src/apm_express` | `ExpressionIntent`, expression adapters, safety monitor |
| `src/apm_revise` | learned behaviour types, change-aware beliefs |
| `src/apm` | AffectLink protocol (the runtime's message contract) |
| `experiments/<programme>/` | preregistrations, scripts, results and reports per programme |

## Principles

- Preregistered confirmatory tests on fresh seeds; deviations logged; failures kept.
- Robot-observable evidence only; simulator truth is never an input.
- APM never bypasses a robot's own controller or safety layer.

Licence: none chosen yet; all rights reserved.
