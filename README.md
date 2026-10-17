# APM — Affective Processing Module

APM is a portable, experience-grounded affective regulation and learning system for robots. A robot connects to it
through **AffectLink**, a small versioned contract of robot-relevant facts:
- **inputs:** entities, self-state margins, goals, predicted outcomes and observed outcomes;
- **outputs (advisory):** persistent affect, appraisal frames, per-target action tendencies, attention, expression
  intent and learned expectations.

Status: **research prototype (private).** The v0.1 core is implemented and tested. Programmes F, G and H (v0.2–v0.4)
then tested which parts of it earn their place; their findings are summarised under
[What the research found](#what-the-research-found). Implemented and measured results are listed separately from
proposals.

**In one paragraph.** What works is simple:
- a learned consequence predictor;
- explicit, outcome-grounded memory of each entity;
- a smoothed risk signal, combined by a transparent rule.

The emotion-like layer of v0.1 (the PAD affect state) did **not** earn its place. It added no predictive information
and made behaviour worse once baselines were tuned equally. Memory-conditioned expressive motion is mechanically
reliable in simulation, but has not been shown to be legible to people.

Programme I (v0.5) solved the revision problem in simulation:
- the robot learns person *types* from self-observed near-misses, without labels, and holds change-aware beliefs
  about them;
- it revises its view of an individual within 2–3 episodes after a change;
- it halves avoidable collisions.

Revision had been slow mainly because collisions are too rare to learn from, not because of avoidance.

## What APM is not

- Not an emotion-recognition product. It does not infer human emotions from faces or voices.
- Not a consciousness detector, and not a claim that a robot feels anything. It implements *functional* affect: an
  internal control state shaped by appraisal and experience.
- Not a motor controller and not a safety system. Every output is advisory; the robot must behave safely with APM
  absent ([docs/safety.md](docs/safety.md)).
- Not an LLM wrapper. The core runs offline on NumPy; no cloud or foundation model is in the affect, motor or
  safety path.

## Why it exists

To test whether one set of affective learning mechanisms can give different robot bodies adaptive, persistent,
context-sensitive behaviour from their own experiences:
- appraisal of explicit predicted and observed outcomes;
- persistent affect dynamics;
- outcome-grounded memory;
- learned outcome prediction.

A common API alone does not make anything universal, so portability, learning and transfer are tested separately.
Each has preregistered criteria ([docs/research_questions.md](docs/research_questions.md)).

## Architecture

```
robot (any body) ──► embodiment adapter ──► AffectLink v1 ──► APM core ──► advisory outputs ──► robot planner
                     (robot-specific)        (versioned)      │                                   (owns safety)
                                                              ├─ StreamGate: validation, ordering, staleness, resets
                                                              ├─ Context: entities, calibrated margins, goals, predictions
                                                              ├─ Appraisal frames (hypothesis / memory / outcome)
                                                              ├─ Affect: tonic attractor + phasic impulses → PAD readout
                                                              ├─ Tendencies (per target), attention, expression intent
                                                              ├─ Memory: entity/context outcome statistics, episodes
                                                              └─ Learning (optional): outcome predictors, shadow learner
```

Details: [docs/architecture.md](docs/architecture.md). The learned predictor is replaceable (GRU today; a
Transformer, an active-inference model or a neuromorphic implementation later) without changing AffectLink.

## AffectLink

- **Schema:** Protocol Buffers ([proto/affectlink/v1/affectlink.proto](proto/affectlink/v1/affectlink.proto)), with a
  Python reference implementation (`apm.protocol`) checked against it field by field.
- **Rules:**
  - units, uncertainty and validity status on every value;
  - missing values are never zero;
  - predictions and outcomes are distinct types;
  - per-stream ordering, duplicate rejection, staleness, reset epochs and size limits.
- **Transports:** the in-process API and JSON codec are implemented. Mappings for a C ABI, ROS 2, Ethernet, CAN FD
  and UART are documented but not implemented ([docs/affectlink.md](docs/affectlink.md)).

## Quick start (no robot, no simulator)

```bash
git clone https://github.com/adityasolanki24/affective-processing-module && cd affective-processing-module
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # add ".[learning]" (PyTorch CPU) for the learning stack
python examples/standalone/scenario.py              # synthetic robot; writes runs/standalone.jsonl
python examples/wheeled_robot/run.py                # second embodiment: simulated wheeled robot, 10 minutes
pytest -q
```

## Examples

| example | what it shows | needs |
|---|---|---|
| `examples/standalone/scenario.py` | synthetic robot: collision with a stranger; on the next visit, memory predicts adverse outcomes, valence falls and an avoid tendency appears before anything happens | APM only |
| `examples/wheeled_robot/run.py` | the same core on a differential-drive robot with obstacles, pedestrians, slippery floor and battery; learned per-pedestrian expectations | APM only |
| `examples/nerva/replay_gate.py` | 160 stored rollouts of the NERVA biped (MuJoCo) through the NERVA adapter, observe-only | stored NERVA telemetry |
| `examples/nerva/live_rollout.py` | one live NERVA simulation with a push: stability loss predicted ≈ 140 ms ahead, then loss, fall and APM's response | a NERVA checkout and its simulator environment |
| `examples/nerva/generate_experience.py` | randomized-push NERVA experience for learning | same |

## Robot integration

Write an embodiment adapter ([docs/robot_integration.md](docs/robot_integration.md)):
1. Declare capabilities and margin calibrations in an `EmbodimentProfile`.
2. Send calibrated `SelfState`, plus entities, goals and predictions if you have them.
3. Detect real outcomes and send `OutcomeSignal`, documenting each threshold.
4. Use `Emitter` for headers.

There are two worked adapters: `apm.adapters.nerva` (biped) and `apm.adapters.wheeled`.

## Learning

See [docs/learning.md](docs/learning.md):
- **Statistical baselines:** goal rate, margin bins, logistic regression.
- **GRU outcome predictor:** 64 hidden units, 17,857 parameters.
- **Labels:** observed outcomes only; no counterfactual credit.
- **Replay buffer and shadow learner:** frozen evaluation, promotion thresholds and rollback; the live model never
  changes after a gradient step.

## Hardware

See [docs/hardware.md](docs/hardware.md). It covers Raspberry Pi 5-class, NXP i.MX 8M Plus-class and Jetson Orin
Nano-class candidates, failure behaviour, a block diagram and a bench test procedure.
- **No hardware was purchased or tested.**
- Runtime numbers are measured on a development PC; board figures are estimates.

## What the research found

Every programme was preregistered: criteria were fixed before confirmatory data, and negative results were kept.

| programme | question | decision / result | report |
|---|---|---|---|
| A–E (v0.1) | build the core; portability; learning; transfer; hardware | portable across two bodies; learning mixed; **no learned transfer**; PAD adds no predictive value | `experiments/{B,C,D,E}_*` |
| F (v0.2) | does the existing APM improve behaviour? | v0.1's PAD layer cut collisions 25% vs the B1 baseline; G later traced this to B2's 9× larger tuning budget. Memory revision too slow (E3); learned regulator B3 not credible (E4) | `experiments/F_behavioral_learning/results/CONFIRMATORY_SUMMARY.md` |
| G (v0.3) | smallest architecture that behaves better through experience | **Decision A:** predictor ensemble + entity memory + EMA risk penalty. PAD **harmful** under equal tuning (+99% avoidable collisions); neural policies and specialised regulators not better; no validated deployment-time weight learning | `experiments/G_architecture_selection/reports/ARCHITECTURE_DECISION.md` |
| H (v0.4) | can experience be expressed through embodied movement? | **Decision B:** memory + fixed lookup → bounded adapter on NERVA's validated controller; lived history recoverable from motion 98–100%, 0 falls. No learned or specialised model beat it. **Human legibility untested**; a researcher self-pilot found the styles hard to tell apart | `experiments/H_expressive_intelligence/reports/EXPRESSIVE_ARCHITECTURE_DECISION.md` |
| I (v0.5) | can the robot revise beliefs about individuals autonomously, including under self-selected avoidance? | **Revision achieved:** dense self-observed evidence (near-miss clearance) + EM-learned person types + change-aware beliefs. Revises in 2–3 episodes (A0: never); avoidable collisions −56%; completion −0.3 pp. Outright memory-driven keep-away creates a real trap; probing fixes it, but targeting adds little. G2b's memory found inverted | `experiments/I_autonomous_revision/reports/AUTONOMOUS_REVISION_DECISION.md` |

**Open problems, in order:**
1. **Second embodiment and robustness for programme I's memory.** NERVA, with graded perturbation as the dense
   outcome; G6's shift conditions; less separable person types.
2. **Deployment-time learning.** It needs a statistically powered promotion gate (G7).
3. **Learned cross-embodiment transfer.** Not demonstrated.
4. **Human legibility of expression.** The H2 study is ready but needs an ethics determination, and probably a
   richer motion vocabulary from NERVA.

## Project status

| area | state |
|---|---|
| AffectLink v1, codec, validation, ordering, staleness, resets | implemented, tested |
| core: context, appraisal frames, attractor affect, tendencies, attention, expression intent, diagnostics | implemented, tested (hand-set transparent baseline, not fitted) |
| memory: entity/context outcome statistics, memory-derived predictions, ambiguous identities, atomic persistence | implemented, tested |
| NERVA biped adapter: replay + live rollout | **measured** (Milestone B) |
| learning on NERVA experience | **measured** (Milestone C), mixed: see below |
| second embodiment (wheeled) + portability | **measured** (Milestone D): same core unchanged |
| cross-embodiment learned transfer | **not demonstrated** (Milestone D, TR2 failed) |
| embedded hardware | design + PC benchmark only (Milestone E) |
| research harnesses for F/G/H (`apm_research`, `apm_select`, `apm_express`, `apm_testbeds.wheeled_ctrl`) | implemented, tested; used by the experiment scripts |
| decision-A regulator as a production runtime | **not yet**: it exists as research code in `apm_select`, while the v0.1 core still carries PAD |
| programme I revising memory (`apm_revise`) | implemented, tested; research code, not yet in the core |

## Milestone A–E results (v0.1)

All preregistered; details in `experiments/*/README.md`. Later programmes, above, supersede the PAD and learning
conclusions here.

| question | result |
|---|---|
| software portability | **demonstrated**: the same core and AffectLink ran unmodified on a simulated biped (NERVA) and a simulated wheeled robot; core file hashes unchanged |
| experience-grounded learning | **partially demonstrated** (see below) |
| PAD's predictive value | **not supported**: in an ablation, PAD added nothing measurable; the appraisal vector carried the gain |
| learned transfer (biped → wheeled) | zero-shot ranking carried over (AUROC 0.83); fine-tuning did not improve sample efficiency, so **no transfer claim** |
| persistent functional affect | bounded, rate-invariant, recovering and memory-carrying affect is **demonstrated in simulation** |

Learning detail:
- **Biped:**
  - calibrated GRU with better ranking than baselines (AUROC 0.79 vs 0.74);
  - failed the preregistered Brier margin and held-out-context criteria;
  - shadow learning was safe but nearly a no-op.
- **Wheeled robot:** the GRU clearly beat logistic regression (slip Brier −17%, collision Brier −38%).

Persistent-affect caveat: the hand-set baseline sat chronically negative on the busy wheeled robot, a calibration
limitation.

## Reproducibility

| step | command | notes |
|---|---|---|
| tests | `pytest -q` | core; `-m learning` needs torch |
| Milestone B | `python examples/nerva/replay_gate.py --gate-dir <NERVA gate rollouts>` | |
| Milestone C | `python examples/nerva/generate_experience.py --nerva <NERVA> --policy <candidate.onnx> --trials 600`, then `python experiments/C_learning/run_learning.py --experience runs/nerva_experience` | seeds fixed in the scripts; data seed 2026 |
| Milestone D | `python experiments/D_portability/run_portability.py --nerva-experience runs/nerva_experience` | wheeled seeds 10,000–10,399 |
| Milestone E | `python experiments/E_hardware/benchmark.py` | |
| programmes F, G, H, I | see each programme's `README.md` (I's cloud runs: `experiments/I_autonomous_revision/cloud/`) | frozen seeds in each `seeds.py`; confirmatory runs refuse to start on a dirty checkout |

Raw data and logs stay local under `runs/` (ignored). Committed results are the JSON summaries in `experiments/`.
The exception is the small F snapshot in `runs/E1` and `runs/E2`, which F's preregistration cites by hash.

Confirmatory runs record the exact commit and source hashes they ran with, in each `run_meta.json`. Reproduce a
result at that commit. Later maintenance (for example the 2026-10-17 cleanup) changes library source hashes, but not
the behaviour the experiments used.

## Licence

No licence has been chosen yet; all rights reserved until a licence is selected.
