# APM — Affective Processing Module

APM is a portable, experience-grounded affective regulation and learning system for robots. A robot connects to
it through **AffectLink**, a small versioned contract of robot-relevant facts:
- **inputs:** entities, self-state margins, goals, predicted outcomes, observed outcomes;
- **outputs (advisory):** persistent affect, appraisal frames, per-target action tendencies, attention, expression
  intent and learned expectations.

Status: **research prototype (v0.1, private).** See [Project status](#project-status) for what is implemented and
what is measured.

## What APM is not

- Not an emotion-recognition product. It does not infer human emotions from faces or voices.
- Not a consciousness detector, and not a claim that a robot feels anything. It implements *functional* affect: an
  internal control state shaped by appraisal and experience.
- Not a motor controller and not a safety system. Every output is advisory; the robot must behave safely with
  APM absent ([docs/safety.md](docs/safety.md)).
- Not an LLM wrapper. The core runs offline on NumPy; no cloud or foundation model is in the affect, motor or
  safety path.

## Why it exists

To test whether one set of affective learning mechanisms can give different robot bodies adaptive, persistent,
context-sensitive behaviour from their own experiences. The mechanisms are:
- appraisal of explicit predicted and observed outcomes;
- persistent affect dynamics;
- outcome-grounded memory;
- learned outcome prediction.

A common API alone does not make anything universal, so portability, learning and transfer are each tested
separately ([docs/research_questions.md](docs/research_questions.md)).

## Architecture

```
robot (any body) ──► embodiment adapter ──► AffectLink v1 ──► APM core ──► advisory outputs ──► robot planner
                     (robot-specific)        (versioned)      │                                   (owns safety)
                                                              ├─ StreamGate: validation, ordering, staleness, resets
                                                              ├─ Context: entities, calibrated margins, goals, predictions
                                                              ├─ Appraisal frames (hypothesis / memory / outcome)
                                                              ├─ Affect: tonic attractor + phasic impulses → PAD
                                                              ├─ Tendencies (per target), attention, expression intent
                                                              ├─ Memory: entity/context outcome statistics, episodes
                                                              └─ Learning (optional): outcome predictors, shadow learner
```

Details: [docs/architecture.md](docs/architecture.md).

## AffectLink

- **Schema:** Protocol Buffers ([proto/affectlink/v1/affectlink.proto](proto/affectlink/v1/affectlink.proto)), with
  a Python reference implementation (`apm.protocol`) checked against it by tests.
- **Rules:**
  - every value has units, uncertainty and a validity status;
  - missing values are never zero;
  - predictions and outcomes are distinct types;
  - per-stream ordering, duplicate rejection, staleness, reset epochs and size limits.
- **Transports:** the in-process API and JSON codec are implemented. Mappings for a C ABI, ROS 2, Ethernet, CAN FD
  and UART are documented in [docs/affectlink.md](docs/affectlink.md) but not implemented.

## Quick start (no robot, no simulator)

```bash
git clone <this repository> && cd affective-processing-module
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python examples/standalone/scenario.py              # synthetic robot; writes runs/standalone.jsonl
pytest -q
```

The standalone scenario is synthetic. In the run summarised below, an unfamiliar person collides with the robot;
on that person's next visit, memory expects adverse outcomes before anything happens. That lowers valence, raises
arousal and produces an avoid tendency toward that person, and the robot's memory does not learn from its own
prediction.

## Writing an adapter for your robot

Start with [docs/robot_integration.md](docs/robot_integration.md):
1. Declare an `EmbodimentProfile`.
2. Convert your signals into `SelfState` margins, using calibrations.
3. Emit `OutcomeSignal`s only for outcomes you can actually detect.
4. Stamp headers with `apm.adapters.base.Emitter`.

## Project status

| area | state |
|---|---|
| AffectLink v1 schema, codec, validation, ordering, staleness, resets | implemented, tested |
| core: context, appraisal frames, attractor affect, tendencies, attention, expression intent, diagnostics | implemented, tested (transparent baseline, not fitted to data) |
| memory: entity/context outcome statistics, ambiguous identities, episodes, atomic persistence | implemented, tested |
| standalone synthetic demo | implemented |
| NERVA embodiment adapter | Milestone B (see docs/roadmap.md) |
| outcome predictors (statistical, GRU), replay, shadow learning | Milestone C |
| second robot (wheeled) and portability/transfer experiments | Milestone D |
| embedded hardware design | Milestone E |

No licence has been chosen yet; all rights reserved until a licence is selected.
