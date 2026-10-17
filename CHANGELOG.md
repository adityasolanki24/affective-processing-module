# Changelog

All notable changes; the project follows docs/versioning.md.

## [Unreleased]
### Added (programmes F–H, v0.2–v0.4 research code)
- `apm_research`: the F behavioural-learning harness (features, datasets, predictor ensemble, policies, RL, two-level
  statistics, provenance).
- `apm_testbeds.wheeled_ctrl`: the controllable wheeled testbed used by F and G.
- `apm_select`: G's architecture-selection code (regulators, revisable memory, stateful predictors, minimal variants,
  shift variations, tuning, survival and bootstrap statistics).
- `apm_express`: H's removable expressive layer (`ExpressionIntent`, biped and wheeled adapters, safety monitor,
  experience states, developmental testbed).
- H2 survey instrument (researcher pilot only).

### Fixed
- Type errors that made CI fail on the research branches.

### Added (v0.1)
- AffectLink v1 schema (`proto/affectlink/v1/affectlink.proto`) and the Python reference implementation: messages,
  JSON codec, validation, stream gate (ordering, duplicates, staleness, reset epochs, size limits).
- APM-0 core: bounded context with calibrated margins, transparent appraisal frames, attractor affect with phasic
  impulses, per-target tendencies, attention, expression intent, diagnostics.
- Outcome-grounded memory (entity and context statistics, ambiguous identity credit, episodes) with atomic
  persistence and corrupt-state fallback.
- Standalone synthetic example; CI (ruff, mypy, pytest).
- NERVA embodiment adapter (arrays only; measured stability margin and actuation stress; documented detectors for
  stability loss, fall, recovery and task interruption; tilt-extrapolation estimator), stored-telemetry replay and a
  bounded live rollout (observe-only).
- Memory-derived internal predictions; seek-assistance control fallback; periodic-only staleness.
- Learning stack: canonical masked features, observed-outcome datasets, statistical baselines, GRU predictor,
  evaluation with grouped bootstrap, replay buffer and shadow learner with promotion and rollback.
- Second embodiment: a wheeled-robot testbed (`apm_testbeds.wheeled`) and its adapter.
- Embedded hardware design document and local resource benchmark.
