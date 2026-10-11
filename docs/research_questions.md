# Research questions, hypotheses and falsification criteria

- **RQ1 (portability):** can one unmodified core and protocol run on materially different embodiments?
  - Falsified if a second embodiment needs core changes or robot-specific fields in AffectLink.
  - Passing this is an engineering result, not evidence of learned transfer.
- **RQ2 (experience-grounded learning):** does a predictor trained on a robot's own experiences predict
  robot-relevant consequences on held-out episodes better than the simplest statistical baseline?
  - Metrics: Brier score, calibration, rare adverse-event detection.
  - Falsified if it doesn't beat the baseline beyond seed variation, or if it loses that edge after incremental
    updates (forgetting).
- **RQ3 (does affect add anything?):** compare
  1. outcome memory alone,
  2. a predictor without PAD,
  3. a predictor with appraisal and affect inputs.

  If the PAD/appraisal inputs add no measurable predictive value, that is recorded; PAD is not protected from
  falsification.
- **RQ4 (cross-embodiment transfer):** for semantically compatible outcomes, does pretraining on one robot improve
  sample efficiency or quality on another?
  - The experiment and criteria are preregistered before held-out results are seen.
- **RQ5 (persistent functional affect):** does affect stay bounded and rate-invariant, recover, and carry learned
  expectations across encounters and restarts? Tested now with synthetic and real telemetry.
- **RQ6 (individual dispositions):** do two robots with different histories develop measurably different,
  stable expectations and tendencies in the same situation?
  - Requires multiple runs; a single anecdote is not evidence.

## Status (2026-10-11)

| RQ | result |
|---|---|
| RQ1 portability | **demonstrated** in simulation: biped and wheeled robot on the unmodified core (Milestone D, P1/P2) |
| RQ2 learning | **partial**: biped GRU calibrated with better ranking, but failed the preregistered Brier margin (C1) and context criterion (C4); wheeled GRU beat logistic regression clearly (−17% slip, −38% collision Brier); incremental shadow learning safe but nearly a no-op |
| RQ3 affect | C3 passed (affect + appraisal inputs helped), but an exploratory ablation showed **PAD adds nothing measurable**; the appraisal vector carries the gain |
| RQ4 transfer | **not demonstrated**: zero-shot ranking transferred (AUROC 0.83), fine-tuning did not improve sample efficiency (TR2 failed) |
| RQ5 persistent affect | bounded, rate-invariant, recovering, memory-carrying in tests and both embodiments; the hand-set baseline is chronically negative on the busy wheeled robot (calibration limitation) |
| RQ6 dispositions | per-identity expectations diverged within one wheeled run (collision p 0.01–0.16); multi-robot, multi-run evidence not yet collected |

## Limitations stated up front
- All affect and appraisal constants are hand-set design values.
- No human evaluation of expressive or social effects.
- Simulation only until a hardware milestone is executed.
