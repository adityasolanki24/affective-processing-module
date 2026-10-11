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

## Limitations stated up front
- All affect and appraisal constants are hand-set design values.
- No human evaluation of expressive or social effects.
- Simulation only until a hardware milestone is executed.
