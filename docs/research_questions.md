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

## Status (2026-10-17, after programmes F, G and H)

| RQ | result |
|---|---|
| RQ1 portability | **demonstrated** in simulation (Milestone D; the H expressive layer also ran on a biped and a wheeled base) |
| RQ2 learning | **demonstrated for consequence prediction:** G's predictor ensemble is the core of decision A. **Not demonstrated for deployment-time weight learning:** G7's promotion gate blocked every update |
| RQ3 affect | **PAD falsified as useful:** no predictive information (F E1), harmful in closed loop under equal tuning (G1/G9: +99% avoidable collisions). An EMA risk filter does PAD's job better |
| RQ4 transfer | **not demonstrated** (D TR2); the software architecture transfers, learned weights do not |
| RQ5 persistent affect | persistence comes from explicit memory and a scalar filter; no recurrent latent beat them (G3, H3) |
| RQ6 dispositions | **demonstrated in simulation** (H4b/H8): identical robots with different histories behave reliably differently (history recovered from motion 98–100%). **Revision after the world changes is slow**, because avoidance censors evidence (F E3, G2): this is the open problem |
| RQ7 expressive legibility (added in H) | **untested** with participants; a researcher self-pilot was negative |

## Status at the end of v0.1 (2026-10-11, superseded above)

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
