# Milestone C preregistration: experience-grounded outcome prediction (fixed before any label is inspected)

Written while the NERVA experience was still being generated (`examples/nerva/generate_experience.py`; 600 trials,
seed 2026). Labels, event counts and model results had not been looked at.

## Data

- **Source:** NERVA's validated neutral candidate in its simulator, 15 s trials:
  - a random neutral command and random action latency;
  - with probability 0.85, one push at U(5, 10) s with speed U(0.2, 1.4) m/s in a random direction.
- **Labels:** observed outcomes only, from the NERVA adapter's fixed detectors (`apm.adapters.nerva`). No risk
  estimate or prediction is ever a label; push parameters are never inputs.
- **Primary target:** `stability_loss` occurs within the next H = 0.5 s.
- **Secondary target (reported, no criterion):** `fall` within 1.0 s.
- **Samples:** every 0.1 s, only while "armed":
  - not during a stability loss (from its onset until the detector's recovery or re-arm);
  - not after a fall.
- **Inputs:** canonical AffectLink-level features computed from the adapter's messages, not NERVA arrays:
  - stability margin and its rate;
  - actuation margin and its rate;
  - energy margin;
  - nearest-entity proximity;
  - closing speed;
  - goal moving.

  Each feature has a validity mask, so missing values are masked, not zeroed. The history window is 2 s
  (20 samples).

## Splits

- **Held-out episodes:** a seeded permutation of trial ids (seed 0) into 60% train, 20% validation, 20% test.
- **Held-out context:** train and validate on trials with non-turn commands; test on turn-command trials
  (`turn_left`, `turn_right`).

## Models (the AffectLink contract is the same for all)

| id | model |
|---|---|
| M0 | outcome memory alone: event rate per goal type (APM's context-memory analogue) |
| M1 | empirical frequency by current stability-margin bin (10 bins) × goal moving, Laplace prior |
| M2 | logistic regression (L2) on the current feature vector |
| M3 | the adapter's tilt-extrapolation heuristic (existing estimator) |
| M4 | GRU, 64 hidden units, one sigmoid head per target, on the 2 s history |
| M5 | M4 plus the APM's affect (PAD) and tonic appraisal vector u, computed by running the APM on the same stream |

M4 and M5 use the same seed, schedule and early stopping on validation Brier score.

## Metrics

All on test sets, with trial-level bootstrap 95% CIs (1,000 resamples):
- Brier score and log loss;
- ECE (10 equal-width bins);
- AUROC;
- recall at 5% false-positive rate (rare adverse detection);
- false associations: the rate of p ≥ 0.5 predictions in trials with **no push** and no observed event;
- parameter count, CPU inference time per sample, training time, memory.

## Criteria (fixed now)

- **C1. Learning beats the simplest statistical baselines.** On held-out episodes, M4's Brier score is
  ≤ 0.9 × the better of M1 and M2, and the bootstrap CI of the difference excludes 0.
- **C2. Calibration.** M4's ECE is ≤ 0.05 on held-out episodes.
- **C3. Does affect add anything?** M5 counts as better than M4 only if its Brier score is lower with a bootstrap
  CI excluding 0. Otherwise the finding is "no measurable benefit from PAD/appraisal inputs", recorded as such.
- **C4. Context generalization.** On held-out turn trials, M4's Brier score is ≤ M2's.

## Incremental learning and shadow promotion

1. **Stage 0:** the validated model (M4) is trained only on "early life": training trials with latency off and
   commands {rest, forward, backward}.
2. **Arrivals:** the remaining training trials arrive in 4 batches, in seeded random order.
3. **Shadow update:** a shadow copy fine-tunes for a fixed 5 epochs on a replay mix of 50% new and 50% sampled
   older experience. The control variant uses 0% older experience.
4. **Frozen evaluation:** on the full validation split and on its early-life subset.
5. **Promotion:** promote only if validation Brier improves by ≥ 2% relative AND early-life validation Brier gets
   no worse than +5% relative. Otherwise reject: the validated model keeps serving, and rollback to the last
   validated model is always available.
6. **Rejection test:** one deliberately corrupted shadow (labels shuffled within the batch) must be rejected.

- **C5. Forgetting.** With replay, the final model's early-life test Brier is ≤ 1.1 × the stage-0 model's, and its
  all-context test Brier is below stage 0's. The no-replay variant is reported for comparison.

## Not claimed

- Loss decreasing during training is not evidence of continual learning; only C1–C5 on held-out data count.
- No counterfactual learning: an avoided event teaches nothing about what would have happened.
