# Milestone C: experience-grounded outcome prediction on NERVA experience

Protocol: [PREREGISTRATION.md](PREREGISTRATION.md) (committed `21948c7` before any label was inspected).
Code: `369ee23`. Results: `results/results.json`. Run: 600 NERVA trials, local CPU, 21 min end to end.

## Data [measured]

- **Source:** 600 simulator trials of NERVA's validated neutral locomotion policy (`examples/nerva/generate_experience.py`,
  seed 2026; 17 min of local simulation).
- **Observed outcomes (adapter detectors):**
  - 276 stability losses;
  - 275 falls;
  - 1 recovery;
  - 11 task interruptions.
- **Recovery is almost absent:** the validated policy almost never recovers once it tilts past 25°, so predicting a
  stability loss is effectively predicting a fall 0.5 s before tilt reaches 25°.
- **Samples:** 42,437 armed training samples with 799 positives (1.9%); test 13,398 samples with 315 positives.
- **Split:** 360 / 120 / 120 trials (train / validation / test).

## Held-out episodes: `stability_loss` within 0.5 s

| model | Brier | log loss | ECE | AUROC | recall @ 5% FPR | false alarms in no-push trials |
|---|---|---|---|---|---|---|
| M0 outcome memory alone (rate per goal type) | 0.0230 | 0.112 | 0.005 | 0.55 | 0.00 | 0 |
| M1 margin-bin frequency | 0.0178 | 0.086 | 0.004 | 0.74 | 0.43 | 0 |
| M2 logistic regression | 0.0167 | 0.083 | 0.004 | 0.74 | 0.44 | 0 |
| M3 adapter tilt-extrapolation heuristic | 0.0172 | 0.107 | 0.015 | 0.75 | 0.44 | 0 |
| **M4 GRU** (64 hidden, 17,857 parameters) | 0.0166 | 0.079 | **0.002** | 0.79 | 0.46 | 0 |
| **M5 GRU + PAD + tonic appraisal** | **0.0152** | **0.073** | 0.002 | **0.83** | **0.52** | 0 |

## Preregistered criteria

| criterion | result |
|---|---|
| C1: GRU Brier ≤ 0.9 × best statistical baseline, CI excluding 0 | **FAIL**: GRU 0.01660 vs logistic 0.01673; difference −0.00013, 95% CI [−0.00073, +0.00057] |
| C2: GRU ECE ≤ 0.05 | pass (0.0018) |
| C3: GRU + affect beats GRU (CI excluding 0) | **pass as specified**: −0.0014, CI [−0.0020, −0.0009] (but see the confound below) |
| C4: GRU ≤ logistic on held-out turn trials | **FAIL**: GRU 0.01298 vs logistic 0.01288 (GRU AUROC 0.82 vs 0.77) |
| C5: no forgetting with replay; final better than stage 0 | passes, but trivially (see incremental learning) |
| corrupted shadow update rejected | pass |

### The C3 confound (important)

M5's extra inputs are the APM's affect state and tonic appraisal vector. The APM computes them from the full 50 Hz
message stream, including the NERVA adapter's tilt-extrapolation hypotheses. The plain GRU sees only 10 Hz
samples of the canonical features. M5's advantage therefore may come from **higher-rate information routed
through the affect machinery, rather than from the PAD representation itself**.

C3 passes exactly as preregistered. It does **not** show that PAD as a representation adds predictive value.

**Exploratory ablation** (post-hoc, `ablation.json`; same data, split, GRU and seed; M5 reproduced exactly):

| added inputs | Brier change vs M4 | 95% CI |
|---|---|---|
| PAD only (V, A, D) | +0.00001 | [−0.00022, +0.00024], **no measurable effect** |
| tonic appraisal vector u only | −0.00127 | [−0.00188, −0.00077] |
| PAD + u (M5) | −0.00140 | [−0.00199, −0.00089] |

**Finding [measured, exploratory]:** PAD adds no measurable predictive benefit in this test.
- The whole gain comes from the appraisal vector u.
- u carries the adapter's 50 Hz tilt-extrapolation risk (as `threat`) and the current body strain.

So the appraisal layer is a useful channel for a robot-side estimate, and PAD as a representation is not supported
here. A fair follow-up would give the plain GRU the same 50 Hz summary directly.

### Secondary target: fall within 1 s (reported only)

| model | Brier | AUROC |
|---|---|---|
| margin bins | 0.0331 | 0.66 |
| logistic | 0.0321 | 0.65 |
| GRU | 0.0318 | 0.73 |

### Held-out context (train non-turn, test turn trials)

| model | Brier | AUROC |
|---|---|---|
| logistic | 0.0129 | 0.77 |
| tilt extrapolation | 0.0134 | 0.77 |
| GRU | 0.0130 | 0.82 |

The GRU ranks events better (AUROC) but is not better calibrated (Brier) than logistic regression, so C4 fails.

## Incremental learning with shadow promotion [measured]

- **Stage 0:** a model trained on 79 early-life trials (latency off; rest, forward and backward). Test Brier 0.01717
  overall, 0.02283 on early life.
- **Replay 50%:** four arrivals; **one promotion** (arrival 1). Arrivals 2–4 were rejected: none improved
  validation by 2% without regressing on early life by more than 5%.
  - The deliberately corrupted shadow (shuffled labels) was rejected.
  - Rollback restored the previous model.
  - Final test Brier 0.01712 overall and 0.02283 on early life: **essentially unchanged from stage 0.**
- **No replay:** **zero promotions.** Every shadow update regressed on the protected early-life validation set
  (+4–9%) and was rejected, so the served model never changed.

**Reading.** The safety mechanics work: frozen evaluation, rejection of a corrupted update, rollback, and the live
model never changing after a gradient step. But **this run does not demonstrate useful continual learning**:
- C5 passes only because almost nothing changed (an overall gain of 0.3%);
- in the no-replay arm, it was the promotion gate, not replay, that prevented forgetting.

## Costs (development PC; `results.json` → costs)

| item | value |
|---|---|
| GRU parameters | 17,857 |
| GRU serialized state | 74.6 kB |
| inference latency | 2.3 ms per single-sample CPU call (PyTorch, includes Python overhead) |
| training time | 102 s for 14 epochs on 42k samples |
| logistic regression | 17 parameters |
| margin bins | 22 parameters |

## Overall

- [measured] A small GRU trained on the robot's own experience predicts imminent stability loss on held-out
  episodes:
  - it is well calibrated (ECE 0.002);
  - it ranks events better than every statistical baseline (AUROC 0.79 vs 0.74);
  - it is **not** better on the preregistered proper-score criterion (C1 fails), and it doesn't generalize better
    to held-out turns (C4 fails).
- [measured] Outcome memory alone (M0) is the weakest model.
- [measured] Adding the APM's affect and appraisal inputs improves prediction, with the confound described above.
- [measured] Shadow learning infrastructure works; meaningful incremental improvement was not demonstrated.
