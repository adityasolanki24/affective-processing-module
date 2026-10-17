# Programme J deviations

Dated entries only. The master preregistration is never edited.

## D1 (2026-10-18, after J2-X analysis): analysis bug in the J2 survival check, corrected

**The bug.** `j2_analyze.py` evaluated the careful-misclassification criterion as `(estimate or 1) <= 0.20`. An
estimate of exactly 0.00 was therefore treated as missing and *failed*. The same pattern existed for AUROC
(`or 0`), but it could not cause a wrong result there.

**Effect.** EU_I was wrongly failed in `fast_pedestrians` and `speed_limit_0.6`, where its misclassification is 0.00.

**Fix.** An explicit `None` check. The survival block was recomputed from the saved per-condition results
(`j2_analyze.py resurvive`), with no data re-read. The criteria are unchanged. The original output is kept in
`results/J2/J2_X.json` under `survival_original_buggy`.

**Corrected verdict.** EU_I fails `overlap_25`, `overlap_40` and `sensor_noise_0.20`; EC fails `overlap_25` and
`overlap_40`. Neither is promoted. The conclusion is unchanged, only the list of failing conditions.

## D2 (2026-10-18, after J5-X, before any J6 test data): J6 run as an exploratory study despite the J5 gate

**The gate.** The master preregistration says J6 runs only if J5 demonstrates the loop. J5 did not: Q3 and Q4
(change_B) failed, and Q5 was not testable.

**Why J6 runs anyway.**
- The pilot-diagnosed cause of J5's failure is the fixed L0 mapping's compressed range. Beliefs of 0.25–0.5 map to
  stopping distances of 0.45–0.55 m, and L0's "cautious" styles lengthen exposure. That mapping is precisely what J6
  varies.
- The final report must answer whether a learned mapping improves anything.
- The cost is small: about 15 min of one VM.

**Constraints.**
- J6 is labelled **exploratory**. Its result cannot by itself adopt an architecture.
- Any architecture it suggests needs its own preregistered confirmation.
- The J6 addendum fixes its systems, price and metrics before J6-X data.

## D3 (2026-10-18, after J6-T fitting, before any J6-X data): the learned mapping is belief-independent; a proximity value is added

**Finding.** With the preregistered utility (time + price × P(strong evidence)), `utility_intent` chooses the same
intent for every belief and every price checked (5–80): proximity 0 (stop at 0.6 m), tempo 1, no hesitation.
- On NERVA, stopping far is both the fastest option and the one that minimises perturbation, so nothing in that
  utility rewards approaching.
- A decision-theoretic mapping learned from time and perturbation alone therefore **removes belief-dependence
  entirely**.
- The GRU trained to imitate it has constant targets, so its validation accuracy of 1.0 equals the majority baseline
  and is meaningless.

This is reported as a J6 result. The preregistered `LEARNED` system (price 10, no proximity value) still runs in J6-X.

**Addition (exploratory, disclosed).**
- A proximity value: `value` seconds per 0.1 m that the robot stops short of 0.3 m. It represents a task that needs the
  robot to come close, such as handing something over.
- **price 40, value 5** was chosen by inspecting the mapping only, with no outcome data, from four settings tried:
  (10, 5), (20, 5), (40, 5), (40, 10). It was the only one whose choices span the whole proximity range across
  beliefs: 0.3 m for b(pushy) ≤ 0.2, 0.4 m for 0.3–0.5, and 0.6 m for ≥ 0.6.
- Systems `LEARNED_V`, `RESIDUAL_V` and `GRU_V` use it. The GRU is retrained on its targets.
- Analyses report both utilities. Each learned system is optimal for its own utility by construction, so a utility
  gain over L0 is expected, not a discovery. What J6 tests is what the gain costs: safety, evidence dependence and
  revision.

**Fix found during this step.** `j6_fit.py` stepped the GRU once on a zero vector before the first prediction; at run
time the first choice is made from h = 0. Training now matches run time, and both GRUs were retrained before J6-X.
