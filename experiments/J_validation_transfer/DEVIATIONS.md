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
