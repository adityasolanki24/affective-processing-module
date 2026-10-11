# G1 — Is PAD necessary, or is simple persistent risk regulation sufficient? (2026-10-12)

Confirmatory run once on partition X (N = 280 lifetimes × 10 episodes, 10 systems + 4 negative controls; commit
`f3806f0`, lock passed, clean tree; `results/G1/g1_X.json`), plus the shift conditions on partition S (4 × 100,
selected systems unchanged; `results/G1/g1_S.json`). Equal tuning budgets (48 configurations or 1,440 lifetime
evaluations per family) on G1-V. Selection (validation): R_best = R2 (EMA).

## Primary family (Holm over five p-values)

| test | estimate [95% CI] | decision |
|---|---|---|
| **P1** R6 (fairly tuned B2/PAD) beats R1 (fairly tuned static B1) | collisions **+25.7% [+16.7, +35.4]**; completion −0.1 pp [−0.6, +0.3] | **fails** — PAD is *worse* |
| **P2** R2 non-inferior to R6 (+10%, −2 pp) | collisions −26.0% [−31.6, −20.1] (one-sided upper −21%); completion +0.3 pp (lower −0.07 pp) | **holds** (p_Holm < 0.001) |
| **P3** R2 beats R1 | collisions −7.1% [−12.2, −1.7] (p_Holm 0.022); completion +0.2 pp | **fails** the −10% practical margin |

## What explains programme F's E2 result

| comparison | collisions | completion |
|---|---|---|
| R1 (B1 re-tuned with B2's budget) vs R0 (F's tuned B1) | **−24.6% [−30.2, −18.8]** | +0.4 pp |
| F's E2: B2 vs B1 (for reference) | −25.4% [−32.0, −18.4] | +0.3 pp |

Giving B1 the same tuning budget reproduces the whole of F's B2 advantage. F's E2 gain was due to **unequal
optimisation freedom** (B1's best λc of 1.0 was too low; the equal-budget search chose 2.5), not to PAD.

## Secondary (unadjusted, exploratory)

- **Persistence is not what helps.** R2's own state replayed from another lifetime (shuffled) or shifted by one
  episode changes collisions by +1.6% [−2.8, +6.2] and +2.0% [−2.7, +6.9] — within ±10%, i.e. practically
  equivalent. R2's small advantage over R1 comes from its (near-static) parameters: λc 3.9, k 0.45, τ 0.6 s.
  The other persistent families did not beat R1: hysteresis R3 +8.2% [+2.0, +15.0], Bayesian hazard R4 +1.1%
  [−4.8, +7.4], learned scalar R5 +14.1% [+7.6, +21.3].
- **Within the B2 family the live affect trace matters** (R6 shuffled +14.5%, shifted +16.0% vs R6), but the family
  as a whole is worse than a static penalty. Replacing PAD by the instantaneous appraisal threat (R7: −25.7% vs
  R6) or by a plain low-pass filter of it (R8: −23.7% vs R6) removes the harm; R7/R8 vs R1: −6.7% / −4.1%
  (R7 loses 0.4 pp completion).
- **Exposure:** every effect holds per 100 m travelled and per person encounter (differences within 1–2 pp of the
  per-minute effects); no system bought safety by moving less (16.3–16.8 m/min for all regulated systems).
- **Behaviour:** R6 uses DETOUR (3.1%) and YIELD (23.2%) more and switches/oscillates most (0.42 / 0.16 per
  decision); hesitation is low for all (≤ 0.9% of decisions).
- **Compute:** B2-family systems need ≈ 9.0 ms per decision (APM core at 20 Hz) vs ≈ 0.6 ms for B1 and the simple
  regulators (15× more).
- **Shift conditions (S, no retuning):** R6 is worse than R1 under estimator miscalibration ×3 (+20%), dense
  (+12%) and sparse (+25%) crowds; R2/R7/R8 stay 7–15% below R1; completion differences ≤ 1.5 pp. No condition
  reverses the nominal ordering.

## Decision

1. **PAD is not necessary** for this benefit and, under fair tuning, the APM v0.1 PAD modulation is **harmful**
   (+26% collisions) and 15× more expensive. Removing PAD from B2 making B2 worse (F) is explained by B2's tuned λc
   being calibrated *with* PAD's average up-weighting; it never showed PAD to be necessary.
2. **Persistent risk regulation is not supported either:** no persistent family beat the equally tuned static
   penalty by the practical margin, and R2's state carries no detectable information (negative controls).
3. Carried forward: **a calibrated static collision penalty** (R1 family; R2's parameters as the selected
   configuration, which behaves as a static penalty). Appraisal-threat modulation (R7/R8) is a possible small
   refinement worth at most ≈ 5–7% and is re-examined in G3 (persistent state) and G8 (ablation), not adopted now.
