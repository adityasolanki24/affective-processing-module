# G2 — Which memory can learn and revise experience? (2026-10-12)

Preregistered in `prereg/G2_ADDENDUM.md`. Regulator fixed to G1's selection (R2). Tuning on G2-V (equal K = 16 per
parametric memory), validation selection on G2-V 100–111 of all nine scenario families, test on G2-X (60 lifetimes ×
30 episodes per family, 9 systems, commit `0beddfa`, lock passed; `results/G2/g2_X.json`).

## Validation selection: no memory qualifies

Every candidate — cumulative (M0), trial-discount forgetting (M1, γ = 0.60), sliding window (M2, W ≈ 6),
hierarchical Bayes (M3), change-point detection (M4, h = 0.26), two-state hidden-Markov filter (M5) and the learned
recurrent memory (M6) — failed criterion (i): in S1 (a risky person becomes safe) the Kaplan–Meier median adaptation
was not reached within 15 episodes. As preregistered, **no M\* was selected**, and the test ran all memories as
secondary comparisons.

## Test (secondary, exploratory labels)

| memory | Brier (observable encounters) | ECE | S1 RMST (episodes, of 15) | S2 RMST | false blame (S5) | collisions vs M0_all | completion |
|---|---|---|---|---|---|---|---|
| M0_all (F) | 0.0472 | 0.035 | 10.7 | 10.9 | +0.007 [+0.003, +0.011] | — | — |
| M0_exp | 0.0479 | 0.041 | 13.6 | 10.7 | +0.005 | −9.5% [−12.0, −6.9] | −0.4 pp |
| M1 | 0.0464 | 0.036 | 11.3 | 11.7 | +0.008 | −12.9% [−15.3, −10.4] | −0.6 pp |
| M2 | 0.0484 | 0.048 | 12.4 | 10.1 | +0.003 | −9.9% [−12.4, −7.3] | −0.4 pp |
| M3 | 0.0452 | 0.030 | 13.5 | 11.7 | −0.001 | **−13.9% [−16.4, −11.4]** | −0.9 pp |
| M4 | 0.0465 | 0.037 | 10.9 | 11.9 | +0.007 | −12.9% [−15.2, −10.6] | −0.5 pp |
| M5 | 0.0463 | 0.017 | 12.5 | 11.1 | −0.004 | −11.1% [−13.4, −8.6] | −0.4 pp |
| M6 (learned) | **0.0438** | **0.016** | 11.9 | 9.8 (median 10) | +0.001 | −10.8% [−13.2, −8.4] | −0.3 pp |

Adaptation medians: S1 not reached by any memory; S2 reached only by M6 (10 episodes). Around 69% of person encounters
carry no evidence (avoided or never close while moving) for every memory; 46–50% of encounters with *careful*
people involve avoidance.

## Findings

1. **The adaptation failure is a feedback problem, not a memory-model problem.** Forgetting, windows, change-point
   detection and hidden-state filtering all revise only when evidence arrives; under avoidance it does not arrive.
   G2b (`prereg/G2b_ADDENDUM.md`) tests the two remedies this implies: evidence-free decay toward the prior and safe,
   budgeted probing.
2. **Exposure-gated evidence is the useful memory change.** Not counting avoided/unexposed encounters as "safe"
   evidence (every M\*_exp model) reduces collisions by ≈ 10–14% at a ≤ 0.9 pp completion cost; differences between
   the statistical models are small (M3 / M1 / M4 best).
3. **Neural memory (M6) is the best-calibrated predictor** (Brier −7% vs M0_all, ECE 0.016) but does not adapt
   better and is not safer than the statistical memories; a 9-float GRU state buys calibration only.
4. **No false attribution:** all memories keep the S5 blame within ±0.03 (exposure gating removes F's small positive
   bias, +0.007).
5. **Limitation:** the "regret vs Oracle" metric uses programme F's converged per-disposition rates, which depend
   on F's policy; under the G1 rule they are miscalibrated (the Oracle's Brier is worse than the memories'), so regret
   is not interpreted. F's median-based adaptation definition can also tie when most identities share a rate (the
   Oracle's S1 median is "not reached" for that reason).

## G2b — revision under selective feedback (`prereg/G2b_ADDENDUM.md`)

Validation again selected no M\* (no candidate met criterion (i) with both medians ≤ 6), so the test (G2-X indices
1000–1059 per family, 6 systems; commit `5dd220e`, lock passed; `results/G2/g2b_X.json`) is secondary.

| system | S1 median [95% CI] | S1 RMST | S2 median | S2 RMST | collisions vs M0_all | completion | probes |
|---|---|---|---|---|---|---|---|
| M0_all | not reached | 10.7 | not reached | 12.5 | — | — | 0 |
| M1 (trial discount) | not reached | 10.8 | not reached | 11.7 | −9.5% | −0.7 pp | 0 |
| M1t (time decay, τ 356 s) | 9 [4, —] | 8.6 | 11 | 9.4 | −11.6% [−14.1, −9.1] | −0.6 pp | 0 |
| **M1t + uncertainty probing** | **6 [2, 13]** | **7.1** | **8** | **8.3** | −10.1% [−12.5, −7.7] | −0.7 pp | 9,242 |
| M1 + uncertainty probing | not reached | 10.4 | not reached | 11.6 | −10.0% | −0.7 pp | 6,047 |
| M1t + random probing (same budget) | 12 | 8.8 | 13 | 9.4 | −11.0% | −0.7 pp | 7,595 |

**Findings.** Revision needs *both* ingredients: evidence-free decay toward the prior (without it, probing alone does
nothing: M1 + probing = M1) and *targeted* information gathering (random probing at a similar probe count is
clearly worse). The combination cuts the time to revise a risky→safe person by a third (RMST 7.1 vs 10.7 episodes) without a
safety cost (collisions still −10% vs F's memory; probes are budgeted SLOW approaches under the unchanged safety
supervisor). It still falls short of the preregistered ≤ 6-episode criterion in S2 (safe→risky: median 8), so G2's
answer remains "no memory met the adaptation criterion"; the best-supported memory design is **exposure-gated
evidence + time decay + uncertainty-guided probing**.
