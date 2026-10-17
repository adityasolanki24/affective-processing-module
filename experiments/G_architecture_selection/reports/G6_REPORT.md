# G6 — Out-of-distribution robustness (2026-10-12)

> **G9 correction (governs):** with avoidable collisions the eligibility test is underpowered (A0: 5 avoidable collisions in 150 nominal lifetimes) and no system is eligible; A0_mem is better than A0 in 11/13 conditions and never worse (significantly under fast pedestrians, slow actuation and identity uncertainty). See `G9_REPORT.md`.

Preregistered in `prereg/G6_ADDENDUM.md`; run once (commit `39fedb0`, lock passed; 14 conditions × 150 lifetimes ×
5 episodes × 20 systems = 42,000 lifetimes; `results/G6/g6_X.json`). Systems: A0, R1, A0_mem (A0 + G2b memory design,
A0's penalty unchanged), A1 (APM v0.1), B0, and the validation-selected neural cells A2 BC→PPO, A3 BC→PPO, A4 PPO
(5 seeds each). The total-collision metric of this run is diluted by testbed defect D8; the avoidable-collision
re-test is in G9 (`G9_REPORT.md`).

## Eligibility (non-inferior to A0 in ≥ 80% of the 13 held-out conditions and never > 2× A0's collision rate)

| system | non-inferior share | worst collision ratio | eligible |
|---|---|---|---|
| **A0_mem** | **0.85** | **1.04** | **yes** |
| A4 (residual on A0) | 0.38 | 1.44 | no |
| A2 GRU (BC→PPO) | 0.23 | 1.45 | no |
| A3 GRU + memory (BC→PPO) | 0.23 | 1.52 | no |
| R1 (static penalty) | 0.23 | 1.52 | no |
| A1 (APM v0.1) | 0.08 | 2.15 | no |
| B0 | 0.00 | 41.2 | no |

## Per-condition highlights (collisions relative to A0, 95% CI; completion difference)

| condition | A0_mem | A4 | R1 | A1 |
|---|---|---|---|---|
| nominal | −9.5% [−16.2, −3.3], −0.8 pp | +7.8%, +6.6 pp | +16% | +24% |
| fast pedestrians | −13.7% [−22.8, −3.5], −1.5 pp | −19.4%, +4.2 pp | +52% | +115% |
| observation delay 0.25 s | −15.8% [−25.9, −4.8], −0.8 pp | −7.9%, +4.7 pp | +18% | +50% |
| slow actuation | −23.3% [−32.2, −13.7], −0.4 pp | **+43.8%**, +5.8 pp | +30% | +90% |
| estimator ×0.5 | −9.0% [−21.6, +4.5] | +20.0% | +26% | +84% |
| careless-heavy | −3.9% | +13.8% | +12% | +32% |

## Findings

1. **A0 with the revisable memory (exposure-gated evidence, time decay, uncertainty-guided probing) is the only
   eligible candidate and is the most robust system tested**: never worse than A0 in any condition and markedly
   better under faster pedestrians, observation delay and sluggish actuation, at ≤ 1.5 pp completion cost.
2. **The neural candidates are not robust**: they keep their ≈ 6 pp completion advantage but their collisions rise
   sharply under unfamiliar dynamics (slow actuation +44% to +52%), and they fail the non-inferiority rule in most
   held-out conditions. A candidate that wins in-distribution but degrades under shift is not selected.
3. APM v0.1 (A1) is the least robust non-trivial system (up to 2.15× A0).
4. The static penalty (R1) is less robust than A0 (R2's slightly higher, calibrated penalty), confirming that the
   *calibration* of risk weight matters more than its dynamics.
