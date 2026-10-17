# E1 addendum: fixed before any E1 partition-X/S data (2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §2 E1 (programme F). Nothing here changes a hypothesis, arm, metric, margin
or criterion. Written in the programme-G session (see `../G_architecture_selection/reports/F_ACCOUNTING.md`) after
E1 stage 1: the datasets P/T/V were regenerated on a cloud VM with F's committed scripts and seeds and are
**bit-identical** to F's committed manifests (wheeled P/T/V and NERVA P/T/V dataset SHA-256 equal; all 760 NERVA
trial files identical), and every arm was trained on T with early stopping on V and predictions on P. Only the
power mode of `e1_analyze.py` (bootstrap SEs and required clusters) was run on P.

## Sizes

| embodiment / target | pilot (P) | required clusters (80% power, margin 5%) | X (confirmatory) | S (shift, secondary) |
|---|---|---|---|---|
| wheeled collision (primary) | 40 lifetimes | 58 | **100 lifetimes** (indices 0–99, 5 episodes each, as T) | 40 lifetimes |
| wheeled stability_loss (secondary) | 40 | 34 | same data | same |
| NERVA stability_loss (primary) | 120 trials | 52 | **160 trials** (F's 60/20/20 split rule with T = 480) | 120 trials |

Rule: X ≥ the pilot requirement; for NERVA the preregistered split proportions, for wheeled the training size.
S uses `e1_data.SHIFT` (wheeled: 4–6 people, faster pedestrians, more low-friction patches) and NERVA's partition-S
push speeds (U(1.0, 1.8) m/s), as already committed in the scripts.

## Procedure (unchanged code, `cloud/f_e1_stage2.sh` in programme G)

Regenerate T and V (checked against the manifests), generate X and S, train every arm (`e1_train.py` as committed:
5 seeds for GRU arms, 1 for LR; NERVA skips the memory arms) on T with early stopping on V and predictions on X and
S, then `e1_analyze.py --part X` per embodiment/target. Baseline selection on V among R/RL/M/RLM (wheeled) or R/RL
(NERVA), as committed. The stage-2 job verifies that this file is committed and no tracked file is modified before
generating X (the programme-G confirmatory lock) and records the provenance.

## Criterion (unchanged)

Per embodiment, primary family U, P, UP vs the selected baseline (Holm, 3 tests): "adds meaningful information" iff
relative Brier ≤ −5% with Holm CI upper < 0 and the arm also beats N and C by ≥ 5%; "no meaningful information" iff
the CI lies within (−5%, +5%); otherwise inconclusive. Decision: if PAD (P) is not meaningful, PAD is no longer
treated as a necessary predictive representation.
