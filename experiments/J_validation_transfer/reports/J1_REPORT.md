# J1 report: programme I's I1 under independent calibration (2026-10-17)

**Governed by:** `MASTER_PREREGISTRATION.md` §2 and `prereg/J1_ADDENDUM.md`.

**Design:**
- Calibration was frozen on J1-V (`configs/J1_calibration.json`) before J1-X existed.
- Test: J1-X, 14,000 new lifetimes; programme I's frozen I1 systems plus the decay-only control R_decay.

**Results:** `results/J1/J1_X.json`.

## Verdict: **I1 survives.** The validity control works.

R_dense meets every criterion with thresholds that never saw the test data:

| criterion | S1 | S2 | S10 | other |
|---|---|---|---|---|
| C1 frozen-calibrated adaptation, median [95% CI] (episodes) | 2 [2, 2] | 3 [3, 3] | 2 [2, 2] | |
| C1b decisive adaptation (deep and stable) | 3 | 4 | 2 | |
| C2 S3 AUROC | | | | 1.00 |
| C3 avoidable collisions vs A0 | | | | **−52.2% [−55.4, −48.8]** |
| C4 completion vs A0 | | | | −0.25 pp [−0.34, −0.17] |
| C5 normalised false blame | | | | 0.008 |

Programme I's original, test-calibrated method gives the same medians on these seeds (2 / 3 / 2). **The calibration
issue did not change the conclusion.**

**Validity.** R_decay stops learning at the switch and can only forget toward the prior. It fails decisive adaptation in
S1, S2 and S10, and also fails the frozen-calibrated form. The measures do not credit forgetting.

## Comparison systems

| system | adaptation S1 / S2 / S10 (frozen) | AUROC | collision log loss, switched identities after the switch (S1 / S2 / S10) |
|---|---|---|---|
| A0 | never / never / never | 0.77 | 0.074 / 0.177 / 0.071 |
| G2b | never / 6 / never | **0.22** (inverted) | 0.085 / 0.126 / 0.083 |
| R_sparse | 1 / never / undefined (no separation) | 0.50 | 0.055 / 0.134 / 0.053 |
| R_static | 13 / never / never | 1.00 | 0.060 / 0.144 / 0.058 |
| **R_dense** | **2 / 3 / 2** | **1.00** | **0.053 / 0.114 / 0.050** |
| R_decay | never / never / never | 1.00 | 0.059 / 0.117 / 0.056 |
| Oracle | 1 / 1 / 1 | 1.00 | 0.063 / 0.325 / 0.060 (out-of-range input scale) |

## What J1 adds

1. **The safety gain is not caused by revision.** R_decay, identical to R_dense until the switch and frozen
   afterwards, cuts avoidable collisions by −55.0% [−58.0, −51.8]: as much as R_dense.
   - The collision benefit comes from holding *discriminating type beliefs*, together with the confidence channel the
     predictor reads (programme I's exploratory ablation).
   - It does not come from revising them. In these worlds most people never change, and switched identities are a
     small share of encounters.

   Revision is real, fast and well calibrated. Collision log loss after the switch is lower for R_dense than R_decay
   in all three families. But its behavioural pay-off is small here.
2. **Decisions barely depend on memory.** The decision read-out is decision A's rule applied to a fixed bank of real
   situations. Its careful-action share moves only 0.45 → 0.49 across the realistic belief range, and responds more to
   confidence than to p.
   - R_dense's read-out revises in 3 / 2 / 2 episodes.
   - Forgetting can also move it in S2 (R_decay "adapts" in 3), because fading confidence makes the rule more careful.
   - The read-out is therefore reported, not used as a criterion.
3. **The observed-decision measure is confounded.** It is the robot's actual careful-action share near the switched
   person. Every system appears to "adapt", A0 and R_decay included, because careful people sidestep and need fewer
   careful actions. It measures the other person's behaviour.
4. **G2b's inversion replicates** (AUROC 0.22).

## Is merging PR #3 justified by J1?

**On J1 alone, yes for programme I's stated claims.** Fast revision, discrimination, safety and efficiency all hold
under independent calibration and new seeds. Two qualifications belong in programme I's decision record:
- the collision benefit is not attributable to revision;
- the decision pathway from memory is weak in decision A's rule.

**J2 decides whether the memory may be promoted as a *default*.**
