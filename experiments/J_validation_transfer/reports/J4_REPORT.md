# J4 report: revisable experience on NERVA (2026-10-18)

**Governed by:** `prereg/J4_ADDENDUM.md`, committed at f0ff2e1 before any J4-X lifetime.

**Data:** J4-X, 1,400 paired lifetimes (7 histories × 200) × 48 encounters. Every encounter's evidence is a test-row
draw (rows 300–999 per type) from 4,000 physically simulated NERVA standing encounters at 0.4 m. The robot measures
it with its own IMU (self-calibrated detector, acc8 code); no simulator velocity is read.

**Results:** `results/J4/J4_X.json`; run metadata `results/J4/J4_X_meta.json`.

## Verdict: **J4 fails (C2 and C3).** Robot-observable memory is experience-dependent and persistent, but too slow to revise.

| criterion | type_kind (primary) | H_A0 (reference) | pass? |
|---|---|---|---|
| C1 experience dependence: history AUROC ≥ 0.90 | **1.00** | 0.94 | ✅ |
| C2 caution effect non-inferior to H_A0 (lower bound ≥ −0.05) | +0.285 vs +0.387; diff **−0.102 [−0.127, −0.076]** | (reference) | ❌ |
| C3 revision ≤ 2 blocks, both directions | pushy → calm: **not reached** (12 of 163 revised within 4 blocks); calm → pushy: median **4** blocks [3, 4] | neither direction reached | ❌ |
| C4 restart persistence: exact | max \|diff\| **0.0** | n/a | ✅ |
| C5 false attribution ≤ 0.5 (expected to fail) | 0.68 | 0.48 | ❌ (expected) |

## Recurrent baseline (triggered: H_A0 and type_kind both fail C3)

`GruMemory` (hidden 8, self-supervised next-detection prediction, trained on V):

| | gru | type_kind |
|---|---|---|
| history AUROC | 0.86 ❌ | 1.00 |
| caution effect | +0.233; C2 diff −0.154 | +0.285 |
| pushy → calm revision within 4 blocks | 50 / 143 | 12 / 163 |
| calm → pushy median | 3 blocks [2, 4] | 4 blocks [3, 4] |
| generalisation to novel B3 | none (no pooling) | +0.166 |

**The recurrent state does not rescue J4.**
- It forgets faster, so it revises somewhat more often.
- That costs experience dependence: AUROC 0.86, below C1.
- It also has the smallest history effect.

This is the usual plasticity–stability trade-off, not a capability the explicit memory lacks. The explicit memory
reaches the same trade-off with a single interpretable parameter, λ.

## What the results say

1. **Memory learning works on NERVA with robot-observable evidence (C1, C4).**
   - From IMU evidence alone, the type memory separates risky from safe histories perfectly.
   - It does not leak to a safe entity of the other kind (risky_A specificity 0.0).
   - It generalises to an unmet entity of the same kind through kind pooling (B3 +0.166).
   - It survives a save/restore exactly.
2. **The history effect is smaller than H's.**
   - The type memory's readout p = b·r is bounded by the two learned detection rates (0.21 and 0.52). Even a
     certain belief spans only 0.31.
   - H_A0's Beta-on-detections readout uses the full scale.
   - This is the same compression that made L0 under-express in J5. It is a property of how the readout is scaled,
     not of what the memory knows: AUROC is 1.00 vs 0.94.
3. **Revision is slow and asymmetric for every memory.**
   - Calm → pushy is reached in about 4 blocks (≈ 4 meetings with B1) by the type memories. H_A0 and count rarely
     reach the deep threshold.
   - Pushy → calm is almost never reached.
   - The reason is evidence asymmetry, not forgetting rate alone. A calm entity still trips the robot's detector 21% of
     the time, against 52% for a pushy one, so each calm meeting moves the belief only a little.
   - The λ = 0.005 chosen by predictive likelihood favours stability, and the objective is nearly flat. The robot's
     own predictive objective does not ask for faster forgetting.
4. **False attribution (zone_B) is substantial for the type memories** (0.68 of the true effect), as expected for
   memories keyed by identity rather than context. H_A0's 0.48 is lower, plausibly because its kind and global pools absorb part
   of the blame.

## Uncertain

- Whether a readout rescaled to the learned type range (e.g. p = P(pushy type)) would pass C2. It probably would, but
  that is a post-hoc change and is **not** claimed.
- Revision speed under a different physical evidence channel (e.g. contact sensing) is untested.
- Every encounter is at the neutral 0.4 m; the closed-loop version with expression-dependent distance is J5.

## Is the next phase justified?

J5 and J6 already ran (J5 failed Q3 and Q4, so J6 is exploratory, D2). J4 agrees with J5's failure mode: slow,
asymmetric revision on NERVA's evidence. **Phase 4 should treat "revisable NERVA memory" as not demonstrated.**
