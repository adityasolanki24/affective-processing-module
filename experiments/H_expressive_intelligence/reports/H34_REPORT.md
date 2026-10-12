# H3 / H4 / H4b / H6 report: internal states, experience-shaped expression, dispositions (2026-10-13)

**Setup:**
- Governed by `prereg/H34_ADDENDUM.md`, `prereg/H4b_ADDENDUM.md` and `prereg/H6_ADDENDUM.md`.
- Experience: **26,880 + 2,880 physically measured encounters**. Each was NERVA standing and pushed (or not) at
  0.15 m/s, observed through measured base speed.
- Partitions: T 280, V 70 and X 210 lifetimes of 48 encounters, plus 60 fresh X lifetimes for H4b.
- Results in `results/H34/` (`X/h34_results.json`, `X4b/h4b_results.json`, `X/h6_results.json`).
- **Expressed caution κ_e** is measured from the intent each robot actually expressed: 0 = relaxed, 1 = cautious.

## Verdicts

| question | answer |
|---|---|
| H4 gate (preregistered) | **fails for every system** on criterion 3. The control was mis-specified (see below). The failure stays on record. |
| H4b gate (corrected control, fresh data) | **passes** for A0 memory + fixed lookup (M_id), GRU (S5), fast/slow GRU (S6) and GRU + memory (S7) |
| H3 selection (smallest non-inferior on E1) | **M_ctx**: explicit identity × zone memory |
| Does any learned or specialised state beat A0 + memory + fixed lookup? | **No** (H34 §8 rule) |
| Is PAD useful for expression? | Not better than a plain scalar EMA on any measure; worst evidence consistency among experience-dependent states |
| Revisable dispositions (H6) | Emerge from *global* states: scalar EMA, small state, PAD, GRU, fast/slow. They are weak in A0 memory (untuned global term) and absent in S7 |

## H3/H4 on X (210 lifetimes; intent level)

| system | E1 consistency | E2 history effect [95% CI] | revision median (fraction revised) | E6 kind generalisation | E7 context | E8 unknown kind (disposition) |
|---|---|---|---|---|---|---|
| S0 none / S1 fixed | 0 / 0 | 0 / 0 | — | 0 | 0 | 0 |
| S2 scalar EMA | 0.43 | +0.31 [0.27, 0.36] | 8 (1.00) | +0.31 | 0 | +0.35 |
| S3 small state | 0.43 | +0.34 [0.29, 0.39] | 12 (0.97) | +0.34 | 0 | +0.38 |
| S4 PAD | 0.41 | +0.23 [0.20, 0.26] | 16 (0.90) | +0.23 | 0 | +0.25 |
| **M_id (A0 memory)** | 0.68 | **+0.82 [0.76, 0.88]** | 24 (0.57) | **+0.75** | 0 | +0.05 |
| M_ctx | 0.72 | +0.82 | > 24 (0.43) | +0.69 | **+0.72** | +0.05 |
| M_id_cp (change point) | 0.71 | +0.82 | 16 (0.73) | +0.75 | 0 | +0.05 |
| S5 GRU | 0.69 | +0.38 [0.32, 0.44] | 8 (0.93) | +0.37 | +0.12 | +0.22 |
| S6 fast/slow | **0.72** | +0.33 [0.28, 0.39] | 8 (0.97) | +0.34 | +0.16 | +0.17 |
| S7 GRU + memory | 0.69 | +0.64 [0.59, 0.67] | 16 (0.77) | +0.62 | +0.07 | +0.03 |
| oracle lookup (told the world) | 1.00 | +0.80 | 4 (1.00) | +0.80 | +0.80 | +0.30 |

Revision is measured in encounters after the world changes.

**Key findings:**
- **Explicit memory produces the largest, most specific history effects.** Kind generalisation comes through its
  pool prior. Context comes only when memory is keyed by context (M_ctx). Memory revises slowly: Beta counts
  accumulate. The change-point variant revises faster.
- **Learned recurrent states revise quickly but express much smaller history effects.** Their world-prediction
  training averages over histories, giving priors near the mean.
  - Fast/slow has the best evidence consistency, but its margin over memory is within the non-inferiority margin.
  - S7 learned a training-set prior: kind B was risky in 4 of 7 training histories. It stays B-cautious even when
    its own outcomes are shuffled (specificity +0.10 [0.05, 0.15]). That is not lifetime experience.
- **Global states (scalar, small, PAD) cannot express entity-specific expectations.** Their B1 effect is entirely
  general mood (specificity 0).
- **Non-inferiority on E1.** Lower bounds vs the best (S6) are: M_ctx −0.035, M_id_cp −0.033, S7 −0.049,
  S5 −0.052, M_id −0.066. So M_ctx is the smallest adequate representation.
- **"Beats the baseline" rule.** No system beats M_id. Every candidate is worse on the history effect (E2) or kind
  generalisation (E6), and M_ctx is also slower to revise.

## Why the preregistered H4 gate failed, and what H4b fixed

**The flaw.** The control "shuffled outcomes" permuted outcomes *within* each lifetime.
- Total experienced harm is preserved: 0.32 under risky_B vs 0.05 under safe.
- So the between-history effect persists for every system that learns at all. A0 memory goes from +0.82 to
  +0.30.
- The control could not null E2 even under the hypothesis it was meant to test.

**Exploratory check (X).** The entity-specific contrast κ(B1) − κ(A1) within risky_B *does* vanish under that
shuffle:
- M_id: +0.78 → +0.02;
- GRU: +0.41 → −0.01.

**H4b's control.** It pools the two paired lifetimes' outcomes and reassigns them at random, so both robots have the
same expected experience. It was preregistered before 60 fresh lifetimes were generated.

### H4b results (fresh X; physical pairs on NERVA, same seed and situation, approach B1)

| system | E2 [CI] | pooled-shuffle E2 [CI] | physical history classification | falls / 60 | arrival | supports claim C |
|---|---|---|---|---|---|---|
| **M_id** | +0.83 [0.79, 0.87] | +0.04 [−0.02, 0.11] | **1.00** | 0 | 1.00 | **yes** |
| S5 GRU | +0.36 [0.30, 0.42] | −0.02 [−0.09, 0.05] | 0.92 | 0 | 1.00 | yes |
| S6 fast/slow | +0.29 [0.23, 0.35] | −0.02 [−0.10, 0.06] | 0.85 | 0 | 1.00 | yes |
| S7 GRU + memory | +0.65 [0.63, 0.68] | +0.02 [−0.02, 0.05] | 1.00 | 0 | 1.00 | yes |
| S4 PAD | +0.24 [0.22, 0.27] | +0.01 | 0.98 | 0 | 1.00 | no (E1 0.41 < 0.50) |
| S0 / S1 | 0 | 0 | 0.50 | 0 | 1.00 | no |

**How the experienced robot moves differently** (M_id, risky_B minus safe, the same situation):
- it stops 0.20 m farther from B1;
- its face is 0.21 rad lower at rest;
- it scans more (+0.05 rad face-yaw SD);
- its path is slightly less direct;
- it progresses 0.008 m/s slower.

These are measured motion differences. Whether people read them as caution is untested (H2).

## H6: dispositions (96-encounter lifetimes from joined X streams; D1 = a kind never met)

| system | world effect at k = 96 [CI] | ICC | revision median (fraction) | revisable disposition? |
|---|---|---|---|---|
| S2 scalar EMA (α 0.07) | +0.36 [0.32, 0.41] | 0.88 | 8 (1.00) | **yes** |
| S3 small state | +0.40 [0.35, 0.44] | 0.89 | 12 (1.00) | yes |
| S4 PAD | +0.29 [0.27, 0.32] | 0.95 | 20 (1.00) | yes |
| S5 GRU | +0.20 [0.17, 0.23] | 0.82 | 20 (0.97) | yes |
| S6 fast/slow | +0.12 [0.09, 0.15] | 0.71 | 8 (0.90) | yes |
| M_id / M_ctx / M_id_cp | +0.07 [0.07, 0.08] | 0.96 | 20 (1.00) | no (effect < 0.10) |
| S7 GRU + memory | +0.03 | 0.46 | 4 (1.00) | no |
| S0 / S1 | 0 | undefined | — | no |

**Reading:**
- A general, repeatable, revisable tendency toward the unfamiliar appears whenever there is a global state
  integrating recent outcomes. The simplest one, a scalar EMA, is as good as any.
- PAD is not better than the scalar: its effect is smaller, revision slower, ICC similar.
- A0's memory has a global EMA term, but H3 tuning could not tune its rate. The rate does not affect the
  world-prediction readout, and the preregistered tie rule chose the slowest (α 0.02). So its disposition is weak.
  This is a tuning-procedure gap, not evidence against global states.

## Limits

- **One outcome dimension:** being pushed while standing. One kind of entity behaviour. Simulation only.
- **Experience is measured with neutral motion.** Expression does not change what the robot experiences.
- **"Coherent" means evidence-consistent and in the designed direction.** Human interpretation is untested.
- **Learned models were trained on a world distribution with a B-risky bias.** S7 learned that prior. A balanced
  training design would remove it.
