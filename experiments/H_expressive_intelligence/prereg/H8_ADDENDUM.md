# H8 addendum: integrated evaluation and selection (fixed before H8 test data, 2026-10-13)

## Systems

Only mechanisms that passed their gates are combined. The interface is the H0b/H1 one in every case.

| name | memory | global state | mapping |
|---|---|---|---|
| **B0 (baseline)** | A0 explicit memory: identity Beta + kind pool (`M_id`, H3-tuned α 0.02) | its EMA | fixed lookup L0 |
| **I (integrated)** | identity × zone memory with change-point beliefs (context keying from H3's selection, change point from M_id_cp's faster revision) | scalar EMA with α 0.07 (H6's S2 rate) | L0 |
| A0 + learned mapping | M_id readout | — | L1 MLP (H5's selected configuration, retrained deterministically, 5 seeds), no separate developmental latent |

**Ablations of I** (each removes one element):
- I − context (identity keying);
- I − change point (cumulative Beta);
- I − global (α → 0: g fixed at the base rate);
- memory only (no global term);
- global only (S2).

**Recurrent reference:** S6 fast/slow (H3 best E1), frozen.

## Data (fresh)

- New X streams: all 7 histories × 30 lifetimes, lifetime indices 2000 + i (never used), same generator.
- **Dispositions:** joined pairs of these streams, as in H6.
- **Physical** (NERVA, H34 §6 protocol, sim seeds `900000 + index`):
  - for B0, I and A0 + L1;
  - (a) the most-important-test pairs (risky_B vs safe, B1 plain, 1.0 m);
  - (b) generalisation to an unfamiliar situation: novel identity B3 in the marked zone, 1.3 m at bearing −0.7 rad;
  - (c) the same pairs on the shift geometry (1.8 m, bearing 2.4 rad).

## Measures

- E1, E2 (+ pooled-shuffle control), E4 revision, E6 kind generalisation, E7 context, and the H6 disposition
  measures, as preregistered in H34, H4b and H6.
- Physical history classification, falls, trips, arrival.
- **Compute:** µs per update and per readout on this laptop, and parameter or state counts.

## Decision rules (fixed)

1. **I replaces B0 only if both:**
   - I is non-inferior to B0 on E1 and E2 (one-sided 95% bootstrap lower bound of the difference > −0.05);
   - I is better on at least one of:
     - revision (median encounters lower and fraction revised higher);
     - context (E7 difference CI > 0);
     - disposition (meets the H6 criterion while B0 does not).

   Otherwise B0 is retained.
2. **"A0 + learned mapping" replaces the fixed lookup only if** it is better than B0 on calibration error CE (CI of
   the difference < 0) and not worse on E1, E2 and E4 (H5 rule, on these fresh data).
3. **Ablations:** an element is "necessary" if removing it makes I worse on its target measure (CI excludes 0):
   - context → E7;
   - change point → revision;
   - global → disposition;
   - memory → E1 and E2.
4. **Physical:** the selected system must keep 0 falls and ≥ 95% arrival in (a)–(c), and history classification
   ≥ 0.80 in (a).

**The final decision (A–E) follows MASTER §6, with these results.** Human interpretability remains untested.
