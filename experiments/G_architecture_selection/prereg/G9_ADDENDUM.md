# G9 addendum: validity re-test on avoidable collisions (fixed before any G9 data, 2026-10-12)

**Why.** Testbed defect D8 (`DEVIATIONS.md`, 2026-10-12): many reported collisions are unavoidable spawn overlaps at
the first step of an episode, identical for every policy. They dilute every collision-rate comparison and make
non-inferiority conclusions anti-conservative. G9 repeats the preregistered primary families with a corrected metric.
The original confirmatory results are kept and reported unchanged.

**Metric.** Avoidable collisions = all reported collisions minus those at t ≤ 0.051 s of an episode
(`apm_select.instrument.collision_split`); rate = avoidable collisions per minute. Every other metric, margin, rule and
multiplicity correction is exactly as in the original addenda (superiority rule, non-inferiority +10% / −2 pp, Holm
within each family). Total-collision results on the same lifetimes are reported alongside.

**Same systems, fresh lifetimes, no reselection.** All selections, tuned parameters and trained models are the frozen
ones (`configs/G1_selected.json`, `G3_selected.json`, `G8_selected.json`, `G8_selected_min.json` (from the G8 stage-2
VM; committed with this file), `G6_systems.json`, the 45 G4/G5 checkpoints and the 30 G3 checkpoints by hash).
Lifetimes: indices **10000 onward** of each experiment's test partition (never used before).

| re-test | systems | lifetimes | primary family (unchanged) |
|---|---|---|---|
| G1 | B0, R0–R8 (no negative controls) | G1-X 10000–10399 × 10 episodes | P1 R6 > R1; P2 R2 ≥ R6 (NI); P3 R2 > R1 |
| G3 | none, ema, lds, gru, fastslow, pad, A0_R2 (no controls) | G3-X 10000–10399 × 10 | Q1 ema > none; Q2 fastslow > ema |
| G4/G5 | A0, A1; A2_bcppo, A3_bcppo, A4_ppo, A3_ppo (5 seeds each) | G4-X 10000–10399 × 5 | G4: each shortlisted cell > A0; G5: A3 BC→PPO > A3 PPO, A3 BC→PPO ≥ A0 (NI) |
| G8 stage 1 | A0 variants + B0 | G8-X 10000–10399 × 10 | removals NI to A0; A0_mem > A0 |
| G8 stage 2 | A0, A0_min | G8-S 10000–10399 × 10 | A0_min NI to A0 |
| G6 | `configs/G6_systems.json` (A0, R1, A0_mem, A1, B0, 3 neural cells × 5 seeds) | G6-X 10000–10149 per condition × 5 | eligibility (NI to A0 in ≥ 80% of 13 conditions, never > 2× A0) |

N = 400 is the upper bound of the master preregistration's ranges (avoidable collisions are rarer, so precision is
maximised within the bounds); G6 keeps 150 per condition. Analysis: `scripts/g9_analyze.py` (avoidable) and the same
with `--metric total`.

**Decision use.** Where G9 and the original test disagree, the G9 (avoidable-collision) result governs the
architecture decision, because it measures what the policy can influence; both are reported.
