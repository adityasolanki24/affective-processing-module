# G2b addendum: revision under selective feedback (fixed before any G2b data, 2026-10-12)

**Why G2b exists.** On validation, no G2 memory met the adaptation criterion: every one failed S1 (a risky person
becomes safe), because the robot keeps avoiding people its memory marks as risky, so no exposed encounters (evidence)
arrive, and every G2 revision mechanism (including M1's discount) acts only when evidence arrives. This is a
follow-up experiment with its own hypothesis and fresh lifetimes; G2's own preregistered test (all memories,
secondary) runs unchanged.

**Hypothesis.** Prompt revision under selective feedback requires either evidence-free decay toward the prior
(time-based forgetting) or safe information gathering (probing), or both.

## Candidates (same regulator and evidence model as G2)

| id | memory | probing |
|---|---|---|
| M0_all | F's cumulative memory, all encounters as evidence (reference) | none |
| M1 | G2-tuned trial-discount memory (γ 0.598) | none |
| M1t | time-decay memory (`TimeDecay`: n, k × e^(−Δt/τ_s) continuously), τ_s tuned | none |
| M1t_probeU | M1t | uncertainty-guided: budget 2/episode, memory confidence < 0.5, nobody else within 2 m |
| M1_probeU | M1 | uncertainty-guided, as above |
| M1t_probeR | M1t | random, p = 0.1 per eligible decision, same budget (control) |

## Tuning and selection (`scripts/g2b_tune.py`, partition V only)

τ_s: K = 16 log-uniform values in [60, 3000] s (sample seed 20261013) on G2-V 0–5 of S1, S2, S3, S5, objective =
per-encounter Brier (as G2). Validation: all six systems on G2-V 200–211 of S1–S9. **M\*** = the candidate among
{M1t, M1t_probeU, M1_probeU} meeting G2's criteria (i)–(iii) on validation (point estimates); ties → lower Brier,
then fewer probes. If none passes, the X run is secondary only.

## Test (`cloud/g2b_test.sh`)

Partition X, **lifetime indices 1000–1059** of each of S1–S9 (60 per family; disjoint from G2's X indices 0–59),
all six systems. Primary family as G2 (M\* vs M0_all; Holm over (i) adaptation, (ii) collision non-inferiority,
(iii) false blame). Secondary: every candidate vs M0_all and vs M1; probing vs random probing (probes used,
collisions, careless collisions, adaptation); Oracle not repeated.
