# G4 + G5 — Architecture tournament and training methods (2026-10-12)

Preregistered in `prereg/G45_ADDENDUM.md` (with the A5 amendment). 45 networks (A2/A3 × PPO, BC, BC→PPO, DAgger and
A4 × PPO; 5 seeds each; ≈ 15k parameters each), all interactions counted; test on G4-X (400 lifetimes × 5 episodes;
commit `39fedb0`, lock passed, all 45 checkpoint hashes verified; `results/G45/g45_X.json`). Shortlist by validation
only: A2 BC→PPO, A3 BC→PPO, A4 PPO.

## G4 primary — does any neural architecture beat A0? **No.**

| candidate vs A0 | completion | collisions / min | collisions / 100 m | idle | decision |
|---|---|---|---|---|---|
| A4 learned residual on A0 (PPO) | **+7.3 pp [+5.8, +8.7]** | +4.7% [−3.1, +13.1] | −2.7% [−9.3, +4.4] | +2.5 pp | not beat (B fails: collision CI upper > +5%) |
| A2 ordinary GRU (BC→PPO) | +6.8 pp [+5.4, +8.0] | +6.0% [−2.1, +14.5] | −0.5% | +3.4 pp | not beat |
| A3 GRU + memory inputs (BC→PPO) | +6.6 pp [+5.3, +7.7] | +8.7% [+0.7, +17.3] | +2.5% | +3.2 pp | not beat |

Holm p for criterion B (completion) < 0.001 for all three, but criterion B also requires the collision-rate CI upper
bound ≤ +5%, which none meets. A1 (APM v0.1) is again worse than A0 (+25.1% collisions).

**Reading.** The learned policies trade differently, not uniformly better: they complete ≈ 7 pp more goals by moving
more (collisions per distance are on par with A0), so collisions per minute rise by 5–9% with CIs reaching +13–17%.
Under the preregistered safety-first rule this is not a win. The specialised residual regulator (A4) and the ordinary
GRU are statistically indistinguishable (A4 vs A2: completion +0.6 pp [−1.1, +2.2], collisions −1.2% [−9.2, +7.2]),
so nothing supports a specialised regulation structure over a standard recurrent network. Explicit memory inputs add
nothing to the GRU either (A3 vs A2: collisions +2.6% [−4.5, +10.5], completion −0.2 pp [−1.7, +1.3]).

## G5 primary — is the training method the bottleneck?

| test | result | decision |
|---|---|---|
| A3 best method (BC→PPO) vs A3 PPO from scratch | collisions **−17.4% [−24.9, −9.5]**, completion +0.2 pp | **beats** (criterion A, p_Holm < 0.001) |
| best ordinary GRU (A3 BC→PPO) non-inferior to A0 | collisions +8.7% (one-sided upper +15.8%) | **not non-inferior** |

| method (A2 / A3, vs A0) | completion | collisions | interactions |
|---|---|---|---|
| BC | +0.2 / +0.4 pp | +0.8% / +0.4% | 0.3M |
| DAgger | +0.2 / +0.3 pp | +3.4% / +3.1% | 0.3M |
| PPO from scratch | +6.6 / +6.4 pp | +27.0% / +31.7% | 2.4M |
| BC → PPO | +6.8 / +6.6 pp | +6.0% / +8.7% | 2.4M |

- Imitation (BC, DAgger) reproduces A0 within ±3% in 0.3M interactions (8× less data) but cannot exceed its teacher.
- RL from scratch gains completion at a large safety cost; initialising RL from imitation keeps the completion gain
  and removes most of that cost (−17% collisions vs PPO). Learning-curve area per decision: A4 30.1, A3 BC→PPO 28.8,
  A2 BC→PPO 28.4, PPO 26.9–27.4, DAgger ≈ 24.4, BC ≈ 14.8 (validation J).
- Training cost per model: PPO 37–45 min (2.4M decisions), BC 16 min, DAgger 23 min (on cloud vCPUs).

## Action use (collapse diagnosis)

| system | GO | SLOW | YIELD | DETOUR | entropy (bits) |
|---|---|---|---|---|---|
| A0 | 0.59 | 0.19 | 0.21 | 0.00 | 1.41 |
| BC / DAgger | 0.59–0.60 | 0.18 | 0.22 | 0.00 | 1.37 |
| PPO (A2, A3) | 0.63 | 0.00 | 0.37 | 0.00 | 0.95 |
| BC → PPO (A2, A3) | 0.63–0.64 | 0.01 | 0.35 | 0.00 | 1.01–1.03 |
| A4 | 0.62 | 0.06 | 0.32 | 0.00 | 1.21 |
| F's E4 B4 (for comparison) | 0.61 | 0.00 | 0.39 | 0.00 | 0.97 |

RL drives SLOW out even when initialised from a teacher that uses it 19% of the time, and the resulting GO/YIELD
policies achieve the highest J. This is the optimum of the objective in this testbed — pedestrians only cut across a
*moving* robot and walk around a stopped one, so stopping dominates slowing for safety and going dominates for
progress — not a learning pathology. It also means J rewards trading a few collisions for many goals; the
preregistered safety criterion does not.

## Decisions

- **G4:** no neural candidate beats A0 under the preregistered rule → A0's family is the G4 choice.
- **G5:** the learning algorithm matters (imitation-initialised RL ≫ RL from scratch on safety), but even the best-
  trained ordinary GRU is not non-inferior to A0 on collisions per minute; a new architecture is not what is missing.
- A learned efficiency layer is promising (+7 pp completion at per-distance safety parity) but its per-time safety is
  unresolved; it would need a safety-constrained objective (e.g. a Lagrangian collision budget) before it could be
  adopted. Not adopted here.
