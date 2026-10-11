# G4 + G5 addendum: architecture tournament and training methods (fixed before any G4 data, 2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G4 and G5, run as one factorial so that every architecture is compared
under the same training methods (G5's fairness requirement). Nothing here relaxes a criterion.

## The bar: A0

A0 = programme F's frozen predictor + F's cumulative memory + G1's selected penalty (R2: λc 3.95, λs 0.99, EMA
τ 0.62 s, k 0.45), i.e. the strongest non-neural architecture validated so far. G2 selected no memory under its
preregistered rule, so A0 keeps F's memory; G2/G2b memory changes are assembled and ablated in G8, not here.
A1 = G1's fairly tuned APM v0.1 (R6) as a labelled reference.

## Architectures (`src/apm_select/learning.py`; inputs = F's 41 features ⊕ predictor outputs for every action)

| id | network | parameters |
|---|---|---|
| A2 | ordinary GRU actor-critic (F's B4 layout), explicit-memory inputs zeroed | ≈ 15.2k |
| A3 | the same with explicit-memory inputs (F's B4) | ≈ 15.2k |
| A4 | learned bounded residual on A0's utility (F's B3 network: fast GRU + slow leaky latent, β = 0.5); starts exactly at A0 | ≈ 14.8k |

A5 (explicit fast/slow regulator) is trained only if G3's Q2 holds (then by an amendment here before G4-X).
A6 (model-based search) is not included: the consequence predictor already scores each action over a 2 s horizon,
and a multi-step search would need a learned feature-dynamics model that no G1–G3 result motivates.

## Training methods (`scripts/g45_train.py`; 5 seeds each; the same hyper-parameters for every cell, no search)

| method | interaction budget (decisions, all counted) | cells |
|---|---|---|
| ppo — recurrent PPO from scratch (F's settings, lr 3·10⁻⁴, 8 lifetimes × 5 episodes per update) | 2.4M | A2, A3, A4 |
| bc — behaviour cloning of A0 (teacher rollouts, cross-entropy, 4 epochs) | 0.3M | A2, A3 |
| bcppo — bc (0.3M) then PPO | 2.4M | A2, A3 |
| dagger — 3 rounds × 0.1M, β = 1, 0.5, 0.25; every visited state labelled by A0; aggregated | 0.3M | A2, A3 |

A4 with bc would only learn its zero residual (= A0), so A4 is trained by PPO (residual policy learning around A0).
Training lifetimes G4-T (5 episodes); validation (greedy mean J on G4-V 0–7) every 96k decisions and after every
BC/DAgger round; the best checkpoint is kept. Learning-curve area = trapezoidal mean validation J per decision.

## Test

G4-X lifetimes 0–**399** × 5 episodes (the training/validation length), all 45 models greedy at their checkpoints
plus A0 and A1. N = 400 is the cap: F's E4 pilot SEs for the same comparison type required 414–506 lifetimes; the
achieved power is below 80% for r = −10% and is reported.

## Primary families (`scripts/g45_analyze.py`; selections use training/validation records only)

**G4** — for each architecture, its best method (highest mean best-validation J over seeds) vs A0, superiority rule,
two-level bootstrap (lifetimes, then seeds), Holm over the six one-sided p-values. If no neural candidate beats A0,
A0's family is the G4 choice.
**G5** — (a) the best architecture's best method vs its PPO-from-scratch cell (superiority; omitted if PPO is the best
method); (b) the best ordinary-GRU cell (A2/A3) non-inferior to A0 (+10% collisions, −2 pp completion). Holm over
the (up to) three p-values.
Secondary: every cell vs A0 and vs its PPO cell; learning-curve area; interactions and wall time; seed variance;
action distributions and entropy (GO/YIELD collapse); A1 vs A0; F's E4 B3/B4 action use (from F's committed
results) for the collapse diagnosis.
