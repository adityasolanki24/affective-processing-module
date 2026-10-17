# G3 addendum: what persistent internal state is needed? (fixed before any G3 data, 2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G3. Nothing here relaxes a criterion.

## Question as operationalised

Every rule-based candidate decides with an action-conditioned consequence predictor. G1 showed that persistent
modulation of the *collision weight* does not help; G3 asks whether the *predictor itself* needs persistent internal
state. All models (`src/apm_select/stateful.py`) predict P(collision within 2 s), P(slip within 2 s) and progress for
the action taken, from the same 41 decision features, trained on the same data with the same heads, losses, early
stopping and 5 seeds (0–4); they differ only in internal state:

| kind | state |
|---|---|
| none | no state (MLP on the current features; F's architecture, retrained on G3 data) |
| ema | fixed exponential smoothing, τ = 2 s and 30 s |
| lds | learned diagonal linear dynamical system (16 dims) over the last 10 s |
| gru | single learned recurrent state (GRU 32) over the last 10 s |
| fastslow | GRU 32 (fast) + lifetime-persistent EMA τ = 120 s (slow) |
| pad | APM v0.1 PAD (V, A, D) + tonic appraisal u as extra inputs (F's affect state) |

Controls (evaluation only): `reset` (no history) and `shuffled` (history from another lifetime).

## Data and training (stage A, partitions T/V/P only)

Exploration lifetimes (F's randomised safe action variation, 5 × 60 s, APM observe-only for the `pad` inputs):
G3-T 0–149 (training), G3-V 0–39 (early stopping), G3-P 0–39 (pilot). Adam 10⁻³, batch 512, ≤ 40 epochs,
patience 5.

## Closed loop

Rule `argmax_a progress − λc·P_col − λs·P_slip` with the 5-seed ensemble of each kind. **Equal tuning:** K = 16
Latin-hypercube (λc, λs) per kind (λc log-uniform 0.1–10, λs 0–4, seed 20261012 + 10·kind index) on G3-V lifetimes
100–129 × 10 episodes; selection by mean J. **best_state** = argmax validation J over the stateful kinds (ema, lds, gru,
fastslow, pad); **best_single** = argmax over ema, lds, gru. Reference A0_R2 = G1's selected rule with F's frozen
predictor (secondary).

## Test (stage B)

- **Predictive (secondary):** G3-X exploration lifetimes 0–99; lifetime-clustered two-level bootstrap of the relative
  collision Brier of every kind (and control) vs `none`, Holm over the five kinds, F's E1 rule (≤ −5% and p < 0.05:
  adds information; CI within ±5%: no meaningful information).
- **Closed loop (primary):** G3-X lifetimes 0…N−1 × 10 episodes, all kinds + A0_R2 + reset/shuffled controls for
  gru, fastslow and best_state. **N** from the stage-A closed-loop pilot (G3-P lifetimes 0–39) by the master rule
  (80% power at r = −10% / +3 pp for Q1 and Q2), bounds 200–400, recorded in an amendment below before X runs.
- **Primary family** (Holm over four one-sided p-values; `scripts/g3_analyze.py`): **Q1** best_state beats none
  (superiority rule); **Q2** fastslow beats best_single (superiority rule).
- Decision: if Q1 fails, the predictor needs no persistent state (memory + the instantaneous predictor suffice); if
  Q1 holds and Q2 fails, a single timescale suffices; a separate slow state is needed only if Q2 holds.

## Amendment (sample size; to be filled before X)

Committed after stage A (data, training, tuning on V, closed-loop pilot on P), before any G3-X data. Validation
selections (`configs/G3_selected.json`): best_state = **ema**, best_single = **ema** (so Q2 compares fastslow with ema).
Pilot (G3-P, 40 lifetimes) required 159 (Q1: ema vs none) and 181 (Q2: fastslow vs ema) lifetimes for 80% power →
**N = 200** (floor). Closed-loop X: G3-X lifetimes 0–199 × 10 episodes; controls for ema, gru and fastslow.
Predictive X: exploration lifetimes G3-X 0–99 are generated from a disjoint seed stream (`g3_data.py` uses
`child(seed, 500)` for the exploration policy; the closed-loop runs use the same world seeds with their own
policies, so worlds are shared but behaviour and outcomes differ).
