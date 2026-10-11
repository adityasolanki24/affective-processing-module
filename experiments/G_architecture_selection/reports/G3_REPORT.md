# G3 — What persistent internal state is needed? (2026-10-12)

Preregistered in `prereg/G3_ADDENDUM.md` (amendment N = 200). Six consequence-predictor kinds trained on identical
G3-T exploration data (5 seeds each), equal λ tuning (K = 16) on G3-V, closed-loop test on G3-X (200 lifetimes ×
10 episodes; commit `bb6409f`, lock passed; `results/G3/g3_X.json`) and predictive test on 100 exploration lifetimes
(`results/G3/g3_predictive_X.json`). Validation chose `ema` as both best_state and best_single.

## Primary (Holm over four p-values)

| test | collisions (rel.) | completion | decision |
|---|---|---|---|
| **Q1** best_state (ema) beats none | +9.6% [+3.1, +16.8] | −0.3 pp | **fails** (state is worse) |
| **Q2** fastslow beats best_single (ema) | −10.8% [−16.8, −4.6] (p_Holm 0.002) | −0.7 pp [−1.3, −0.1] | **holds** (criterion A) |

## Secondary: every state vs no state

| kind | closed loop: collisions | completion | predictive: rel. collision Brier vs none |
|---|---|---|---|
| ema (2 s, 30 s) | +9.6% [+3.1, +16.8] | −0.3 pp | +2.9% [+1.2, +4.8] |
| lds (learned linear) | +20.8% [+12.0, +31.0] | −0.7 pp | −0.2% [−1.5, +1.3] |
| gru (10 s) | +18.0% [+8.6, +28.6] | −0.7 pp | +3.4% [+1.9, +4.9] |
| fastslow (gru + 120 s EMA) | −2.2% [−7.8, +3.4] | −1.0 pp [−1.6, −0.4] | +3.7% [+2.0, +5.5] |
| pad (F's affect state as inputs) | +7.9% [+1.1, +15.6] | −0.5 pp | −0.4% [−1.5, +0.7] |
| A0 (F's frozen predictor, G1 rule) | +3.4% [−2.2, +9.7] | +0.1 pp | — |

Controls: the stateful models do *use* their history — replacing it with another lifetime's history raises collisions
by 53–55% (ema, gru) and resetting it costs 5–13 pp completion — but that dependence does not make them better than
the stateless predictor.

## Decision

Q1 fails: **the consequence predictor needs no persistent internal state**; current features plus explicit memory
suffice. Q2 holds only relative to a weak single-timescale choice (ema is worse than no state); the fast + slow model
is merely on par with no state (−2.2%, CI spans 0) while losing 1.0 pp completion and costing 4× the compute, so no
slow state is warranted. PAD readouts as predictor inputs add nothing predictively (−0.4%, within ±5%) and worsen
closed-loop collisions (+7.9%), consistent with E1 (programme F).
