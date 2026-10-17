# J6 addendum: does expression need to be learned? (exploratory; committed before any J6-X data, 2026-10-18)

This addendum supplements `MASTER_PREREGISTRATION.md` §4.4.
- **J6 is exploratory** (DEVIATIONS D2): J5's gate failed. Its result cannot by itself adopt an architecture.
- The proximity value and the `_V` systems are a disclosed addition (DEVIATIONS D3).

## Supervision (no human judgements; simulator truth is never a target)

**Time.**
- `TimeModel`: least squares of the robot's own arrival time on (1, proximity, tempo, hesitation,
  tempo·hesitation).
- Data: J6-T exploration, 200 RANDINTENT lifetimes, 6,000 encounters. R² 0.905.
- Coefficients: [19.71, 10.23, −17.49, −1.58, 6.68].

**Perturbation.** P(acc8 ≥ 6 | learned type, distance band), from the J5 distance-conditioned type model, which was
learned from robot evidence:

| type | 0.2 m | 0.4 m | 0.6 m |
|---|---|---|---|
| type 0 (calm-like) | 0.135 | 0.112 | 0.109 |
| type 1 (pushy-like) | 0.610 | 0.445 | 0.234 |

**Utilities.** U = arrival time + price × strong-evidence rate (lower is better).
- **U** (preregistered in the J6 machinery): price 10, no proximity value.
- **U_V** (D3): price 40, plus value 5 s per 0.1 m of shortfall beyond 0.3 m.

## Mappings and what they choose

| system | mapping | fitted parameters | choice by belief b(pushy) |
|---|---|---|---|
| CL | H's fixed L0 | 0 | 0.40–0.53 m (J5), slow tempo, hesitation |
| LEARNED | argmin U over 20 grid intents | 5 (time) | **0.6 m, tempo 1, no hesitation, for every b** |
| LEARNED_V | argmin U_V | 5 | 0.3 m (b ≤ 0.2); 0.4 m (0.3–0.5); 0.6 m (≥ 0.6); always tempo 1, no hesitation |
| RESIDUAL_V | L0, proximity ± {0, 0.15, 0.3} by U_V | 5 | L0 shifted toward U_V's distance |
| GRU_V | GRU (hidden 16) over the entity's raw evidence history, imitating LEARNED_V | ≈ 1,800 | see `configs/J6_gru_v.json` |
| FROZEN | neutral | 0 | 0.4 m, neutral tempo |

**GRU_V training** (`j6_fit.py`, J6-T, split 80/20 by lifetime): validation accuracy **0.998**, against a majority-class
baseline of 0.566 (target classes: 3 intents).

LEARNED's GRU (`configs/J6_gru.json`) imitates a constant and is not run.

## Frozen inputs (git-normalised SHA-256, first 16 hex)

| file | hash |
|---|---|
| `configs/J6_model.json` | `32658d9943afbe23` |
| `configs/J6_model_v.json` | `cec059a8699afec0` |
| `configs/J6_gru_v.json` | `670c16dd7bbe507b` |
| `configs/J6_gru.json` | `51041ea16795c3a7` |
| `configs/J5_types_distance.json` | `01468d894b80975d` |
| `configs/J4_edges.json` | `f89ed63dc918634b` |
| `configs/H_admitted_interface.json` | `abeb2e0575f5e9ce` |

## Test (J6-X)

**Command:**

```
j5_closed_loop.py --partition X --first 40 --n 40 --lam 0.02 --systems CL,FROZEN,LEARNED,LEARNED_V,RESIDUAL_V,GRU_V
    --types-model configs/J5_types_distance.json --edges configs/J4_edges.json
    --j6-model configs/J6_model.json --j6-model-v configs/J6_model_v.json --gru-v configs/J6_gru_v.json
```

**Size:**
- 6 systems × 4 histories × 40 = **960 lifetimes**, 28,800 closed-loop encounters;
- J5 seeds at fresh indices 40–79, disjoint from J5-X.

The protocol is J5's: efference gate, sensing pause, H's admitted interface, safety monitor.

## Measures (`j6_analyze.py --price 10 --v-price 40 --v-value 5`; paired by seed; bootstrap 95% CIs)

| | measure |
|---|---|
| E1 utility | each learned system minus CL, under its own utility (LEARNED: U; `_V`: U_V) and under the other |
| E2 safety | falls (0 required), safety trips (≤ 1%), arrival rate; contacts and max push vs CL |
| E3 loop (J5 Q3) | risky_B, late B1 encounters: max push and contacts minus FROZEN |
| E4 evidence dependence | Spearman(belief, stopping distance) over B1 encounters; stopping-distance effect risky_B − safe; history AUROC (Q1) |
| E5 revision | Q4 and Q5 (`j5_analyze.py` on the same data) |
| E6 complexity | fitted parameters, as in the table above |

## Reading rules (fixed now)

**A learned mapping "improves on L0"** only if all of the following hold:
1. E1, under its own utility: the CI upper bound < 0;
2. E2: 0 falls, and contacts are not worse than CL (difference CI upper bound ≤ +0.05 per encounter);
3. E4: Spearman ≥ 0.5, and the stopping-distance effect has CI lower bound > 0. The improvement must not come from
   abandoning evidence dependence.

**"Persistent recurrent state is necessary"** only if GRU_V beats LEARNED_V on U_V (paired CI upper bound < 0) with E2
and E4 also met. Otherwise the explicit memory plus a stateless mapping is sufficient.

**The loop is closed by a learned mapping** if, for that system, E3 shows CI upper bound < 0 for max push, with
J5's Q1, Q2 and Q4 criteria also met.
