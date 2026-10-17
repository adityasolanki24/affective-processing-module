# H1b report: push re-test after the stop-transition fall (2026-10-12)

## Verdict: **G1′ passes, so H1 passes** (G2–G5 carried over from H1)

Governed by `prereg/H1b_ADDENDUM.md` (commit `766057c`), on fresh partition-X indices. Results are in
`results/H1b/X/`.

### Transition push sweep (1,200 trials)

Setup: walk 5 s, then rest; push at 0–2 s after the stop; 8 directions; 5 seeds.

| magnitude (m/s) | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 |
|---|---|---|---|---|---|
| falls / 240 | 0 | 0 | 0 | 1 | 4 |

- m** = 0.10 m/s, which meets the preregistered floor of 0.10.

### Push arm re-test at 0.10 m/s

- 7 styles × 8 geometries × 3 seeds = 168 trials;
- **0 falls, 0 trips, 100% arrival.**

## Push tolerance across the programme

| state | tolerance (m/s) | source |
|---|---|---|
| steady walking | 0.30 | NERVA's gate |
| steady standing | 0.15 | H0b |
| just after a stop (expressive rhythms) | 0.10 | H1b |

**The expressive interface is certified for pushes up to 0.10 m/s.** Pauses and stop-go rhythms, which are central
to several styles, place the robot in its least robust state. This is a cost of expression on this controller,
reported for NERVA. It is not hidden by the H1 verdict.

## H1 conclusion (claim A, simulation)

**Repertoire:** cautious, assertive, curious, hesitant, relaxed and energetic, plus neutral. They are reliably
distinct and controllable from non-speed motion features:
- in unseen geometries;
- at matched speed;
- under pushes;
- with 0 falls at ≤ 0.10 m/s pushes, and 100% task completion.

H3/H4 may start (`prereg/H34_ADDENDUM.md`).
