# H1 report: NERVA's expressive movement repertoire (2026-10-12)

## Verdict: **H1 gate FAILS on G1 (safety)**; G2–G5 pass

**Setup:**
- Confirmatory run governed by `prereg/H1_ADDENDUM.md` (commit `e68087a`, locks passed);
- 976 simulated trials, local;
- results in `results/H1/X/`, `results/H1/S/` and `results/H1/X/h1_results.json`.

| criterion | result |
|---|---|
| G1 safety | **fail.** One fall in 112 push-arm trials at the 0.15 m/s push. The core (280), matched (240), random (80), switch (96), shift (56) and E1′ (112) arms had 0 falls and 0 trips. |
| G2 task | pass: arrival 100% for every style in core and matched |
| G3 distinctness | pass: non-speed leave-one-geometry-out balanced accuracy **1.00** (all 7 classes, recall 1.00) |
| G4 matched speed | pass: speeds within −1.6% / +3.4% of the mean; non-speed balanced accuracy **1.00** (6 classes) |
| G5 perturbation | pass: classifier trained without pushes, tested on pushed trials, 0.98 |

As preregistered, H3/H4 do not start on this result. The failure stays on record.

## The fall

**What happened** (style relaxed, geometry 5, sim seed 1199366656, push 0.15 m/s at t = 9.98 s):
- The push came from behind (169° relative to heading).
- It landed during a rhythmic pause, 0.2 s before the next forward burst.
- The robot had **just stopped** and was still rocking backward: tilt rising from 2.6° to 11° before the push.
- Afterwards the adapter commanded forward, then a turn correcting the displaced heading; tilt reached 25.8° (trip)
  and the robot fell.
- Replayed without the push, the same trial does not fall.

**Diagnosis.** H0b measured standing robustness only at *steady* rest, after 3 s of standing. A pilot sweep of pushes
after a walk-to-rest transition (partition P, 480 trials, `results/H1b/pilot/`) shows the gap:
- 0.15 m/s pushes 0.4 s after the stop topple the robot in 2/16 trials;
- ≤ 0.125 m/s pushes never did, at any delay in {0, 0.2, 0.4, 0.7, 1.0, 2.0} s.

**The stop transition is the controller's weakest state, and expressive rhythms create many of them.**

## What H1 established (claim A, simulation)

**Repertoire.** Six designed movement qualities (cautious, assertive, curious, hesitant, relaxed, energetic) plus
neutral are reliably distinct:
- distinguishable from non-speed features in task geometries the classifier never saw (1.00);
- still distinguishable without the head channel (non-speed, non-posture: 0.99);
- still distinguishable at matched speed (1.00; without posture 0.93);
- still distinguishable under pushes (0.98) and in new geometries (0.98);
- arrival 100%.

**By feature group** (core, leave-one-geometry-out):

| group | accuracy |
|---|---|
| posture | 1.00 |
| spatial | 0.97 |
| temporal | 0.75 |
| acceleration | 0.50 |

The controller's own gait dominates acceleration, so acceleration is a weak channel.

**Speed range.** The progress guarantee floors slow styles at ≈ 0.029 m/s. The full range is 0.029–0.065 m/s.

**X1 vs X2.**

| | falls | arrival | SPARC | power (W) | non-speed diversity |
|---|---|---|---|---|---|
| X1 designed | 0 | 100% | −5.52 | 3.42 | 6.04 |
| X2 random | 0 | 100% | −5.95 | 3.54 | 6.20 |

- Random intents are as safe, slightly less smooth, and about as diverse.
- **Being designed adds no measurable motion advantage.** Whether it adds *meaning* is H2's question.

**X3 dimensionality (PCA of X2, non-speed).** Ten components are needed for 90% of the variance; the first three
explain 51%. The interface produces high-dimensional variation, not a low-dimensional style manifold.

**E1′ secondary configuration:** 0 falls in 112 trials, 100% arrival, 1.00 distinctness.

**Caveat.** Distinctness is high partly because the adapter executes deterministic, fixed programmes. "Reliably
controllable" here means repeatable in physics across geometries, seeds, latency and perturbations. It says nothing
about what observers perceive (H2, untested).

## Next

`prereg/H1b_ADDENDUM.md`:
- measure push tolerance across stop transitions on X;
- re-test G1 at that magnitude on fresh seeds;
- **if the transition-validated magnitude is below 0.10 m/s, H1 fails definitively.**
