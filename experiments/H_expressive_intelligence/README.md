# Programme H: APM v0.4 developmental expressive intelligence

Question: can a robot develop persistent, experience-dependent internal states and express them through
physically embodied movement, and what is the smallest mechanism that does it? The answer must beat a predictor +
memory + fixed style-lookup baseline.

## Documents

| file | contents |
|---|---|
| `MASTER_PREREGISTRATION.md` | claims A/B/C, gates, decision rules A–E |
| `LITERATURE_REVIEW.md` | prior work |
| `EXPERIMENT_MATRIX.md` | status of every experiment |
| `RESEARCH_LOG.md` | what was done and when |
| `DEVIATIONS.md` | departures from the preregistration |
| `prereg/` | per-experiment addenda |
| `reports/` | audit and results |

## Code

The expressive layer lives in `src/apm_express/` and is removable; APM v0.3 code is unchanged.

| module | role |
|---|---|
| `intent.py` | `ExpressionIntent`: bounded, embodiment-independent movement qualities |
| `adapter.py` | `BipedExpressionAdapter`: intent → NERVA's 7 validated commands, minimum dwell, head-at-rest offsets, rate limits |
| `safety.py` | `SafetyMonitor`: physical state only, NERVA's trip rule + joint-margin / saturation trips |
| `biped.py` | `NervaBiped`: NERVA's frozen ONNX policies behind NERVA's own deployment contract |
| `episode.py` | closed loop: intent → adapter → controller, with the monitor and neutral fallback |
| `features.py` | motion features (speed, temporal, posture, acceleration, spatial, cost) |
| `styles.py` | H1 designed styles (X1) and random styles (X2) |
| `kinematic.py` | kinematic stand-in for unit tests only |

## Running NERVA simulations

NERVA simulations need:
- NERVA's inference environment: Open_Duck_Playground at `b9be205a`, mujoco 3.3.0, onnxruntime 1.21.0, pinned to
  Open Duck's inference lock;
- the environment variable `NERVA_ROOT` pointing to the NERVA checkout;
- `PYTHONPATH=<repo>/src;<NERVA_ROOT>`.

Example:

    python experiments/H_expressive_intelligence/scripts/h0_interface.py --partition P --suite switching --out DIR
