# Architecture

The current APM architecture, as validated by programmes G–J. For which parts are demonstrated, see
[STATE.md](STATE.md).

```
 sensors (robot-specific)
   │  only robot-observable signals: IMU, contacts, own commands, tracked entities
   ▼
 ┌─ embodiment observation model ───────────────┐   robot-specific; e.g. NERVA: efference-copy gate,
 │ encounter evidence: dense outcome category   │   sensing pause, self-calibrated push detector, acc8 code
 └──────────────────────────────────────────────┘
   │ Obs(identity, kind, context, category, detected)
   ▼
 ┌─ APM core (robot-independent) ─────────────────────────────────────────────────────────────┐
 │ 1 world learning      consequence predictor: P(outcome | state, action)    apm_select       │
 │ 2 experience memory   per-entity belief b over learned types, change-aware apm_revise       │
 │                       (EM-fitted emissions; λ forgetting; kind pooling)    apm_transfer     │
 │ 3 regulation          readout (p, u, g) → action preferences / risk penalty                 │
 │ 4 expressive intent   ExpressionIntent: proximity, tempo, hesitation,      apm_express      │
 │                       directness, gaze, scanning, lean, head elevation                      │
 │                       mapping: L0 lookup (current) | decision-theoretic (candidate)         │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
   │ ExpressionIntent (bounded, abstract)
   ▼
 ┌─ embodiment adapter ──────────────────────────┐   NERVA: which of 7 validated commands, when;
 │ realises intent with validated primitives only│   min dwell; clipped, rate-limited head and torso
 └───────────────────────────────────────────────┘
   │ motion request
   ▼
 ┌─ independent safety monitor ──────────────────┐   overrides everything: tilt, joint margins,
 │ never depends on APM; neutral fallback        │   saturation → neutral controller
 └───────────────────────────────────────────────┘
   ▼
 validated motion controller (NERVA neutral ONNX policy / wheeled base)
```

## Components

| component | responsibility | interface | learning |
|---|---|---|---|
| observation model | turn raw sensing into encounter evidence; exclude self-caused motion | robot signals → `ObsX` | thresholds self-calibrated from the robot's own quiet baseline |
| consequence predictor | predict collision/slip/progress per action | features → probabilities | supervised on the robot's own outcomes (offline) |
| experience memory | belief over latent entity types; revisable | `observe(Obs)`, `readout(Probe) → (p, u, g)`, `state()/load_state()` | EM on unlabelled outcome counts (offline); Bayesian update online |
| regulation | trade risk against task progress | readout + predictions → action preferences | λ tuned on validation data |
| expressive intent | abstract movement qualities | `ExpressionIntent` (8 channels in [0, 1]) | L0: none. Decision-theoretic: 5-parameter time model from the robot's own exploration |
| embodiment adapter | realise intent on one body | intent → validated commands | none |
| safety monitor | authoritative envelope | physical state → trip / neutral | none |

Code that knows NERVA joint names or wheeled pedestrian identities lives only in adapters, observation models and
testbeds. Memory and the mappings take abstract `Obs` and `Readout` objects.

## Learning mechanisms

All learning signals come from observed outcomes, never from the robot's own affect or its predicted risk.

| what | how | when |
|---|---|---|
| entity types | EM with BIC over outcome-category counts per entity | offline, on training partitions |
| entity belief | Bayes update per encounter; relaxes toward the prior at rate λ | online, every observable encounter |
| consequence predictor | small MLP ensemble on logged outcomes | offline |
| time model (candidate mapping) | least squares of the robot's own arrival time on intent | offline, from exploration |

No deployment-time weight updates are validated (programme G, G7).

## Safety boundaries

- APM outputs are **advisory**. The robot must behave safely with APM absent, late or wrong.
- The safety monitor and the validated controller are never modified by APM, and APM never bypasses them.
- Memory is trained only on observed outcomes. Avoiding an entity teaches nothing about it, and there is no
  counterfactual credit.
- No foundation model sits in the motion or safety path.

## Adding an embodiment

1. **Observation model.** Which encounter outcomes can this robot actually sense? Code them densely, and exclude
   self-caused signals.
2. **Expression adapter.** Map each `ExpressionIntent` channel to validated primitives, or ignore it. Never invent
   capabilities.
3. **Fit the type model** on that robot's own unlabelled training data. The memory code is unchanged.
4. **Safety monitor:** that robot's own, independent of APM.

For the v0.1 AffectLink runtime, the worked adapters are `apm.adapters.nerva` (biped) and `apm.adapters.wheeled`.

## The v0.1 runtime and AffectLink

`src/apm` holds the original runtime:
- AffectLink protocol, codec and validation;
- PAD attractor affect, tendencies and memory;
- optional GRU predictors.

**AffectLink** ([affectlink.md](affectlink.md)) remains the robot-facing message contract. The **PAD core is
superseded**: programme G showed it harmful. Migrating the current architecture into a production runtime behind
AffectLink is on the roadmap. It happens only after the interfaces above are validated on more than one embodiment.
