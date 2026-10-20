# Architecture

The current APM architecture, as validated by programmes G–L. For which parts are demonstrated, see
[STATE.md](STATE.md). Since programme L, the whole loop runs in one runtime, `apm_transfer.runtime` (below).

```
 sensors (robot-specific)
   │  only robot-observable signals: IMU, depth camera, contacts, own commands, tracked entities
   ▼
 ┌─ embodiment observation model ───────────────┐   robot-specific; e.g. NERVA: efference-copy gate,
 │ encounter evidence: dense outcome category   │   sensing pause, self-calibrated push detector, acc8 code;
│ + camera code from motion perception         │   depth camera: MotionPerception (pixels, efference copy,
│                                              │   navigation distance; never simulator state)
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
 │                       mapping: decision-theoretic (current; L0 retired by K3b)              │
 │ 5 information gather  non-myopic VOI; OBSERVE from beyond reach (needs a camera)  infoseek  │
│ 6 risk cap            predicted risk ≤ cap, else NO_SAFE_ACTION → HOLD beyond reach         │
│ 7 supervisor          independent re-check of every decision (own risk model) → veto/HOLD  │
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
| expressive intent | abstract movement qualities | `ExpressionIntent` (8 channels in [0, 1]) | decision-theoretic: 5-parameter time model from the robot's own exploration; task values and risk cap fixed from requirements |
| information gathering | decide whether to sense longer or from farther away | `infoseek.choose` | none; uses the robot's own learned models (KG* value of information) |
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

## The integrated runtime (programme L)

`apm_transfer.runtime.Runtime` runs layers 2 and 4–7 for any body:
- `decide(identity, kind)` → `Decision` (grid intent, OBSERVE or HOLD; windows; stand-off; diagnostics);
- the body adapter executes it through the body's own expression adapter and safety monitor, and returns an
  `Observation`;
- `observe(obs)` updates memory.

Further properties:
- **Persistence.** State is saved atomically (`save`/`load`); a restart gives bit-identical decisions.
- **Determinism.** Each decision draws from its own seeded stream, and `replay(log)` re-derives every decision
  from the JSONL log.
- **AffectLink.** `ingest(EntityObservation)` binds tracks to identities, and `affectlink(decision)` emits
  `MemoryExpectation`, `ActionTendency` and `ExpressionIntent` (all validated).
- **Bodies:** `NervaBody`, `WheeledBody`, `HumanoidBody` (MuJoCo); `apm_isaac.body.IsaacBody` (Isaac Sim).
- **Baselines share the runtime:** `policy` selects the decision-theoretic mapping (greedy, the v1.0 default; or
  VOI) or a conventional baseline (lookup, monotone rule, fixed style, neutral, reactive, random). `supervise`
  switches APM's risk supervisor; the physical safety layers always apply.
- **Demo:** `experiments/L_audit_perception/scripts/f_runtime_demo.py`.

## Isaac Sim integration (APM v1.0, `src/apm_isaac`)

The runtime is unchanged. Isaac Sim 5.1 (PhysX, RTX) supplies the bodies, scenes, people and sensors:

```
 Runtime.decide ──> IsaacBody (runtime adapter: identity → station, distance → abstract band)
                      │
                      ▼
 IsaacWorld.encounter: driver per body ── NERVA: H's BipedExpressionAdapter + SafetyMonitor
                      │                    Nova Carter / T1: (v, ω) WheeledAdapter + the body's supervisor
                      ├─ people: animated characters (AnimationGraph), kinematic capsule bodies,
                      │          lunges from the K/L type distributions (surface speed ≤ 0.15 m/s)
                      ├─ bystanders: patrol walkers on raycast-validated paths
                      ├─ sensing: IMU from the body's own link (quiet windows, efference gate),
                      │           depth eye posed from the PhysX link pose each frame → MotionPerception
                      └─ truth (evaluation only): PhysX contact reports, lunge contacts
                      ▼
 Observation ──> Runtime.observe
```

| body | controller (never changed by APM) | physical safety layer |
|---|---|---|
| NVIDIA Nova Carter (stock asset) | the asset's wheel drives | shove/tilt supervisor: zero command for 0.5 s, counted trip |
| Booster T1 (stock asset) | Isaac Lab PPO policy + frozen velocity map (`policies/t1_isaaclab_v1`) | tilt supervisor: zero command for 1 s |
| NERVA (MJCF import, `apm_isaac.nerva`) | frozen neutral ONNX with NERVA's control step | NERVA's SafetyMonitor (neutral fallback) |

**Rules carried over.**
- No simulator state reaches APM. People's types, lunge parameters and contacts are evaluation truth only.
- The camera reads only pixels, the robot's own-motion flags, its navigation distance and its own mounting.
- Each robot's type model, IMU edges and time model are fitted on its own exploration data (`k_fit`).

## The v0.1 runtime and AffectLink

`src/apm` holds the original runtime:
- AffectLink protocol, codec and validation;
- PAD attractor affect, tendencies and memory;
- optional GRU predictors.

**AffectLink** ([affectlink.md](affectlink.md)) remains the robot-facing message contract. The **PAD core is
superseded**: programme G showed it harmful. Migrating the current architecture into a production runtime behind
AffectLink is programme L's runtime, in simulation. Replacing `src/apm`'s PAD core as the shipped runtime is
still open.
