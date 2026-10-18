# APM v1.0 protocol: experiments A–F in Isaac Sim

## Governance

- **Append-only.** Each section is frozen by the commit that adds it, before any data it governs exist (partitions V
  and X). A frozen section is never edited. Changes go to `DEVIATIONS.md`, with a reason, as a new iteration with
  fresh confirmation data.
- **Seeds** come only from `seeds.py` (new entropy, independent of programmes F–L):
  - P (pilot): debugging only;
  - T: exploration data for fitting;
  - V: calibration of the simple rule;
  - X: locked test;
  - D: demonstration lifetimes for the videos.
- **Simulation only.** No human-participant data. No claim of emotions, consciousness or production safety.

---

## 1. Fixed setup (frozen 2026-10-19, before any T, V or X data)

**Engine.** Isaac Sim 5.1.0 / PhysX, Isaac Lab 2.3.2 (`TRACKER.md`). Every result is labelled with its engine.

**Robots.** All three passed their M3 gate batteries (`TRACKER.md`, `results/m3`):

| robot | controller | physics | approach distance | expression adapter |
|---|---|---|---|---|
| NVIDIA Nova Carter | the asset's wheel drives | 500 Hz | 2.2 m | (v, ω) `WheeledAdapter` |
| Booster T1 | Isaac Lab PPO v1 + frozen velocity map | 200 Hz | 2.2 m | (v, ω) `WheeledAdapter` |
| NERVA | frozen neutral ONNX (sha 610b1c59…), NERVA's control step and SafetyMonitor | 500 Hz | 1.1 m | H's `BipedExpressionAdapter`, admitted channels |

**Scenes** (`configs/scenes.json`): warehouse (development), home (development), hospital (held out). Each scene has
three stations (P1–P3) at 2.2 m from its hub, and one bystander walking a 3 m patrol at 1 m/s, chosen automatically
in free space away from the stations and approach lines. The hospital is used only in §5 (D).

**People.** Stock Isaac characters, animated through Isaac's people AnimationGraph. The physical body is a kinematic
capsule (r 0.22 m, 1.7 m). Lunges follow the K/L type distributions (calm, pushy), mapped to physical geometry by
`agents.GAP0`. Surface speed is capped at 0.15 m/s, NERVA's survivable repeated push (gate battery).

**Sensing.** No simulator truth reaches APM. Per encounter the robot sees:
- its IMU (quiet windows, efference-copy gate, its own self-calibrated detector);
- its depth eye (64×48, 40° hFOV, −10° pitch, noise and dropout as L) through `MotionPerception` (articulated
  target, abstract-frame target distance from navigation, the person's standard radius);
- its odometry.

**Encounter.** As K/L, from the approach point:
- settle 1 s;
- approach under the intent;
- sensing windows of 4 s (1–3), with abort on the robot's own detection;
- a stand-off of 0.3 m for OBSERVE/HOLD.

**Lifetime.** 24 encounters:
- B1 at P2 (probability 0.5), A1 at P1 and C1 at P3 (0.25 each);
- A1 and C1 are calm;
- B1 follows the history: change_B (pushy → calm) or late_B (calm → pushy), switching at encounter 10.

A fall ends the lifetime and is counted.

**Runtime.** One integrated `apm_transfer.runtime.Runtime`:
- memory: change-aware type memory, λ 0.02, transition each meeting;
- task: handoff;
- risk supervisor on for the APM systems;
- each robot's physical safety layer always on.

**Default APM system:** DT (decision-theoretic, no VOI).

**Systems:**

| system | decision | memory |
|---|---|---|
| DT | decision-theoretic, no VOI | learns |
| NOMEM | DT | never learns (kind prior every encounter) |
| FROZEN | DT | learns until the switch, then frozen |
| SHUFFLED | DT | learns, each observation filed under a random identity |
| FIXED | proximity 0.5, tempo 0.7 every time | – |
| NEUTRAL | neutral motion at the hand-over distance | – |
| LOOKUP | H's lookup, L's tuned parameters (c=0, a=4) | learns |
| MONOTONE* | monotonic rule calibrated per robot in §3 | learns |
| REACTIVE | memory-free conventional baseline: neutral approach, 3 windows, leaves on its own detection (same observations) | – |

## 2. Fitting (partition T)

Per robot, RANDINTENT lifetimes (random grid intent, 1–3 windows):
- histories risky_B and safe;
- T indices 0–11 in the warehouse and 12–23 in the home scene (48 lifetimes per robot).

`k_fit.py --robot {nova_carter|t1_isaac|nerva_isaac}` fits:
- the robot's own acc8 edges (pooled quantiles);
- its time model (least squares on its own arrival times);
- its type model (EM, K = 1–4 by BIC; IMU blocks in three distance bands plus the camera block).

**Task values.** As K's frozen rule:
- `retry_s` = the robot's mean predicted approach time over the intent grid (its own time model);
- `price_s` 20 s per lunge contact;
- `risk_cap` 0.30;
- `d_task` 0.30 (handoff).

## 3. Calibration of the simple rule (partition V)

Per robot, in the warehouse:
- MONOTONE with (lo, width) ∈ {0.1, 0.2} × {0.2, 0.4};
- V indices 0–2 per history (6 lifetimes per configuration).

MONOTONE* is the configuration with the lowest mean realised cost (§6); ties go to the earlier configuration. It is
written to `configs/M6_tuned.json` and committed before any X run.

## 4. Main study (partition X, warehouse): experiments A, B, C, E

Per robot:
- systems DT, NOMEM, FROZEN, SHUFFLED, FIXED, NEUTRAL, LOOKUP, MONOTONE*;
- histories change_B and late_B;
- X indices 0–9 (20 lifetimes per system).

Systems are paired by index: the same seeds give the same lunge streams.

## 5. Environments (partition X, home and held-out hospital): experiment D

Per robot and scene:
- systems DT, MONOTONE*, REACTIVE;
- histories change_B and late_B;
- X indices 10–15 (12 lifetimes per system).

## 6. Measures

Per encounter, with d the abstract stopping distance from the robot's own odometry:
- `contacts`: the person's lunges that reached the robot (PhysX truth; evaluation only);
- `success`: the robot reached the interaction zone, d ≤ d_task + 0.05;
- **realised cost** = arrival time (s) + `price_s` × contacts + `retry_s` × (1 − success).

## 7. Hypotheses and criteria (per robot; unit = lifetime; paired by index)

| | hypothesis | measure (per lifetime) | test |
|---|---|---|---|
| A1 | experience changes behaviour | change_B, pre-switch B1 encounters after the 2nd B1 meeting: contacts per encounter | DT < NOMEM |
| A2 | identity-specific memory | same window | DT < SHUFFLED |
| B1 | adapts calm → pushy | late_B, post-switch B1 encounters after the 2nd post-switch B1 meeting: contacts per encounter | DT < FROZEN |
| B2 | adapts pushy → calm | change_B, post-switch B1 encounters after the 2nd: success rate | DT > FROZEN |
| C1 | expression from memory beats a fixed style | all encounters, both histories: mean realised cost | DT < FIXED |
| C2 | beats neutral motion | same | DT < NEUTRAL |
| C3 | beats the fixed lookup | same | DT < LOOKUP |
| C4 | beats the calibrated simple rule | same | DT < MONOTONE* |
| D1 | holds in the held-out hospital | mean realised cost | DT < REACTIVE |
| D2 | same | mean realised cost | DT < MONOTONE* |
| D3 | development home scene | mean realised cost | DT < REACTIVE |
| E | holds across robots | A1, B1, B2, C2, D1 | the number of robots (of 3) on which each passes |

- **Test.** One-sided paired sign-flip permutation test on per-lifetime differences (10 000 permutations, fixed seed).
  Holm correction across A1–D3 within each robot (α 0.05). 95% bootstrap CIs of the mean difference are reported.
- **Safety, all systems.** Falls, trips, contacts and executed approaches above the risk cap (APM systems:
  must be 0, recomputed from the logged belief) are reported. Any fall is listed individually.
- **Verdict.** Each hypothesis is Demonstrated (passes on all 3 robots), Partially demonstrated (1–2 robots) or Not
  demonstrated (0). A wrong-direction significant result is reported as Failed.

## 8. Robustness (F; partition X, warehouse, Nova Carter unless noted; descriptive)

| check | procedure | pass |
|---|---|---|
| restart | save the runtime after encounter 12, new process, load, continue (DT, 2 lifetimes) | decisions identical to an uninterrupted run |
| replay | `runtime.replay` on every logged §4 DT lifetime | 0 mismatches |
| missing sensor | camera off (IMU only), DT, 4 lifetimes per robot | no falls, no supervisor violations |
| identity uncertainty | AffectLink identity confidence below 0.5 on B1 (anonymous track) | kind prior used; no B1 memory update |
| controller interruption | motion commands dropped for 2 s mid-approach, 10 encounters | no fall; the encounter completes or aborts |
| camera dropout | dropout 0.3, DT, 4 lifetimes | no falls |
| safety veto | the decision layer forced to an over-cap approach | supervisor vetoes; not executed |
| unexpected contact | a scripted 0.15 m/s contact during the approach | IMU detection, no fall |
| repeated perturbations | pushes every 1–2 s during an encounter | no fall at ≤ 0.15 m/s |
| long operation | one 100-encounter DT lifetime per robot | no fall; memory stays bounded; replay identical |
