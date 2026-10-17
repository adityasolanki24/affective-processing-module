# H0 audit: APM v0.3, NERVA's robot, controllers and earlier expressive-motion work (2026-10-12)

Each item is labelled **[verified]** (run or read in code today) or **[record]** (taken from NERVA's committed reports
and checked against the code where possible). Nothing below is assumed from notes alone.

## 1. APM v0.3 (frozen reference)

Branch `research/apm-v0.3-architecture-selection`, commit `5fb2dd7`.

**Decision A**, a simple predictive regulator:
- a learned consequence-predictor ensemble (MLP);
- explicit outcome-grounded entity memory;
- a calibrated EMA risk penalty;
- a transparent argmax rule.

**Facts carried into programme H:**
- PAD-modulated regulation is harmful under fair tuning (+99% avoidable collisions vs a static penalty, 15× the
  compute).
- The EMA risk state beats the static penalty (−47%).
- Removing memory costs +224%.
- No neural regulator beats A0.
- Revision under selective feedback misses the preregistered criterion.
- No promoted deployment-time update.
- Learned cross-robot transfer is not shown.
- The wheeled testbed has defect D8: unavoidable spawn collisions. Any wheeled analysis in H must use
  avoidable-collision metrics (`apm_select.instrument.collision_split`).

The v0.3 code (`src/apm`, `src/apm_research`, `src/apm_select`, `src/apm_testbeds`) is not modified by programme H.
The new layer lives in `src/apm_express/` and is removable.

## 2. NERVA's robot [verified]

The model is read from `Open_Duck_Playground` at commit `b9be205a`, scene `scene_flat_terrain_backlash.xml`.

- **Open Duck Mini v2 biped** in MuJoCo 3.3.0. Physics runs at 2 ms; the policy runs at 50 Hz (decimation 10).
- **14 position actuators**, class `sts3215`: kp 17.11, force range ±3.23 N·m.
  - 10 leg joints: hip yaw, roll and pitch; knee; ankle; per side.
  - 4 head joints: `neck_pitch` [−0.35, 1.13], `head_pitch` [−0.79, 0.79], `head_yaw` [−2.79, 2.79],
    `head_roll` [−0.52, 0.52] rad.
  - Leg joints have backlash joints in the training scene.
- **No arms, no antennae, no other expressive actuators.** The head is the only non-locomotor body part.
- **Safety in simulation:**
  - NERVA's `SafetySupervisor` stops at tilt ≥ 25° or stability risk ≥ 0.75, and resumes after tilt ≤ 12° for
    0.5 s;
  - a joint-target rate limit of 5.24 rad/s;
  - MuJoCo force and control ranges.
  - Hardware e-stop and torque/temperature limits do **not** exist, per NERVA's own note.

## 3. Controllers

| policy | status | inputs | evidence |
|---|---|---|---|
| upstream `BEST_WALK_ONNX_2` | legacy, no measured envelope | continuous vx ∈ ±0.15, vy ∈ ±0.2, ωz ∈ ±1.0; clock factor | RQ1/RQ1b/RQ1c [record] |
| B2, S1–S6 | superseded; S-policies failed their criteria | style vector (S) | `style_policy_design.md` [record] |
| **neutral turn-translation candidate** (`610b1c59…`) | **passes the neutral long motor gate** | the 7 neutral commands only | gate reproduced today, below |
| **E1′ candidate** (`f67d5cac…`) | passes the same gate at e = (0, 0); E1′ criteria failed (pitch too small, crouch ignored) | 7 commands + e = (e_pitch ∈ [−1, 1], e_crouch ∈ [0, 1]) | gate reproduced today; style sweep reproduction below |

**Neutral motor contract `neutral_motor_v1` [verified in code].** `nerva/sim/open_duck.py` and
`nerva/training/neutral_reference.py` accept exactly these commands; anything else raises `ValueError`:
- rest;
- forward / backward at ±0.074 m/s;
- left / right at ±0.074 m/s;
- turns at ±0.60 rad/s.

The contract also fixes:
- the gait period at 27 steps (0.54 s), through NERVA's own phase clock;
- head offsets at zero;
- style domain-checks for E1′.

**Consequences for expression:**
- No continuous speed, cadence, stride or step-height input exists on a validated controller.
- No validated head channel exists.
- Expression on validated controllers is therefore limited to:
  - **behaviour-level sequencing** of the seven commands: timing, pauses, path, orientation, retreat;
  - E1′'s small pitch input;
  - a head channel, if one can be shown to be safe.

### Reproduction on this machine [verified]

Setup: a fresh Windows virtual environment pinned to Open Duck's inference lock, NERVA at `726c8c6` (clean), and
NERVA's own unchanged runner (`neutral_gate.py`).

| candidate | criteria (stability / steady 5/5 / transitions / pushes), no latency & latency | falls | shadow interventions | agreement with NERVA's recorded trials |
|---|---|---|---|---|
| turn translation | all pass, both conditions | 0 / 160 | 0 | **160/160 trial records byte-identical** (1,038 numeric fields, max difference 0) |
| E1′ at e = 0 | all pass, both conditions | 0 / 160 | 0 | **160/160 byte-identical** |

Results are in `results/H0/reproduce_gate_*`. The simulator is deterministic across Linux (NERVA's runs) and this
Windows machine.

## 4. Earlier expressive-motion experiments: what failed and why [record, cross-checked]

| experiment | method | outcome | audit reading |
|---|---|---|---|
| RQ1 | runtime gait-clock factor on the upstream policy | cadence controlled exactly; speed, stride, pitch and sway all co-vary; 0/10 falls | one *coupled* axis; out of the policy's training distribution; not available on the neutral contract (fixed clock) |
| RQ1b | speed-matched comparison | command dead zone below 0.08 m/s; shared speed range only 0.045–0.056 m/s; command confound | speed and style are hard to separate with continuous commands |
| RQ1c | runtime head offsets (upstream policy) | head follows; walking speed loses up to 84%; the policy ignores the head command in its observation; the sign labels were inverted (positive head_pitch = face **up**) | untrained head motion is a dynamics disturbance; it may still be usable at rest or with small offsets, which **H0 measures for the validated candidates** |
| S1 | style-conditioned imitation (tempo, step height, torso pitch) | tempo and pitch (≈ 1/3 of reference) expressed; step height not; verdict fail | imitation term too weak for small posture changes |
| S2–S6 | added rewards and penalties | trade-offs; S5/S6 collapsed to standing | reward hacking: the touchdown-gated swing cost is avoided by not stepping. **Lesson: no ad-hoc expressive rewards.** |
| E1 | posture conditioning, reference gate first | stopped at the reference gate: raising the body above ≈ 0.218 m is infeasible | feasibility checks before training work as intended |
| E1′ | pitch × crouch on the validated candidate | neutral gate kept; pitch monotonic and speed-independent while moving, but 1.1–3.4° (9–30% of reference); crouch ignored; trained 80 M instead of ≤ 160 M (code deviation) | the only validated continuous posture channel, and it is small |
| human evaluation | — | never started | claim B has never been tested for NERVA |

**Lessons applied in H0:**
1. Do not retrain locomotion for expression in programme H. Any such training needs its own preregistration, a
   reference-feasibility gate and a separate budget.
2. Express through validated channels first, and measure every new channel's envelope before use.
3. Speed matching must be designed in from the start.

## 5. Candidate expressive degrees of freedom (to be measured in H0)

| channel | realisation | status before H0 measurement |
|---|---|---|
| progress tempo (mean speed) | duty cycle of forward vs rest | available, but **speed alone never counts** as a style |
| pausing / hesitation, stop-go, small retreats | rest / backward segments | available if command switching is safe at the needed dwell (**to measure**) |
| approach path (directness, curvature), stopping distance (proximity) | sequencing turns and forward | available if switching is safe |
| body orientation toward / away from target, body scanning | turns in place | available if switching is safe |
| acceleration / deceleration timing | only through switch timing; the controller's own response is fixed | limited |
| torso lean | E1′ e_pitch, 1–3.4° while moving | measured small; within-episode changes untested (**to measure**) |
| head elevation and head yaw (gaze, scanning) | head offsets added after the policy | untested on these candidates; violates the deployment contract's training assumption (**envelope to measure**; admitted only by preregistered criteria) |
| cadence, stride, step height, crouch, continuous speed | — | **not supported** by any validated controller |
| gestures (arms, tail, antennae) | — | **no such actuators** |

## 6. Compute and cloud audit [verified]

- **Local benchmark:**
  - NERVA candidate, native MuJoCo: 20 s simulated in 1.3–1.7 s on one core (≈ 12–15× real time);
  - ≈ 5 s per-process start-up;
  - 3 worker processes give the best aggregate throughput on this laptop (an i7-1355U; 6 workers are slower).
  - The full 160-trial gate takes ≈ 530 s per candidate with one process.
- **Cloud (read-only audit):**
  - no compute instances, disks, static addresses, snapshots or custom images exist in the configured project;
  - **one Cloud Router with a Cloud NAT gateway exists that programmes F and G did not create.** Its name is not one
    of ours. It was created 2026-10-11 18:25 PDT and is probably from another project of the user.
    Cloud NAT is billed hourly while it exists. **It was not touched**; it is reported to the user.
  - Budget information is not readable from this login.
- **No spending authorization exists for programme H.** H0–H1 fit comfortably in local compute. Any cloud request
  will come with a measured estimate.
