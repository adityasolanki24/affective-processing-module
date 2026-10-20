# Programme K protocol: recovering from negative experience (frozen 2026-10-18, before any K-X / K5-X / K3b data)

**Branch:** `research/apm-v0.7-recovery`.
**Code:** `src/apm_transfer/{nerva_world,nerva_memory,infoseek}.py`, `experiments/K_recovery/scripts/`.
**Seeds:** `seeds.py` (new entropy).
**Cloud cap:** US$40, approved 2026-10-18.

## 1. The question

After bad experience NERVA keeps its distance, and its belief that an entity is disruptive never recovers when the
entity changes. Can it learn caution and then revise that caution on new evidence, without leaving the validated
safety envelope?

## 2. Diagnosis (K1, done before this protocol; `results/K1_information.json`)

Evidence per 4 s sensing window, in nats, from J3-T standing encounters (calm vs pushy). The two numbers in each IMU
column are KL(pushy‖calm), which governs learning caution, and KL(calm‖pushy), which governs recovering from it.

| distance | IMU detector | IMU graded (acc8) | ideal touch sensor (reference) | camera on the entity's lunges (extension) |
|---|---|---|---|---|
| 0.2 m | 0.58 / 0.55 | 2.63 / 0.93 | 0.59 | 2.94 / 0.74 at any distance in view |
| 0.4 m | 0.33 / 0.29 | 1.88 / 0.55 | 0.75 | |
| 0.6 m | 0.03 / 0.03 | 0.39 / **0.14** | 0.47 | |

**What is observable**, separated as follows:
- **Observable now:** contacts that move NERVA's IMU. A calm entity never reaches 0.6 m, but a pushy one's pushes
  there are weak and mostly below the robot's detection. That leaves about 0.14 nats per calm meeting, so a confident
  belief needs about 20 calm meetings at 0.6 m to return to 50%.
- **Simulator-only:** the entity's lunges and their reach.
- **Needs an added sensor:** seeing the lunges (a head camera). It works at any safe distance.
- **Needs unsafe interaction:** standing within reach (≤ 0.4 m) of a possibly pushy entity.

**Consequence:** with NERVA's current sensors, information gathered at a protected distance is weak but not zero.
A memory architecture cannot recover information the sensors do not deliver.

## 3. Mechanisms tested (K2; implemented and unit-tested before this protocol)

- **Protected observation.** The robot may stand for 1–3 sensing windows of 4 s each. Every window is a separate
  piece of evidence.
  - If its own detector fires and more windows were planned, it leaves (`abort_on_detect`). It never stays to
    collect repeated pushes.
- **Approach geometry.** Value of information may pick a closer stop, but only inside the risk cap.
- **Value of information** (`infoseek.voi`). It is **non-myopic** (knowledge-gradient*): the largest per-meeting
  reduction in the expected cost of the next task decision if the observation were repeated for m = 1, 2, 3 or 5
  meetings, times H = 5 future meetings. It uses only the robot's own learned models.
  - A myopic version was rejected *before any data*. When inspected on the fitted models, it gave every observation
    zero value once the belief was confident, which recreates the trap.
- **Observe out of reach** (`OBSERVE`, camera only). The robot stops 0.9 m from the entity: proximity 0 plus a 0.3 m
  stand-off, beyond every type's 0.8 m reach.
  - The task fails for that encounter, at retry cost; no perturbation is possible.
  - Only camera codes are used as evidence. IMU windows out of reach are discarded, never read as "safe".
- **Camera extension** (`nerva_world.Camera`). This is a declared sensor that NERVA does not have in simulation
  today:
  - 40° field of view, 2 m range, detection probability 0.9, reach noise σ 0.05 m;
  - it reads the entity's motion, never its type;
  - reported as an extension, never as a NERVA capability.
- **Task values** (`configs/K_tasks.json`), frozen before any K data from task requirements:
  - price 20 s per strong perturbation;
  - risk cap 0.30 expected strong perturbations per encounter;
  - retry 12.6 s (the robot's own mean approach time);
  - handoff needs a stopping distance ≤ 0.3 m.

## 4. K3 recovery study (confirmatory)

**World.** J5's closed-loop NERVA encounter:
- validated neutral controller, H's admitted interface, safety monitor;
- efference-copy gate, sensing pause.

**Lifetime.**
- 40 encounters; B1 with probability 0.5, otherwise A1 or C1 (always calm).
- Histories:
  - risky_B (pushy);
  - safe (calm);
  - change_B (pushy → calm at encounter 16);
  - late_B (calm → pushy at 16).
- Task: **handoff**.

**Systems** (`k_run.py`, same seeds for all):

| # | system | role |
|---|---|---|
| 1 | L0 | H's fixed mapping (J's memory, λ 0.02) |
| 2 | DT_STATIC | J's decision-theoretic mapping, task values frozen, no forgetting (λ 0) |
| 3 | DT | same, change-aware belief (λ 0.02, about the world's change rate): "dynamic revision, no active information" |
| 4 | DT_VOI | DT + safe information gathering (VOI), IMU only |
| 5 | DT_UNC | DT + uncertainty-triggered observation (3 windows when normalised entropy > 0.6), IMU only |
| 6 | ORACLE | DT acting on the true current type's canonical belief: evaluation reference, not deployable |
| 7 | DT_CAM | DT with the camera extension, no active information gathering |
| 8 | DT_CAM_VOI | DT_VOI with the camera extension |

**Type model:** `configs/K_types.json`.
- EM on K-T exploration (RANDINTENT, camera on), using training-world lifetimes only; K by BIC.
- Time model: J6's (`J6_model.json`).

**Test (K-X):** indices 0–49 → 8 systems × 4 histories × 50 = **1,600 lifetimes**.

**Primary criteria** (`k_analyze.py`; anchors from the type model; B1 meetings counted after the switch; horizon 14):

| | criterion |
|---|---|
| P1 recovery of belief | change_B: Kaplan–Meier median pushy → calm revision ≤ 6 B1 meetings |
| P2 recovery of behaviour | change_B: KM median ≤ 8 B1 meetings until B1 stopping distance is within 0.05 m of the same system's safe-history mean, for 2 meetings |
| P3 learning caution is kept | late_B: KM median calm → pushy revision ≤ 4 B1 meetings |
| P4 safety | 0 falls; safety trips ≤ 1% of encounters; repeated-push encounters (≥ 3 contacts, after the switch) not above DT's: paired 95% upper bound ≤ +0.02 in change_B and late_B |
| P5 task | arrival rate ≥ 0.99 |

**The recovery milestone is met** by a candidate (DT_VOI with IMU only; DT_CAM_VOI with the camera extension) only if
P1–P5 all pass for that candidate. Each candidate is judged separately.

**Expected on the diagnosis** (recorded before data): DT_VOI may fail P1, because the information is not there.
DT_CAM may already pass, because the camera removes the bottleneck. Such outcomes are reported as they are.

**Secondary (reported, not decisive):**
- belief correctness; excessive avoidance; informative observations; contacts and push size; task success;
  approach and encounter time; expressive differentiation;
- paired differences against DT;
- the gap to ORACLE;
- price sensitivity (10 and 40 s), run only if the budget allows.

## 5. K3b frozen-task confirmation of the learned mapping

The tasks in `configs/K_tasks.json` (waypoint, inspect, handoff, separation) are frozen as above.

**Systems:** L0 vs DT (λ 0.02, IMU). **Histories:** risky_B and safe. **Test:** 4 tasks × 2 systems × 2 histories × 40
lifetimes.

**Per task**, DT is promoted over L0 only if, for that task:
1. task cost (time + 20 s × strong perturbations + 12.6 s × task failure) is lower than L0's, paired 95% upper
   bound < 0;
2. contacts per encounter are not worse, with upper bound ≤ +0.05;
3. there are 0 falls.

**The learned mapping replaces L0** if this holds in at least 3 of 4 tasks.

Expressive differentiation (risky − safe stopping distance) is reported per task. It is **not** a criterion: a larger
difference is not better. For waypoint and separation, DT is expected to show none.

## 5b. K5 second embodiment (same protocol, a physically different robot)

**Robot:** the MuJoCo differential-drive robot (`wheeled_world`):
- its own wheel controller, independent safety supervisor (shove/tilt brake) and H7 expression adapter;
- its own acc8 edges, time model and type model (`configs/K_wheeled_*.json`), all fitted on its own K5-T
  exploration.

The memory and decision code are unchanged.

**Test:** K5-X, indices 0–49, systems L0, DT, DT_VOI, DT_CAM, DT_CAM_VOI and ORACLE × 4 histories, task handoff.

**Criteria:** P1–P5 as in §4, judged per candidate. Fall here means tipping over (> 60°).

## 5c. Pilot (partition P; debugging only; `results/pilot/`)

- **Runs:** NERVA, all 8 systems × 4 histories × 4 lifetimes; wheeled, 6 systems × 4 histories × 6 lifetimes.
- **Safety and task:** no falls, no trips; arrival 100%.
- **DT_VOI with IMU only** made the same choices as DT throughout. Its value-of-information rule never paid for
  itself with IMU-only evidence, consistent with §2.
- **No change** to systems, values or criteria followed the pilot.

## 6. Rules

- Pilot (partition P) only debugs. Any change after this protocol goes in `DEVIATIONS.md` with a reason.
- Test data are analysed once with the committed scripts.
- Falls, trips and every failure are reported.
- No simulator truth is used as a live observation; the oracle is evaluation-only.
