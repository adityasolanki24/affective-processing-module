# APM v1.0 release report

**Branch:** `release/apm-v1.0`, from programme L's `research/apm-v0.8-audit` at 84aeef9. Not merged.

**Scope.** Simulation only throughout: no hardware, no human participants.

**What is and is not claimed.**
- APM is a memory-and-decision runtime. Its beliefs are estimates of expected physical perturbation from specific
  people. They are not emotions. Nothing here supports claims about feelings or consciousness, and the learned
  latent types have no psychological meaning.
- How people interpret the robots' movement is untested. The perception study is prepared, not run.
- Safety results are simulation envelopes. They are not certification.

**Governance documents.**
- Protocol: [PROTOCOL.md](PROTOCOL.md), committed at f70d9aa before any V or X data.
- Deviations: [DEVIATIONS.md](DEVIATIONS.md), D1–D4.
- Milestones and gate evidence: [TRACKER.md](TRACKER.md).
- Raw records: one JSON line per lifetime, with every encounter's sensor codes, decision, risk, PhysX contact truth
  and the motion measured from the pose trace. Records are in the job archives (`runs/M/cloud/<job>/out`, not in
  git); per-decision runtime logs sit beside them. The analysed numbers are in [results/m6](results/m6).

## 1. What was built

1. **One integrated runtime** (`apm_transfer.runtime.Runtime`) drives every robot. It comprises:
   - sensor coding (IMU and rendered depth vision);
   - explicit, change-aware memory per identity;
   - the decision-theoretic expression mapping (greedy, no value of information, by default);
   - APM's risk supervisor;
   - persistence, deterministic replay, and AffectLink messages.

   Baselines are policies of the same runtime: lookup, monotone rule, fixed style, neutral, reactive and random.
2. **Isaac Sim integration** (`src/apm_isaac`; Isaac Sim 5.1.0, PhysX, RTX; Isaac Lab v2.3.2):
   - three robots:
     - **NVIDIA Nova Carter**: the stock asset with its wheel drives;
     - **Booster T1**: the stock USD with an Isaac Lab PPO velocity policy trained in this sprint
       (`policies/t1_isaaclab_v1`) and a frozen velocity map;
     - **NERVA** (Open Duck Mini v2): imported from its MJCF with MuJoCo-matched kinematics and inertials, upper-body
       colliders, its frozen neutral ONNX controller and its SafetyMonitor;
   - three scenes: warehouse, home, and a held-out hospital (stock Isaac environments);
   - people: animated stock characters (AnimationGraph) with kinematic capsule bodies, and a walking bystander per
     scene;
   - a rendered depth eye per robot, posed from the PhysX link pose every frame;
   - two safety layers: APM's risk supervisor and each robot's own physical supervisor.
3. **The Isaac Lab T1 task** (`apm_isaac/lab/t1.py`) and gate batteries for all three robots.
4. **Experiment tooling:**
   - the lifetime runner (`isaac/m5_lifetime.py`), with fault injection for F;
   - the NERVA MuJoCo runner (`scripts/m6_nerva_mujoco.py`);
   - fitting, calibration and analysis (`scripts/m5_fit.py`, `m6_tune.py`, `m6_analyze.py`, `m6_motion.py`,
     `m6_robustness.py`);
   - a capped, self-deleting GCP job queue.
5. **Four videos** composed from recorded demonstration lifetimes (§11).
6. **A blinded perception-study package** ([human_study/](human_study/)): protocol, stimulus generator and analysis.
   Not run.
7. **Release files:**
   - the asset manifest with SHA-256 (`configs/assets_manifest.json`);
   - the pinned Isaac environment (`cloud/isaac_env.sh`, `cloud/isaac_pip_freeze.txt`, `cloud/versions.txt`);
   - an integration test in CI;
   - README and docs.

## 2. What was fixed

**From programme L:**
- **Task-scoring artefact.** OBSERVE and HOLD no longer count as completing a movement task. `--legacy-task-scoring`
  reproduces K/L (M1, 433294a).
- **NERVA falls under clustered pushes (L's E3).** Characterised, not hidden. A push-reflex threshold sweep found
  no threshold that prevents the clustered-push falls. The envelope is reported: repeated 0.20 m/s pushes at rest
  cause 5/10 falls in both MuJoCo and Isaac. People's lunges are therefore capped at 0.15 m/s, frozen before any
  data (TRACKER M3).
- **GRU question closed.** On held-out exploration lifetimes, the explicit memory predicts contacts better than a
  GRU trained on the same data ([gru_nova_carter.json](results/m6/gru_nova_carter.json),
  [gru_humanoid.json](results/m6/gru_humanoid.json)):

  | robot | log loss, explicit memory | log loss, GRU | AUC, explicit memory | AUC, GRU | GRU − memory log loss, 95% CI |
  |---|---|---|---|---|---|
  | Nova Carter | 0.515 | 0.809 | 0.77 | 0.72 | [0.14, 0.44] |
  | T1 | 0.455 | 1.127 | 0.84 | 0.72 | [0.48, 0.89] |

- **Humanoid vision false positives.** Not fixed; characterised (§9).

**Found and fixed in this sprint** (details in the commit history):
- **NERVA import:** mass, gains (57× too stiff), joint axes and backlash joints.
- **NERVA body collision:** its MJCF collides only at the feet (D2).
- **People:** T-pose and positioning through the AnimationGraph.
- **Eyes:** cameras parented to links stay at the spawn pose in PhysX; the eyes are now posed from PhysX every
  frame.
- **Perception:** egomotion rejection removed lunges by articulated people, fixed by `articulated_target`.
- **T1:** velocity deadband (velocity map anchored at 0).
- **Nova Carter:** placement and depenetration spikes.
- **Infrastructure:**
  - process hangs and crash handling;
  - `/` in system names;
  - an out-of-memory failure in the video panel;
  - the anonymous-track KeyError (D4).

## 3. Which robots work, and where

| robot | Isaac Sim: warehouse | Isaac Sim: home | Isaac Sim: hospital (held out) | comparative experiments run on |
|---|---|---|---|---|
| Nova Carter | works | works | works | Isaac Sim (A–F) |
| Booster T1 | works | works | works | Isaac Sim (A–F) |
| NERVA | walks, perceives, survives most contacts (96 demonstration encounters, 4 contacts, 0 falls) | **fails**: deterministic falls, steps in place on the floor | walks (demonstration lifetimes) | **MuJoCo 3.3.0** (A, B, C, E; D3) |

**Gate batteries (M3, Isaac Sim):** all three pass ([results/m3](results/m3)).

| robot | nominal conditions | pushes |
|---|---|---|
| NERVA | 0 falls | 0 falls up to 0.15 m/s |
| T1 | 0 falls | 0 falls up to 0.3 m/s; falls under 0.5–0.8 m/s pushes while turning |
| Nova Carter | speed error 0.002 m/s, turn error 0.023 rad/s, stops in 0.10 m | supervisor trips on a 0.6 m/s shove, no fall |

**Cross-physics characterisation per robot:**

| robot | MuJoCo vs Isaac (PhysX) |
|---|---|
| NERVA | Same controller and battery: rest tilt 10.9° vs 10.5°; forward speed 0.072 vs 0.051 m/s; turns +0.65/−0.56 vs +0.64/−0.61 rad/s; repeated 0.20 m/s pushes at rest 5/10 vs 5/10 falls. |
| T1 | Kinematics match to 0.07 mm, but the MuJoCo-trained policy falls in PhysX within 0.6 s, and even a passive PD stance topples. The Isaac controller is therefore a new Isaac Lab policy; T1 has no shared controller across engines. |
| Nova Carter | Isaac-only validation. No MuJoCo model exists. |
| Custom wheeled robot | Cross-physics reference only, not in videos. MuJoCo: speed error 0.001 m/s, turn error 0.024 rad/s, stop 0.067 m ([wheeled_mujoco.json](results/m3/wheeled_mujoco.json)). |

## 4. Key results: preregistered confirmatory tests (partition X)

**Design.**
- 624 test lifetimes of 24 encounters each:
  - Nova Carter: 232 lifetimes (Isaac);
  - T1: 232 (Isaac);
  - NERVA: 160 (MuJoCo, D3).
- Paired by seed index. One-sided sign-flip permutation test, Holm within robot, 95% bootstrap CIs. The unit is the
  lifetime.
- **Measures:**
  - A1, A2, B1: lunge contacts per encounter;
  - B2: success rate;
  - C, D: realised cost in seconds, i.e. arrival time + 20 s per contact + retry time if the task fails.

**Differences, DT − comparator** (negative is better except B2) ([analysis.json](results/m6/analysis.json)):

| | vs | Nova Carter (Isaac) | T1 (Isaac) | NERVA (MuJoCo) |
|---|---|---|---|---|
| A1 experience changes behaviour | NOMEM | −0.60 [−1.07, −0.20], p 0.030, *n.s. after Holm* | **−1.47 [−2.03, −0.88], p 0.004** | −1.70 [−3.30, −0.50], p 0.017, *n.s. after Holm* |
| A2 identity-specific memory | SHUFFLED | −0.52 [−0.98, −0.08], p 0.047, *n.s.* | −0.57 [−1.08, −0.07], p 0.061, *n.s.* | −1.10 [−2.80, 0.00], p 0.125, *n.s.* |
| B1 adapts calm → pushy | FROZEN | **−0.85 [−1.22, −0.50], p 0.005** | **−1.10 [−1.58, −0.60], p 0.009** | **−2.41 [−3.01, −1.75], p 0.002** |
| B2 adapts pushy → calm (success) | FROZEN | **+0.70 [0.55, 0.83], p 0.002** | **+0.57 [0.42, 0.74], p 0.002** | +0.25 [0.06, 0.50], p 0.060, *n.s.* |
| C1 vs fixed style | FIXED | **−12.9 s [−15.2, −10.8]** | **−13.5 s [−14.7, −12.2]** | **−20.1 s [−22.6, −17.7]** |
| C2 vs neutral motion | NEUTRAL | **−2.3 s [−4.0, −0.8]** | **−4.6 s [−11.2, −0.6]** | **−6.0 s [−8.7, −3.5]** |
| C3 vs fixed lookup | LOOKUP | **−14.0 s [−15.2, −12.8]** | **−21.2 s [−22.4, −20.2]** | **−11.9 s [−13.2, −10.6]** |
| C4 vs calibrated simple rule | MONOTONE* | **−4.9 s [−6.1, −3.6]** | **−5.9 s [−7.3, −4.7]** | **−3.1 s [−4.1, −2.1]** |
| D1 held-out hospital | REACTIVE | **−10.2 s [−12.8, −7.7]** | **−8.3 s [−9.8, −7.0]** | not run (D3) |
| D2 held-out hospital | MONOTONE* | **−3.9 s [−5.4, −2.5]** | **−5.9 s [−7.2, −4.6]** | not run (D3) |
| D3 home | REACTIVE | **−8.5 s [−11.0, −5.9]** | **−6.5 s [−9.9, −1.9]** | not run (D3) |

How to read the table:
- **Bold** means the hypothesis passes after Holm correction.
- C1–C4 and D1–D3 have p ≤ 0.009 throughout.
- n = 10 pairs for A, 9 for B (one lifetime per robot lacks enough post-switch B1 meetings, e.g. because a fall
  ended it), 20 for C and 12 for D.

**Mean realised cost, warehouse, in seconds:**

| robot | DT | FROZEN | SHUFFLED | NEUTRAL / NOMEM | MONOTONE* | FIXED | LOOKUP |
|---|---|---|---|---|---|---|---|
| Nova Carter | 17.6 | 20.3 | 19.5 | 19.9 | 22.5 | 30.5 | 31.6 |
| T1 | 16.6 | 18.5 | 18.5 | 21.2 | 22.5 | 30.1 | 37.8 |
| NERVA (MuJoCo) | 15.1 | 20.0 | 18.1 | 21.1 | 18.2 | 35.2 | 27.1 |

**Hospital and home, in seconds (DT / MONOTONE* / REACTIVE):**

| robot | hospital | home |
|---|---|---|
| Nova Carter | 19.3 / 23.2 / 29.6 | 21.2 / 23.2 / 29.7 |
| T1 | 18.7 / 24.6 / 27.1 | 21.4 / 25.5 / 27.9 |

**Verdicts (§7 rule: all 3 robots = Demonstrated, 1–2 = Partially, 0 = Not):**

| verdict | hypotheses |
|---|---|
| Demonstrated | B1, C1, C2, C3, C4 |
| Partially demonstrated | A1 (T1 only); B2 (Nova Carter, T1); D1, D2, D3 (Nova Carter, T1; NERVA has no Isaac comparative data) |
| Not demonstrated | A2 (identity-specific memory) |
| Failed (significant in the wrong direction) | none |

**E (number of robots passing):** A1 1, B1 3, B2 2, C2 3, D1 2.

**Caveats that change the reading:**
- **NOMEM is physically identical to NEUTRAL.** In all 20 Nova Carter lifetimes, without memory the runtime chose
  the neutral approach every time, and the same holds in the summary statistics of the other robots. A1 is
  therefore "DT versus neutral motion in the A window".
- **A2 was not demonstrated on any robot.** The point estimates favour identity-specific memory, but the paired
  difference against SHUFFLED is not significant. The data do not show that memory must be attached to the right
  person, rather than being general caution learned from the population.
- **The C and D effects are large partly because of the comparators.**
  - FIXED and LOOKUP are poor (they hold back or keep their distance from calm people too).
  - The most informative comparison is C4: DT against a simple rule calibrated on partition V, which DT beats by
    3–6 s per encounter on every robot.
- **A fall ends a lifetime.** The cost measure averages only the encounters that took place; it does not price
  the fall itself (§6).

## 5. Expressive movement (Q6)

Measured from each robot's own pose trace, on all X approaches of all systems ([motion.json](results/m6/motion.json)).
Correlations are Spearman.

| | Nova Carter | T1 | NERVA (MuJoCo) |
|---|---|---|---|
| approaches | 5,283 | 4,972 | 3,659 |
| distinct executed intents | 345 | 114 | 228 |
| graded proximity (strictly between grid extremes) | 83% | 86% | 88% |
| proximity vs final distance | −0.75 | −0.80 | −0.75 |
| tempo vs mean speed | +0.70 | +0.65 | not measured (MuJoCo record has no pose trace) |
| tempo vs arrival time | −0.71 | −0.57 | −0.57 |
| hesitation vs pauses | +0.86 | +0.99 | not measured |

Nova Carter's stopping distance falls in steps with proximity (median abstract distance per proximity bin):

| proximity bin | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1 |
|---|---|---|---|---|---|
| median abstract stopping distance | 0.59 m | 0.45 m | 0.38 m | 0.25 m | 0.16 m |

Caveats:
- On Nova Carter the physical "final distance to the person" feature has a very narrow interquartile range
  (0.508–0.510 m), while the abstract stopping distance from odometry is clearly graded. The physical feature
  probably saturates (measured to the station rather than the person's body), so the abstract distance is the
  better evidence.
- On T1, hesitation is realised as an explicit stop, so the near-perfect hesitation–pause correlation shows that
  the command is executed, not that it is subtle.

## 6. Safety

**Falls in all test lifetimes, all systems** (each fall is listed in `analysis.json` → `safety`):

| robot | lifetimes | encounters | falls | physical-supervisor trips | lunge contacts |
|---|---|---|---|---|---|
| Nova Carter | 232 | 5,562 | 1 (SHUFFLED, late_B 3, encounter 17) | 1,633 | 2,204 |
| T1 | 232 | 5,363 | 16 | 13 | 2,129 |
| NERVA (MuJoCo) | 160 | 3,840 | 0 | 0 | 1,680 |

**T1 falls by system:**

| system | falls |
|---|---|
| DT | 4: home 2, hospital 1, warehouse 1 |
| NEUTRAL | 3 |
| NOMEM | 3 |
| FROZEN | 2 |
| MONOTONE* | 2 |
| REACTIVE | 1 |
| SHUFFLED | 1 |

All 16 occurred with the pushing person (B1). Several are at the same seed and encounter across systems (late_B 9,
encounter 16 in five systems), so they come from the lunge stream rather than from APM's choice. T1 is the robot
that falls; its IMU contrast is weak (31 kg body, 0.15 m/s contacts).

**Risk cap:** 0.30. The APM systems (DT, FROZEN, SHUFFLED, NOMEM, all supervised) executed no approach above it.
Their largest executed predicted risk was 0.300, 0.296, 0.300 and 0.268. The unsupervised baselines exceed the cap
by design.

**Other:**
- Nova Carter's physical supervisor tripped 1,633 times (shoves above 0.35 m/s or tilt above 25°). Each trip stops
  the robot. This is the second safety layer working, not a failure.
- No claim of production safety, and nothing here is hardware certification.

## 7. Robustness (F; Isaac, warehouse, DT, change_B, seeds X40–X41, X50)

Source: [robustness.json](results/m6/robustness.json).

| check | Nova Carter | T1 | §8 pass condition | result |
|---|---|---|---|---|
| restart after encounter 12 (save, new process, load) | decisions and observations identical to the uninterrupted run (2 lifetimes) | identical (2) | identical | pass |
| replay, all X DT logs | 480 decisions, 0 mismatches | 473, 0 mismatches | 0 mismatches | pass (tolerance 1e-9; 255 / 350 bit-exact, the rest differ in the last float digit between Linux and Windows) |
| replay, F long, reference and no-camera logs | 196 decisions, 0 mismatches | 196, 0 mismatches | — | pass |
| camera off (IMU only) | 0 falls, 48/48 arrived | 0 falls, 48/48 | no falls, no violations | pass (2 lifetimes, not 4, D4) |
| camera dropout 0.3 | 0 falls | 0 falls | no falls | pass (2 lifetimes, D4) |
| controller interruption (2 s) | 0 falls, 48/48 completed | 0 falls, 48/48 | no fall; completes | pass |
| safety veto (forced over-cap decisions) | 12 vetoes; max executed risk 0.288 | 19 vetoes; 0.268 | vetoed, not executed | pass |
| scripted 0.15 m/s contact during the approach | 0 falls | 0 falls | IMU detection, no fall | **no fall: pass. IMU detection: not demonstrated.** On 28 / 23 paired contact-free approaches the IMU peak is indistinguishable from the reference run (median 0.196 vs 0.200; 0.139 vs 0.141). |
| repeated 0.15 m/s perturbations | 0 falls, 18 trips | 0 falls | no fall | pass |
| identity uncertainty (anonymous B1 track) | 29 B1 meetings, one belief before every meeting (the kind prior) | 29 meetings, kind prior only | kind prior; no B1 memory | pass |
| long operation (100 encounters) | 0 falls, 100/100 arrived, replay identical | 0 falls, replay identical | no fall; replay identical | pass (memory size not separately measured) |

NERVA was not in F (D3).

## 8. Perception

The depth eye check measured a scripted 0.25 m lunge, whose true abstract reach is 0.40:

| robot | reach estimate |
|---|---|
| Nova Carter | 0.36 |
| T1 | 0.46 |
| NERVA | 0.33 |

There is no simulator truth in any `Observation`. PhysX contacts are evaluation-only.

## 9. Humanoid vision false positives (L's open item)

Camera-positive rate, all X approaches:

| robot | calm person, no contact | pushy person, no contact | encounters with a contact |
|---|---|---|---|
| Nova Carter | 8% | 29% | 57% |
| T1 | **21%** | 54% | 78% |
| NERVA (MuJoCo) | 9% | 31% | 58% |

T1's false-positive rate is about 2.5 times the others'. The fitted likelihoods absorb it (T1's learning results
above hold despite it), but the underlying cause is not fixed. It is consistent with the camera's own walking
motion. It remains open.

## 10. Reproduction

- **Install and CI:** see the [README](../../README.md). `python experiments/M_release/scripts/m1_integration.py`
  runs the end-to-end check on MuJoCo.
- **Isaac:**
  - `experiments/M_release/cloud/isaac_env.sh` builds the pinned environment: Isaac Sim 5.1.0, Isaac Lab v2.3.2,
    NVIDIA driver 580, L4 GPU. The full `pip freeze` and versions are in `cloud/isaac_pip_freeze.txt` and
    `cloud/versions.txt`.
  - `cloud/m6_plan.py --stage V|X|D|F` prints the exact job command for every run in this report.
- **Seeds:** [seeds.py](seeds.py) (entropy 0xA9_0010_2026_1019; partitions P/T/V/X/D).
- **Analysis:**
  `scripts/m6_analyze.py --runs <X dirs> --tasks nova_carter=configs/M5_nova_carter_tasks.json humanoid=configs/M5_humanoid_tasks.json nerva_mujoco=configs/M5_nerva_mujoco_tasks.json --out results/m6/analysis.json`,
  then `m6_motion.py` and `m6_robustness.py`.
- **Not yet independently reproduced.** Every number here was produced once, by this sprint, on our VMs. The raw
  lifetime archives (several GB, video archives about 13 GB) are in the project bucket, not in git.

## 11. Videos

All footage is recorded Isaac Sim 5.1 / PhysX demonstration lifetimes (partition D, DT system). It contains no
artificial motion, and every frame carries an engine label. NERVA's footage is Isaac Sim (warehouse and hospital),
although its comparative numbers come from MuJoCo.

<!-- VIDEOS: filled when the build finishes -->

## 12. Deviations

See [DEVIATIONS.md](DEVIATIONS.md):

| entry | summary |
|---|---|
| D1 | Exploration ran short; models fitted on the completed lifetimes. |
| D2 | NERVA body colliders; NERVA exploration redone. |
| D3 + correction | NERVA's comparative data come from MuJoCo. |
| D4 | Two T1 LOOKUP lifetimes completed after a timeout; anonymous-track fix; F sizes; how contact detection was assessed. |

No threshold, seed, reward or criterion was changed after its data were seen.

## 13. Failures and limitations

1. **NERVA does not run APM's comparative experiments in Isaac Sim.**
   - It cannot walk in the home scene.
   - Its Isaac exploration data were contact-free before body colliders were added (D2), so it has no valid Isaac
     type model.
   - Its A–C results are MuJoCo results. D was not run for NERVA.
2. **A2 not demonstrated.** Identity-specific memory is not shown over a shuffled-identity control on any robot.
3. **A1 passes only on T1, and B2 not on NERVA,** after Holm correction.
4. **T1 falls:** 16 in 5,363 encounters, all with the pushing person, across all systems including DT (4).
5. **T1's camera false positives** stay high (21% with calm people). T1's IMU contrast is weak.
6. **The scripted-contact IMU detection check is not demonstrated.** The coded IMU evidence does not register a
   0.15 m/s push during the approach.
7. **Cost measure:**
   - A fall ends a lifetime and is not priced.
   - The cost weights (20 s per contact, retry time) are design choices, frozen before the data.
8. **NOMEM equals NEUTRAL physically,** so A1 is a comparison against neutral motion.
9. **Nova Carter:** validation is Isaac-only (no MuJoCo model).
10. **T1 has no shared controller across engines.** Its MuJoCo policy fails in PhysX, and the Isaac policy is new.
11. **F checks ran 2 lifetimes each** (§8 asked for 4 in two checks) and no NERVA F runs.
12. **No human study.** The interpretation of expression by people is untested. No hardware.
13. **Not independently reproduced.**

## 14. Q1–Q10

| # | question | answer | evidence |
|---|---|---|---|
| Q1 | Does APM genuinely learn from experience? | **Demonstrated** (in simulation) | B1 passes on all three robots: after a calm person turns pushy, DT has fewer contacts than the same runtime with frozen beliefs. A1 (DT vs no memory) passes on T1, with the right direction and p < 0.05 uncorrected on the others. The explicit memory predicts held-out contacts better than a GRU. §4, §2. |
| Q2 | Do those experiences change behaviour in matched present circumstances? | **Partially demonstrated** | Paired lifetimes have the same seeds and lunge streams, so the only difference is the system. Behaviour differs and the outcomes differ (A1, B1, B2). But identity-specific memory (A2: right person vs shuffled) is not significant on any robot. §4. |
| Q3 | Does changing behaviour change subsequent outcomes? | **Demonstrated** | Contacts per encounter in the A window: 0.55 vs 1.15 (Nova Carter), 0.10 vs 1.57 (T1), 0.50 vs 2.20 (NERVA), DT vs no memory. B1 contacts and B2 success differ from FROZEN. Costs are lower in C. §4. |
| Q4 | Can it revise outdated beliefs and recover behaviour? | **Demonstrated** (calm → pushy, all three robots) / **Partially demonstrated** (pushy → calm: Nova Carter and T1; not NERVA after Holm) | B1, B2. §4. |
| Q5 | Does it improve task performance or safety against competent baselines? | **Demonstrated for task cost; not demonstrated for falls** | C1–C4 on all three robots, including the simple rule calibrated on V (C4: 3–6 s per encounter) and neutral motion (C2). D1–D3 on two robots in a held-out scene. Falls: T1 DT 4 vs 1–3 for the baselines; no safety advantage is shown. §4, §6. |
| Q6 | Does it produce physically measurable expressive movement beyond binary avoidance? | **Demonstrated** (physical measurement); **not demonstrated** (meaning to people) | 114–345 distinct executed intents; 83–88% graded proximity; proximity, tempo and hesitation each map monotonically onto measured stopping distance, speed or arrival time, and pauses. §5. |
| Q7 | Can the same core run across NERVA, a humanoid and a wheeled robot? | **Demonstrated** (same runtime on all three); **partially** in Isaac Sim (NERVA's comparative data are MuJoCo) | One runtime class and the same systems and analysis on all three. Nova Carter and T1 in Isaac; NERVA in Isaac for gates, perception and demonstrations; its experiments in MuJoCo. §3. |
| Q8 | Does it work in realistic, physically interactive simulation environments? | **Partially demonstrated** | Nova Carter and T1: warehouse, home and held-out hospital with animated, physically present people and a walking bystander; D1–D3 pass. NERVA fails the home scene. These are stock Isaac scenes with scripted lunging people, not human behaviour models. §3, §4. |
| Q9 | Can someone independently reproduce the results? | **Partially demonstrated** | Pinned environment, frozen seeds, job commands, raw records and deterministic replay (0 mismatches). Not yet reproduced by anyone else; raw archives are in a private bucket; Isaac needs an RTX GPU. §10. |
| Q10 | What remains unsupported? | — | See the list below. |

**Q10: what remains unsupported.**
- Human emotional interpretation of any movement: the study is not run.
- Any claim of emotion, feeling or consciousness.
- Hardware transfer: no robot hardware was used.
- Production safety.
- Identity-specific memory (A2).
- NERVA's experiments in Isaac Sim and in the home scene.
- T1's fall rate under pushing.
- IMU detection of light unexpected contacts.
- Behaviour with real people (the people here are scripted).
- Independent reproduction.

## 15. Cloud spend and cleanup

<!-- COST: filled at teardown -->

## 16. QA checklist

| item | status |
|---|---|
| Protocol committed before V/X data (f70d9aa) | yes |
| MONOTONE* calibrated on V and committed before each robot's X runs (b19b0c2, af5fbe1, 6dc9f1e) | yes |
| X partition only in the confirmatory analysis; T/V/X seeds disjoint (`seeds.py`) | yes |
| Held-out hospital unused for fitting and tuning | yes (T: warehouse and home; V: warehouse) |
| No simulator truth in observations (PhysX contacts are evaluation-only) | yes (`apm_isaac/body.py`, `world.py`) |
| Every deviation logged | D1–D4 |
| Raw traces per encounter, per-decision logs, replay | yes |
| Negative results reported | A2, A1 (2 robots), B2 (NERVA), contact IMU, T1 falls, NERVA home |
| Neutral controllers preserved; expression through adapters only; two safety layers | yes |
| Videos authentic and engine-labelled; mixed backends labelled | yes (all video footage is Isaac Sim) |
| No project IDs, bucket names or personal paths committed | checked |
| ruff, mypy (CI numpy and local), pytest | see the commit at release |
| Cloud resources deleted | see §15 |

## 17. Recommendation

Freeze this as **APM v1.0 (research release, simulation only)**, with the scope stated above. In particular:
- NERVA's comparative results are MuJoCo results;
- A2 is not demonstrated.

**The case for freezing.** The core claims are preregistered and tested on fresh seeds:
- learning changes behaviour;
- revision after a change;
- better task cost than competent baselines, including a calibrated simple rule;
- graded, physically measured expression;
- one runtime across three bodies.

They hold on two robots in Isaac Sim and largely on NERVA in MuJoCo.

**Not to be described as shown:**
- identity-specific memory;
- NERVA in realistic Isaac scenes;
- T1 safety under pushing;
- anything involving human interpretation or hardware.

**Next steps, in order:**
1. NERVA exploration, calibration and test in Isaac (warehouse and hospital).
2. A test of identity specificity with more power.
3. T1 push robustness.
4. The human perception study, after ethics approval.
