# APM v1.0 release: milestone tracker

## Corrective continuation — 19 October 2026

- Branch: `codex/apm-v1-remediation`, base `c150288`.
- **First delivery:** improved T1 walking comparison and unlabeled expressive-motion
  comparison. User visual acceptance gates final validation and packaging.
- Cloud authorization: user explicitly confirmed the same US$500 total cap in this
  chat. Reserve US$185 for previous work pending billing reconciliation. Initial
  corrective compute allocation: at most US$12, one L4 worker, hard runtime 2 h;
  no training until video/trace diagnosis and target definition.
- Read-only cloud inventory found no VMs, disks or routers in the configured project.
- Existing T1 footage sampled at 30–36 s: full feet are clipped in several frames;
  the lower overlay obscures motion. Candidate clips require full-body framing.
- Confirmed code defects: double-argsort ranks mishandle ties; motion distance uses
  root-to-person-centre rather than the separately recorded physical stopping gap.
- Candidate under inspection: official Booster T1 policy, pinned at
  `da396a06d6eed99e2de72d7749c48ee8748950f9`, Apache-2.0. PhysX transfer unvalidated.
- Status: diagnosis and pilot tooling in progress. Neither requested video exists
  yet; no improved gait or visible-expression success is claimed.
- Physical pilot `GAIT_PILOT2`: both controllers completed 24 s without falls.
  Candidate foot clearance increased, but the initial speed summary mistakenly
  counted sway as progress. Correct body-forward speed was only 0.041 m/s against
  0.200 requested. Not promoted. The 0.78 command-scale trial also failed progress.
  A 2.0 scale reached 0.166 m/s; the next short pilot checks 2.25 plus smoother motor
  targets. All attempts and traces are retained. Camera framing now includes feet/head.
- `EXPRESSION_PILOT`: exploratory physical histories, then a common conservative
  inspection task, with richer designed mapping only after six encounters.
  NERVA's Isaac model has a constant zero perturbation readout; confidence in its
  learned belief can still modulate inspection, but this does not establish a
  useful learned risk estimate on NERVA. Camera false positives remain a concern.
- One worker created with a 7200 s max runtime and 10 min idle deletion. Existing
  queue bootstrap's unavailable NumPy pin was observed; Isaac's restored venv
  works independently. Fix the bootstrap before any fresh-install claim.
- A second L4 worker has a 1800 s hard cap for gait checks; first zone stockout,
  second zone succeeded. The combined compute allocation remains US$12.
- Ten focused regressions pass, including tied ranks, separating sway from forward
  progress, controller observation order, physical distance and matched expression.
- The official simulation PD gains resolved the transfer defect: pilot seed 71001
  stance slip 0.0511 -> 0.0149 m/s; joint acceleration RMS 65.2 -> 28.1 rad/s2;
  tilt p95 7.65 -> 4.19 degrees; body-forward speed 0.183 -> 0.183 m/s. No fall
  on seeds 71001–71003 (71002 includes two 0.3 m/s disturbances). These are short
  pilots, not the final stability battery or evidence of human preference.
- First review video exists: `runs/remediation/review/01_T1_walking_comparison.mp4`.
- Initial richer expression remains weak between physical histories. A documented
  alternative now learns camera-event expectations per identity; motion mapping
  remains designed. Original and alternative pilots are kept separate.
- Camera-memory review: all six present-task pilot runs completed without falls.
  T1 head-yaw span differs (0.405 vs 0.201 rad); NERVA's spans remain essentially
  identical (0.465 vs 0.470 rad), despite a clear difference from neutral (0.049).
  **NERVA history differentiation is still unresolved.** No expression success or
  visual acceptance is claimed. Physical history-pair stopping gaps differ by
  <1 cm; neutral controls differ by up to about 7 cm, so call them command-matched,
  not exact realised-distance controls.
- Framing QA: close-up T1 head was clipped. A capped 30 min worker re-renders
  only its three views with a wider lens. The earlier footage remains intact.

### Continued (Claude), after the user's review of the two gate videos (2026-10-19)

The user's verdict: the T1 candidate is promising but is a candidate only (run full validation); expression is
not acceptable yet, and NERVA's near-identical histories are a failed gate.

**T1 gait battery** (protocol frozen before data; deviation G1 = start-point search):
- design: 260 matched pairs, 4 environments x 13 scenarios x 5 fresh seeds;
- gait quality: the candidate is far better (stance slip -69%, joint acceleration -57%, tilt p95 -48%, torque
  proxy -40%, foot clearance x3.2, push peak tilt -41%, stop distance -30%);
- failures: 3 falls at 0.5 m/s walking pushes (home, hospital, warehouse; 1/5 each) against v1's 0, and settle
  time +42%;
- verdict: **not promoted under the frozen rule; v1 remains the controller**
  (`results/remediation/gait_battery_v1_vs_candidate.json`).
- Cause: after a strong push the official policy develops a growing oscillation, and the supervisor trips only once
  the robot is already falling.
- Next iteration: the official weights fine-tuned in Isaac Lab (`lab/t1_booster_ft.py`, Booster's exact
  observation, action and motor conventions; stronger pushes; stand-still term).
  - run 1 diverged (NaN at iteration ~145);
  - run 2 adds explosion guards.
  - A fine-tuned policy must pass the same frozen rule on fresh seeds.

**Expression v2** (`experience_state.py`, `expressive_program.py`, `posture.py`, `isaac/expression_v2.py`):
- **Learned, label-free experience memory:**
  - disturbance: unintended self-displacement while commanded to stand;
  - camera evidence: looming;
  - thresholds from a quiet-period calibration.
- **Designed multi-phase program:**
  - observe: watching time and gaze;
  - approach: path shape and gaze;
  - interact: guarded vs open posture, side-on torso;
  - depart: backing away facing the person vs turning off.
- **Body adapters** use only joints the locomotion policy does not control; posture signs and stability were
  verified with stills.
- **T1 findings:**
  - the learned states separate the histories clearly (wariness 0.17 vs 0.73);
  - the candidate controller fell in the pushy history (meeting 5), the same push weakness the battery found.
- **NERVA findings:**
  - contact topples it, so its pushy person makes near-miss lunges instead;
  - its reach estimates are biased at close range, so looming replaces them.
  - Pilot 5 is running.
- Fixed along the way:
  - head motion during sensing blinded the camera (the head now holds still on the person while sensing);
  - teleport transients inflated the calibration;
  - CRLF in shipped scripts (jobs now strip it).

**Corrected comparisons** (protocol frozen before data; fresh R/Y seeds; every system under the same supervisor):
- MONOTONE* re-tuned under supervision: NERVA lo=0.1/width=0.4 (M6_tuned_R.json).
- NERVA (MuJoCo), 0 falls in every system (`results/remediation/corrected_nerva_mujoco.json`):
  - DT beats MONOTONE* by only 0.67 s (M: 3.1 s), and FIXED, LOOKUP and NOMEM;
  - **DT does not beat neutral motion + supervisor (+0.78 s)**: on NERVA, M's C2 advantage came largely from
    memory-driven safety vetoes (218 vetoes), not from DT's expressive choices;
  - **identity-specific memory supported**: DT beats POOLED (-2.5 s), SWAPPED (-15.0 s) and SHUFFLED (-7.9 s),
    with fewer contacts from the pushy person than POOLED (-0.19 per encounter) and lower cost with calm people.
- Nova Carter: re-tune finishing; RY needs about 10 GPU-hours (L4 stock is scarce; spot VMs used where possible).
- T1: waits for the controller decision.

**Security audit, commit 75c3b86:**
- what it exposes: a `gcloud compute instances list` dump in a public repo, still reachable by SHA. It shows the
  project ID, the runner service account's email, the bucket name (VM metadata), internal 10.x IPs and the
  startup script;
- what it does not expose: keys, tokens or passwords;
- risk is low: the bucket has no public access and uniform bucket-level access is on;
- options (each needs the user's authorisation):
  - ask GitHub Support to purge the unreachable commit;
  - enforce public-access prevention on the bucket.
- No rotation is needed.

### Final remediation pass (Claude, 2026-10-19/20). The account is `REMEDIATION_REPORT.md`

- **T1:** v1 kept (D7). The push-reflex variants fail even on development seeds; fine-tuning stopped.
- **D6, a person-model defect** (Carter ejections) plus two defects in its first fixes, all found by
  verification runs. Every affected job was discarded and rerun (tags *7_*, EXPR10/11, DEMO10/11).
- **Expression** (D8 changes, NERVA validated on fresh seed 73302): all three robots move differently after
  different histories. The T1 and Carter demos show learning; T1 also shows partial recovery.
- **Corrected comparisons:** NERVA (MuJoCo) and Nova Carter done; T1 finishing. DT is never better than
  neutral + supervisor. Identity-specific memory is supported on NERVA and Carter.
- **Env snapshot** restored from soft delete; setup hardened.

---

This is the only tracker for the final sprint. The final account is `REPORT.md`. Confirmatory tests are frozen in
`PROTOCOL.md` before their data exist.

**Branch:** `release/apm-v1.0`, from `research/apm-v0.8-audit` at 84aeef9. Nothing is merged without approval.

**Cloud cap:** US$500, approved 2026-10-18. Spending is logged in the cost section of `REPORT.md`.

**Pinned stack:**
- Isaac Sim 5.1.0 (pip wheel, Python 3.11, Ubuntu 22.04);
- Isaac Lab v2.3.2;
- GCP L4 (Ada, RT cores, 24 GB);
- NVIDIA driver 580 (Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580`).

Not chosen:
- Isaac Sim 6.x: Isaac Lab's stable branch still targets 5.1.
- Isaac Lab 3.0: early access.
- A100 and H100: they lack RT cores.

| # | milestone | gate (must pass before the next safety-critical step) | status |
|---|---|---|---|
| M1 | freeze and stabilise v0.8 | clean integration test (runtime → sensors → memory → decision → bounded command → safety veto → motion → persist/restart → replay → AffectLink) passes from a clean checkout; L's open defects fixed (task-scoring artefact, falls characterised) | done (433294a) |
| M2 | Isaac Sim infrastructure | pinned env restores on a fresh VM; headless render of a stock scene; physics/sensor smoke test | done |
| M3 | three embodiments in Isaac | each robot imported, controller validated by a battery (stand, walk, stop, turn, pushes); MuJoCo vs Isaac characterised | gates passed (see below) |
| M4 | perception and interaction | camera/IMU/odometry → perception → `Observation`, no simulator truth; moving agents with physical presence; ≥ 2 functional scenes | done (see below) |
| M5 | integrated APM | the same runtime drives all three robots in Isaac; body-specific expressive realisation | done: models fitted (T), MONOTONE* calibrated (V) per robot; NERVA comparative on MuJoCo (D3) |
| M6 | final evaluation | `PROTOCOL.md` frozen; experiments A–F on fresh seeds; held-out scene | done: 624 X lifetimes, F checks; `REPORT.md` §4–§7 |
| M7 | videos and packaging | 4 videos from recorded traces; trace files, seeds, commands | done: `REPORT.md` §11 |
| M8 | QA and release | checklist verified; cloud audit; docs; release report | done: `REPORT.md` §15–§17 |

## M3 gate evidence (Isaac Sim 5.1 / PhysX unless stated)

**Robots.** NERVA (Open Duck Mini v2), Booster T1, and NVIDIA Nova Carter. Nova Carter replaced the custom wheeled
robot by user decision on 2026-10-19; the custom robot remains only as a MuJoCo cross-physics reference.

| robot | controller | battery (10 seeds × 12 s per condition) | result |
|---|---|---|---|
| NERVA | frozen neutral ONNX (sha 610b1c59…), NERVA's control step and SafetyMonitor | rest, forward, backward, turns, stop/start, head motion, repeated pushes 0.05–0.20 m/s walking and at rest | 0 falls in all nominal conditions and with pushes ≤ 0.15 m/s; repeated 0.20 m/s pushes at rest: 5/10 falls (MuJoCo: the same 5/10) |
| T1 | Isaac Lab PPO v1 (`policies/t1_isaaclab_v1`, sha b14d86f3…) + frozen velocity map | speeds 0–0.5, turns ±0.5/±1, stop/start, repeated pushes 0.1–0.8 m/s walking/turning/standing | 0 falls nominal; pushes ≤ 0.3 m/s 0 falls; 0.5 m/s turning 3/10, 0.8 m/s 2–7/10; with the map, speed error < 0.008 m/s over 0.05–0.20 m/s |
| Nova Carter | the asset's wheel drives, (v, ω) interface | straight, turn, arc, stop, shove 0.6 m/s | speed error 0.002 m/s, turn error 0.023 rad/s, stop 0.10 m, supervisor trips on shove, no fall; signed drive test confirms forward motion |

**NERVA cross-physics** (same battery and stack; MuJoCo = the validated backlash model):

| condition | MuJoCo | Isaac |
|---|---|---|
| rest tilt (95th pct) | 10.9° | 10.5° |
| forward 0.074 m/s | 0.072 | 0.051 |
| backward −0.074 m/s | −0.085 | −0.068 |
| turn ±0.60 rad/s | +0.649 / −0.562 | +0.641 / −0.609 |
| repeated 0.20 m/s pushes at rest | 5/10 falls | 5/10 falls |

**Failed or superseded iterations** (kept for the record):
- The MuJoCo-trained T1 policy falls in PhysX within 0.6 s, even though the kinematics match to 0.07 mm. It was
  replaced by Isaac Lab training.
- T1 v2 (yaw-rate commands, tighter tracking, 0.8 m/s training pushes) failed its battery: a crouched gait below the
  0.45 m trunk threshold, and no tracking at 0.1 m/s.
- The NERVA import needed four fixes: merged defaults, MuJoCo inertials, joint frames rebuilt from MuJoCo, and gains
  in SI units. The backlash joints are removed.

**Frozen from the gates, before any experiment data:**
- the people's lunge surface speed is capped at 0.15 m/s (NERVA's survivable repeated push);
- the T1 velocity map.

## M4 evidence

- **Scenes.** Warehouse, home and hospital (held out). Each has three stations at 2.2 m from the hub and one
  bystander walking a 3 m patrol at 1 m/s, found automatically by PhysX raycasts.
- **People.** Stock characters, animated through Isaac's AnimationGraph (retargeted idle and walk) and moved with
  the graph's character API. The physical body is a kinematic capsule; PhysX contacts are evaluation truth.
- **Eyes.** A depth eye per robot is posed every frame from the PhysX link pose and is roll-free, as L's camera.
  The eye check (person 0.8 m ahead, scripted 0.25 m lunge, true abstract reach 0.40):

| robot | reach estimate |
|---|---|
| Nova Carter | 0.36 |
| T1 | 0.46 |
| NERVA | 0.33 |

- **Perception.** `MotionPerception(articulated_target=True)`: only pixels moving away outside the target band count
  as egomotion, because a person's limbs move. K/L behaviour is unchanged by default.
- **Encounters** (experiment mode, warehouse): all three arrive and honour stand-offs; IMU and camera detect pushy
  lunges; no falls.
  - IMU contrast: clear on Nova Carter; weak on T1 (31 kg, 0.15 m/s contacts); dominated by gait impacts on NERVA,
    so NERVA relies mostly on its camera.
  - NERVA's head-mounted eye looks about 16° up when it stands, so its depth image sees no floor; the person still
    fills the centre columns.
