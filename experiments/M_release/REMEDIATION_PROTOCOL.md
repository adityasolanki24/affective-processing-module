# Visual review gate (exploratory corrective work)

The first deliverables are two actual-physics review videos. No final validation,
release composition or packaging proceeds until the user accepts both visually.
Original M results and the release candidate at c150288 remain intact.

1. T1: old Isaac Lab v1 versus a candidate, identical task, seed and camera rule.
   Show stand, forward walking, stop/start and turning at real time with visible feet.
   Retain failures and record joint position/velocity, root motion and foot poses.
   Pilot seeds 71001–71003; these are not confirmatory seeds.
2. Expression: same morphology/controller and initial memory, different physical
   encounter histories followed by a matched present task. No emotion names,
   belief bars or history-revealing captions. Include neutral and two histories,
   then speed- and distance-matched controls. Preserve the hidden assignment key
   separately. No time warping, animation playback or root stabilising forces.

Diagnose existing footage before candidate selection. The published official
Booster T1 policy is the first sim-to-sim candidate; it is not assumed compatible
with the stock Isaac asset. Verify names, observation order, torque limits and
physical behaviour. A candidate that falls is not promoted. If adaptation fails,
retain its evidence before deciding whether Isaac Lab fine-tuning is warranted.

This gate establishes engineering visibility, not human emotion recognition.
Learned Bayesian memory and designed expression mapping must be distinguished.
No participants, hardware, merge, publication, history rewrite or deletion of old
data are authorized by this protocol. Final confirmatory seeds and full-lifetime
failure analysis must be frozen in an addendum after visual acceptance.

## Exploratory deviation, after first video pilot

The inherited type readout barely distinguished the two T1 histories, and its
perturbation readout is identically zero on NERVA. Confidence alone produced weak
history contrasts. Do not amplify those results and call the inherited model good.

Pilot an explicit alternative: a per-identity Beta memory of substantial camera
motion events (estimated reach >= 0.14 m, the 0.05 m motion threshold plus three
times 0.03 m navigation uncertainty). Exponential retention is 0.9 per meeting.
Only recorded camera estimates enter; absent sensing windows are skipped. This
predicts detected camera movement, not collision risk or an emotional state.
Its mapping to inspection amplitude is designed. Selection used pilot data, so
all resulting comparisons are exploratory; a later held-out test is still required.
Replay restores the first six actual encounters from logged sensor records;
subsequent evaluation physics is fresh and explicitly discontinuous from history.

## T1 gait validation battery (frozen before any battery data, 2026-10-19)

The user accepted the candidate only for full validation, not as a replacement.

**Stacks compared.** Complete controller stacks, each as it would be deployed:
- **v1:** the Isaac Lab PPO policy, its velocity map, 5 ms physics and PhysX PD;
- **candidate:** the official booster_gym policy (pinned `da396a0`) with its simulation PD gains as torque PD at
  2 ms, without the deployment target filter.

Each stack keeps its own supervisor (tilt trip 25°, fall at 50° or a root below 0.45 m).

**Environments.** A flat ground plane, plus the warehouse, home and hospital floors. In each scene the robot starts
at the hub, heading along the first 15° direction whose 4 m path is collider-free and at least 1 m from every
station.

**Scenarios** (stand 2 s first; command profiles fixed in `gait_battery.py`):

| scenario | profile |
|---|---|
| speed_v | forward v ∈ {0.1, 0.2, 0.3, 0.4} m/s for 12 s, then stop 4 s |
| turn_walk_± | 0.2 m/s with ±0.5 rad/s for 8 s |
| turn_place_± | ±0.5 rad/s in place for 8 s |
| startstop | 4 cycles of 3 s at 0.2 m/s and 2 s stopped |
| push_walk_m | 0.2 m/s with velocity pushes of m ∈ {0.2, 0.3, 0.5} m/s at 8 s and 12 s, random direction |
| push_stand_0.3 | standing, pushes at 4 s and 8 s |

**Runs.** Seeds 72001–72005, fresh and never used for pilots. Pairs are matched on environment, scenario and seed.

**Measures** (from the recorded physics):
- falls;
- forward-speed and yaw-rate tracking error;
- stance slip, swing clearance, stride time and its asymmetry;
- tilt p95;
- joint acceleration RMS and a torque proxy (mean Σ τ², from measured joint efforts);
- stop distance and settle time;
- peak tilt after a push.

**Promotion rule** (all conditions must hold):
1. No condition where the candidate falls more often than v1, and 0 candidate falls in all nominal scenarios and
   with pushes ≤ 0.3 m/s.
2. Median paired improvement in stance slip, joint acceleration RMS and tilt p95 across all runs, and no scenario
   family (speed, turn, startstop, push) where any of the three is worse by more than 10% (median paired ratio).
3. Speed tracking: per speed, the candidate's mean absolute error is at most max(0.03 m/s, v1's error + 0.02).
   Turn rate: error at most v1's + 0.1 rad/s.
4. Stop distance and settle time no more than 20% worse than v1 (median).

If any condition fails, v1 stays the controller and the failure is reported. A command calibration (velocity map)
for the candidate may be fitted only on separate calibration seeds 72101–72103, not on battery seeds.

### Battery deviation G1 (before any warehouse data)

The warehouse hub has no collider-free 4 m straight path (shelving). Start points are now tried from the hub
outward (rings at 0.5 m and 1.0 m, every 45°), with the same 4 m, collider-free and ≥ 1 m-from-stations rule. The
other environments' starts are unchanged: their hub qualifies first. No scenario, seed, measure or criterion
changed.

## Corrected comparisons (5.1–5.3; frozen before any data, 2026-10-19)

Programme M's results stay as recorded. These are new tests on fresh seeds: experiment `R`, partition `Y`, never
used before this addendum.

**5.1 Matched comparisons.** Every system runs under the same APM risk supervisor and cap (`--supervision
matched`), with the same observations (IMU and camera), the same action grid and limits, and the same task.
- MONOTONE* is re-tuned under supervision, by M's rule (lowest mean realised cost on the 4-configuration grid) on
  fresh validation seeds R/V 0–2, before any R/Y run.
- Systems: DT, MONOTONE*, FIXED, NEUTRAL, LOOKUP, NOMEM, plus the identity controls below.
- Robots: Nova Carter (Isaac), NERVA (MuJoCo, labelled), and T1 (Isaac), the last with the controller the gait
  battery selects.
- Warehouse, histories change_B and late_B, Y indices 0–14 (30 lifetimes per system and robot).

**5.2 Fall-aware outcomes.** Reported separately, never folded silently into cost:
- full-lifetime completion rate and falls per lifetime (each fall listed);
- contacts and the peak contact impulse;
- task success, approaches completed and supervisor vetoes;
- cost conditional on survival (the encounters that took place);
- cost over all 24 planned encounters, where each encounter lost to a fall counts as a failed encounter (the
  45 s approach budget plus the retry time).

The second cost adds no extra penalty for the fall itself, because fall risk is reported on its own.

"Better overall" requires both of these:
- lower cost over all planned encounters (one-sided paired sign-flip test, Holm within robot, α 0.05);
- no stability regression, meaning the fall rate is not higher. Higher means a one-sided Fisher exact test at α
  0.05 on lifetimes with a fall.

**5.3 Identity-specific memory.**
- History risky_B: B1 pushy throughout, while A1 and C1 are calm.
- DT is compared with:
  - POOLED: one shared memory for everyone, i.e. population-wide caution;
  - SWAPPED: when deciding, the robot recalls another person's memory (A1 ↔ B1); experiences are stored under the
    true person;
  - SHUFFLED: as in M.
- Measures:
  - cost over all planned encounters;
  - B1 contacts per encounter, after B1's second meeting;
  - A1/C1 cost per encounter, i.e. needless caution with calm people.
- Y indices 0–14 (15 lifetimes per system).

Identity-specific adaptation is supported only if DT beats both POOLED and SWAPPED on cost (Holm, α 0.05), and
does not have more B1 contacts than POOLED. If not, the limitation is reported; no effect is amplified.
