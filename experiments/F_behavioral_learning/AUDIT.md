# APM v0.2 repository audit and experimental blockers (2026-10-11)

Audited at commit `62e771e` (branch `research/apm-v0.2-experiments-t8kijm`, identical to `main`). Code was read, not
only the README. Environment: cloud container, 4 vCPU, 15 GB RAM, no GPU, Python 3.13, torch 2.14.1 (CPU),
NumPy 2.x. Baseline at audit start: `ruff check .` clean, `mypy src/apm` clean, `pytest` 42 passed (core +
learning).

## Prior findings (preserved, not re-interpreted)

The ten findings listed in the programme brief are consistent with the committed results
(`experiments/C_learning/README.md`, `experiments/D_portability/README.md`, `docs/research_log.md`). Nothing in
this audit changes them. In particular C1, C4 and TR2 failed as preregistered; C3 passed only as specified and is
confounded; PAD-only showed no measurable predictive effect (exploratory).

## What the code actually does (relevant to v0.2)

| area | finding |
|---|---|
| `apm_testbeds/wheeled.py` | `WheeledWorld._control()` owns navigation (goal heading + obstacle repulsion). There is no action input: every APM experiment so far was observe-only. No safety supervisor exists apart from the reactive repulsion; a contact stops the robot. Pedestrians move ballistically, ignore the robot and have no individual behaviour, so identity carries no information about hazard in v0.1 data. |
| `apm/adapters/wheeled.py` | Owns the outcome detectors and a time-to-contact (TTC) collision estimator `p = σ((1.5 s − TTC)/0.4 s)` using radial closing speed only (no lateral miss distance). Hypotheses are emitted while closing and never retracted. |
| `apm/core` | Appraisal/PAD/tendencies are hand-set (documented). Outputs are advisory and nothing consumes them in any loop. |
| `apm/memory/entities.py` | One Bernoulli trial per encounter (track lifetime) per outcome kind, credited to identity (weighted by identity confidence) and type pool. Encounters end when a track expires in `APM.step()`. |
| `apm/learning` | GRU / statistical predictors are offline only; not in any inference or decision loop. Dataset builder: features at sample time t include messages stamped ≤ t; labels count events in (t, t + H] (no same-step leakage found). |

## Suspected defects, verified by tests before any fix (`tests/test_v02_defects.py`)

| id | defect | verification | action |
|---|---|---|---|
| D1 | A sender `Reset` replaced the context but left entity encounters open forever: their evidence was never credited, and a reused track id after restart merged two encounters. | `test_d1_reset_ends_open_encounters` **failed** on `62e771e` | **fixed in core** (`runtime.py`): open encounters end at the Reset stamp, exactly as an expired track does |
| D2 | When the bounded context (64 tracks) evicted its oldest track, the matching encounter was never ended (unbounded leak; evidence lost). | `test_d2_evicted_tracks_end_their_encounters` **failed** (70 open encounters for 64 tracks) | **fixed in core** (`context.py` returns the evicted track; `runtime.py` ends its encounter) |
| D3 | Stale predictions: v0.1 wheeled TTC hypotheses are not retracted, so after an entity stops closing the APM keeps appraising the last probability for up to its horizon (≤ 3 s). | test passes = stale held hypotheses observed | v0.1 adapter **kept unchanged** (validated baseline). The v0.2 controllable adapter retracts (probability 0) when the condition ends |
| D4 | Contact chatter: after a contact the robot can separate for one step and touch again; the detector reports a new collision with the same entity. 10 of 65 collisions in 20 v0.1 episodes (seeds 50,000–50,019) were re-contacts < 0.5 s after the previous one. | test passes = repeats observed | v0.1 testbed kept; the v0.2 testbed uses a 1.0 s per-entity refractory period and reports contact episodes, not contact steps |
| D5 | One RNG drives layout, slip noise and new goals, so any trajectory difference changes all later goals: policies cannot be compared with common random numbers. | test passes = goal sequences diverge after one extra draw | v0.2 testbed uses separate seeded streams (layout, goals, each pedestrian, slip, sensing) and a pre-generated goal sequence |
| D6 | Excessive risk-estimator triggering. Over 20 v0.1 episodes, a collision hypothesis is live on 49% of control steps, p ≥ 0.5 on 21% of steps, and only 21% of p ≥ 0.5 alarms are followed by a collision within 1 s. This is the input that keeps hand-set affect chronically negative (mean final valence −0.63 over those 20 runs). | `test_d6_…` passes | kept as the robot-side estimator for continuity (all systems receive it equally); its calibration is a measured E0 result and a E7 stress condition |

Other checks:
- **Outcome attribution.** Only targeted outcomes (collisions with a contact target) are credited to an entity;
  untargeted outcomes (slip) are never attributed to nearby entities. Verified by reading `APM._outcome` and
  `EntityMemory.outcome`. Not a defect, but E3 tests it under co-occurrence.
- **Censoring (design limitation, not a code defect).** An encounter in which the robot avoided an entity counts as
  a no-collision trial. Memory therefore cannot distinguish "safe" from "avoided", and avoidance can erase the
  evidence that motivated it. E3 tests this explicitly.
- **Contact-slip semantics.** The wheeled adapter reports wheel slip as `stability_loss` but declares
  `contact_slip` UNSUPPORTED; its stability margin is the traction margin computed from the true friction
  coefficient (an idealised slip estimator). Kept; noted as a limitation.
- **Restart.** Saved state holds memory, affect and seen outcome ids; open encounters and the current context
  window are intentionally not saved (lost on crash). Covered by existing tests plus D1.
- **Counterfactual learning.** Hypotheses never teach memory (existing test `test_predictions_never_teach_outcomes`).
  No code path labels an action not taken. In the v0.2 testbed blocked actions (safety interventions) are logged
  as interventions; their outcomes are whatever actually happened.
- **Train/test leakage in existing tests/experiments.** C and D split at episode level (no sample-level leakage).
  But **old evaluation sets must not be reused as fresh data**: v0.1 wheeled seeds 0–11, 10,000–10,399, 10,999,
  50,000–50,019, 70,000–70,003 and NERVA C trials (master seed 2026, simulator seeds 0–599) are excluded from all v0.2
  confirmatory sets. **Hazard found:** `examples/nerva/generate_experience.py` seeds each simulator with the trial
  index, so a "new" NERVA dataset with a different master seed would reuse the same simulator noise streams. v0.2
  NERVA data must use simulator seeds ≥ 100,000.

## Blockers and constraints

| # | blocker | consequence | status |
|---|---|---|---|
| B1 | No behavioural intervention interface exists (wheeled control is internal). | E2–E5, E7 behavioural tests impossible. | resolved by E0 (new controllable testbed) |
| B2 | v0.1 pedestrians carry no individual hazard; identity is uninformative. | Memory/disposition (E3) and the value of identity history cannot be tested on v0.1 data. | resolved by E0 (hidden per-identity dispositions, change schedules) |
| B3 | The validated NERVA policy `candidate.onnx` is not in Git (it lives in the user's local `NERVA/experiments/cloud_runs/…`), and the 600-trial C experience is local to the user's PC. MuJoCo 3.3.0, onnxruntime 1.21.0 and the public Open Duck Playground assets do install here. | Fresh NERVA data (E1 NERVA arm, E6) need the policy file. | **open**: needs the user's folder access or the file |
| B4 | NERVA telemetry has no entities, collisions or behavioural interface, and NERVA must stay observe-only. | No social/behavioural cross-robot claim is possible (E6 info mismatch); behavioural experiments are wheeled-only. | structural, recorded |
| B5 | No paid cloud budget. | All compute must fit this container unless the user approves a ceiling. | benchmarks decide (E0) |
| B6 | `pytest` on this container's PATH is a uv tool without the package; use `python3 -m pytest`. | none | workaround |
