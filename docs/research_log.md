# Research log

Newest first. Measurements only from runs actually executed; hypotheses labelled.

## 2026-10-11: Milestones D and E
- **D (`ece9723` preregistration):**
  - wheeled robot added; the same core ran unchanged (hashes identical);
  - 10,899 demo messages, none rejected;
  - zero-shot NERVA→wheeled slip prediction AUROC 0.83, but poorly calibrated;
  - fine-tuning was worse than training from scratch at 5 and 20 episodes, so **no learned-transfer claim**;
  - the wheeled GRU beat logistic regression on slip (−17% Brier) and collision (−38%).
  - The hand-set baseline is chronically negative-valenced on the busy wheeled robot.
- **E:** hardware design document; PC benchmark:
  - APM step 0.32 ms median;
  - GRU 0.52 ms single-thread inference;
  - 3,400 training samples/s;
  - 76 kB state.

  No board was available.

## 2026-10-11: Milestone C (learning on NERVA experience)
- **Data:** 600 locally simulated NERVA trials (validated policy, random pushes up to 1.4 m/s): 276 stability
  losses, 275 falls, 1 recovery.
- **Preregistered (`21948c7`):**
  - C1: GRU vs statistical baselines on Brier: **fail** (0.0166 vs 0.0167);
  - C2: calibration: pass;
  - C3: affect inputs help: pass but confounded;
  - C4: context generalization: **fail**;
  - C5: forgetting: passes trivially, since incremental learning changed almost nothing;
  - the corrupted shadow update was rejected.
- **Exploratory ablation:** PAD adds no measurable predictive value; the gain comes from the appraisal vector (which
  carries the adapter's 50 Hz risk estimate).

## 2026-10-11: Milestone B (NERVA through AffectLink, observe-only)
- **Replay** of 160 stored NERVA gate rollouts:
  - no outcomes, consistent with the gate;
  - affect bounded;
  - 0 rejections;
  - the adapter's tilt extrapolation raised p ≥ 0.5 false alarms in 14 of 160 trials.
- **Live rollout:** a 1.0 m/s push; stability loss predicted ≈ 140 ms ahead, then loss and fall observed.
  - It exposed a seek-assistance flaw (fixed with a regression test).

## 2026-10-11: Milestone A (repository, AffectLink v1, APM-0 core)
- Implemented the schema, codec, validation and core with 25 tests (synthetic inputs).
- The standalone synthetic scenario exposed four implementation defects, all fixed before the milestone closed:
  1. phasic gain large enough to drive valence to the −1 bound after one collision;
  2. event streams wrongly marked stale;
  3. strain from comfortable margins;
  4. memory expectations with no effect on appraisal.
  Memory now raises internal predictions for tracked entities, which are never learned as outcomes.
- [design] All appraisal and affect constants are hand-set, transparent baseline values, not fitted to any data.
