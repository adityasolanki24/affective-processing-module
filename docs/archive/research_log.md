# Research log

Newest first. Measurements only from runs actually executed; hypotheses labelled. Programmes F–H keep detailed
logs in their own folders; this log summarises them.

## 2026-10-17: programme I (v0.5), autonomous revision
- **The cause.** Revision had been slow because collisions carry about 0.005 nats per encounter. Self-observed
  near-misses carry 1.1–1.5 nats, and EM finds person types from them without labels (K = 3). From collisions alone
  it finds K = 1.
- **I1** (12,000 test lifetimes): the new memory passes all criteria.
  - revision in 2/3/2 episodes (A0: never);
  - AUROC 1.00;
  - avoidable collisions −56%, all with careless people;
  - completion −0.3 pp.
- **I2** (16,000): under outright keep-away, probing restores 8.7 pp completion in the trap world. Targeting adds only
  0.7 pp, and probing narrowly fails efficiency non-inferiority overall. Thompson sampling freezes the robot.
- **G2b's memory ranks people inverted** (AUROC 0.21).
- **Compute:** first use of 128-core cloud VMs for this programme, about 18× the laptop; about US$2 of a US$40 cap.

## 2026-10-17: programmes F, G and H merged into `main`
- **F (v0.2):** the v0.1 PAD regulator beat B1 by 25% (E2), but memory revision failed (E3), and the learned
  regulator B3 was not credible (E4).
- **G (v0.3):** decision A, the simple predictive regulator. PAD is harmful under equal tuning; the earlier
  advantage came from tuning budget. Explicit memory is essential (removing it: +224% avoidable collisions).
  Deployment-time weight learning is blocked until there is a powered promotion gate. Cloud spend ≈ US$13.3 of
  US$30 approved.
- **H (v0.4):** decision B, an experience-conditioned expressive mapping. Gates H0 → H0b and H1 → H1b passed after
  re-tests; H4 → H4b passed with a corrected control. US$0 cloud spend.
  - H2 human study pending an ethics determination.
  - A researcher self-pilot (not blind) found the styles hard to tell apart.
- **Housekeeping:** CI type errors fixed (CI had been red on the research branches since programme G, because of
  numpy stub changes and torch typing); README and docs updated; a leftover idle Cloud NAT gateway, created by
  another tool, was deleted.

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
