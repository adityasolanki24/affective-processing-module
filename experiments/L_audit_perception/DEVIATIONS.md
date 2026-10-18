# Programme L deviations

Every change after a `PROTOCOL.md` section is frozen is recorded here: what, why, when, and before or after which
data.

## LD1. A spot preemption lost the first LB1 V run (logged, as §B.5 requires)

- **What happened.** VM `apm-l-1` was preempted while running LB1 V; the job's output was lost.
- **What was done.** The job was rerun from the same commit (ffdc691) with identical seeds on an on-demand VM
  (`20261018-184022-LB1Vr`). The two queued jobs (LB2 V, LB1 X) were moved unchanged to that VM.
- **Effect.** None: the code, seeds and analysis are the same.

## LD2. The Phase D predictive-model trigger fired; the GRU was not run in L

- **What happened.** §D says the small predictive model is run "only if M1 and M2 both fail D1 in ≥ 2
  scenarios". After the LD X data, that condition held in 8 scenarios.
- **What was not done.** The GRU was not trained or run in programme L. This departs from §D.
- **Reasons.**
  1. J4's GRU reads IMU categories only. Testing it fairly here needs a new input coding (camera codes), a new
     self-supervised training run on LC5-T, and a mapping from its scalar prediction to a type belief. Each of
     those is a design choice that §D did not fix.
  2. D1 failed mainly because the task-cost difference against the no-memory baseline is small. A different
     memory cannot change that. In every scenario with risky meetings, M1 already halves contacts per risky
     meeting.
- **Status.** The comparison remains open and is listed as next work in the report.

## LD3. §F was written after the first runtime demos

§F is an engineering verification with binary checks and no tunable thresholds. Nothing was changed in response to
the demos. The protocol section says so itself.

## LD4. The Phase G video was re-rendered after a renderer defect (rendering only; data unchanged)

- **The defect.** In MuJoCo 3.3 (NERVA's environment), closing one `mujoco.Renderer` destroys the GL context that
  every renderer in the thread shares. The robot's depth renderer was closed after each encounter, so the
  third-person video went black from the first sensing window onwards; the robot's own depth inset was
  unaffected.
- **The fix.** The continuous world now owns one depth renderer for its whole lifetime.
- **What changed and what did not.**
  - The pre-declared lifetime (LG-D 0) was rendered again.
  - The depth images, decisions and trace are those of the same seeds.
  - The LG X data were produced without any video renderer and are unaffected.

## LD5. Tooling added after §B froze (no effect on frozen analyses)

- **`k_fit.py --edges`:** reuses a body's fixed acc8 edges, so §C.2's vision models share K's body configuration.
- **`wheeled_world.closed_loop_encounter(target=...)`:** used for §D's positions and context scenarios. The
  default call is unchanged; for the default target, the stand-off aim is K's exactly.
- **The uncertainty-triggered policy's per-encounter cap check** (an A3 follow-up). It was committed with §B,
  before any L data.

## LD6. §C.3's P4 refers to DT, which §C.3 did not run

- **The problem.** §C.3 adopts §B.3's criteria "exactly". P4 includes "repeated-push encounters not above DT's
  (paired upper bound ≤ +0.02)", but DT is not among §C.3's systems. That clause therefore cannot be evaluated, and
  the analysis script scored it as failed for every system. This was found after the LC X analysis.
- **How it is reported.** Both readings are given:
  1. the mechanical result, in which no system meets the milestone;
  2. P4 from the parts that can be evaluated: falls, trips and cap compliance.
- **Effect on Q1 and Q2.** None, on either body:
  - NERVA: every system has the same 2 falls. They come from one seed (LC4-X 24), at the first meeting with the
    pushy entity, where every system approaches at the prior.
  - Wheeled: DT_IMUVIS_VOI fails P2 and P3, and DT_VIS_VOI fails P3.
