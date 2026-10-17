# G7 addendum: lifelong learning (fixed before any G7 data, 2026-10-12)

Supplements `MASTER_PREREGISTRATION.md` §5 G7. G7 runs only for architectures surviving the earlier gates. Every
result so far (G1–G3; G4–G6 pending at the time of writing) points to the non-neural A0 family; its learnable part is
the action-conditioned consequence predictor, plus the explicit memory (which updates continuously by design). This
study therefore asks whether *updating the predictor* during deployment yields real held-out improvement without
forgetting or unsafe updates. If G4–G6 later select a neural architecture, a separate G7 study for it is added before
its data exist.

## Design (`scripts/g7_lifelong.py`)

- Phases: PH1 nominal; PH2 faster (U(0.8, 1.6) m/s) and more careless (60%) pedestrians; PH3 slippery (3–6
  low-friction patches) and crowded (4–6 of 8 people). None of these worlds was used to train F's predictor.
- Systems (A0's rule and memory unchanged; start = F's frozen predictor): L0 frozen; L1 periodic retrain on base ∪
  all experience, deployed without a gate; L2 the same candidate behind a promotion gate (validation: 20 G7-V
  lifetimes of the current phase, paired; promote iff mean J ≥ current and collision rate ≤ 1.05 × current, else keep
  the current predictor = rollback); L2c as L2 with 30% of new collision labels flipped (false outcomes); L3 retrain on
  the latest phase only (no replay; forgetting reference).
- Base data: 150 exploration lifetimes (G7-T 0–149). Experience: 60 closed-loop lifetimes per phase per system,
  collected with that system's deployed predictor. Predictors: 3-seed ensembles of F's MLP, F's training procedure,
  85/15 split of the training rows for early stopping.
- Frozen test suites: after each phase's update, **N = 150** G7-X lifetimes (5 episodes) of the current phase and of
  PH1; all systems on the same lifetimes. (N: no pilot exists for this design; 150 is between E5-type reduced runs
  and G6's sizes, recorded as a design choice; results are labelled inconclusive where CIs straddle the margins.)

## Criteria (`scripts/g7_analyze.py`)

**Lifelong success** iff, in PH2 or PH3, L2 *promoted* an update that improves the current-phase suite's mean J by
≥ 5% of L0's with bootstrap CI lower > 0, with collision r CI upper ≤ +5%, while the PH1 suite's mean J is not worse
than L0's by more than 5% (relative CI lower > −5%). Rejected updates are never counted as learning. Secondary: the
same for L1/L3 (no gate), forgetting on PH1 (L3), whether the gate rejected the corrupted candidate (L2c), number
of promotions.

## Cross-embodiment (reduced; secondary)

Only signals with shared semantics: `stability_loss` prediction from the canonical features of both robots (no people
or identities on NERVA, so no social transfer is studied). Programme F's Milestone D already showed that NERVA →
wheeled fine-tuning did not improve sample efficiency; G7 does not repeat it on new data unless a G4–G6 survivor has a
learned encoder that could transfer (none of A0's components is a learned encoder of raw signals).
