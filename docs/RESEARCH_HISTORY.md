# Research history

Compact, chronological. Each programme's full record (preregistration, addenda, deviations, results, reports) stays
in its own `experiments/` folder and is never rewritten. **Later corrective results govern earlier ones.** Where that
happens, it is marked ⟶.

| programme | dates | question | decisive result | record |
|---|---|---|---|---|
| A–E (v0.1) | 2026-10 | build a portable core; learning; transfer; hardware | same core ran unmodified on NERVA and a wheeled sim; GRU predictor mixed; **no learned transfer**; PAD added no predictive value; hardware design only | `experiments/B_nerva_integration` … `E_hardware` |
| F (v0.2) | 2026-10 | does v0.1's APM improve behaviour? | PAD rule −25% collisions vs B1 ⟶ **G9: explained by a 9× tuning budget**; memory revision too slow; learned regulator not credible | `experiments/F_behavioral_learning/results/CONFIRMATORY_SUMMARY.md` |
| G (v0.3) | 2026-10-12 | smallest architecture that behaves better through experience | **Decision A:** predictor ensemble + entity memory + EMA risk penalty. PAD **harmful** (+99% avoidable collisions); neural regulators not better. G2b memory recommended ⟶ **I: G2b beliefs inverted, withdrawn** | `experiments/G_architecture_selection/reports/ARCHITECTURE_DECISION.md` (G9 governs G1–G8 on avoidable collisions) |
| H (v0.4) | 2026-10-13 | can experience be expressed through embodied motion? | **Decision B:** memory → fixed lookup L0 → bounded NERVA adapter; history recoverable from motion 98–100%, 0 falls; learned/recurrent mappings not better. Human legibility untested | `experiments/H_expressive_intelligence/reports/EXPRESSIVE_ARCHITECTURE_DECISION.md` |
| I (v0.5) | 2026-10-17 | autonomous revision under self-selected avoidance | **I1 passed:** dense near-miss evidence + EM types + change-aware belief; revises in 2–3 episodes; −56% avoidable collisions. VOI probing escapes outright keep-away traps. PR #3, **not merged** | `experiments/I_autonomous_revision/reports/AUTONOMOUS_REVISION_DECISION.md` |
| J (v0.6) | 2026-10-18 | validate I; transfer to NERVA; close the expressive loop | J1: I1 survives frozen calibration, but **its safety gain is not caused by revision**. J2: not a universal default. J4: NERVA memory learns from IMU, but revision is slow and one-directional. J5: **loop not closed with L0**. J6 (exploratory): 5-parameter decision-theoretic mapping beats L0; GRU adds nothing. **Decision E** | `experiments/J_validation_transfer/reports/FINAL_REPORT.md` |
| K (v0.7) | in progress | recovery from negative experience; frozen-task confirmation; continuous demonstration; second embodiment | — | `experiments/K_recovery/` |

## Architecture recommendations over time

Only the last row is current.

1. v0.1: PAD affect core behind AffectLink ⟶ superseded by G (PAD harmful).
2. G, decision A: predictor + cumulative entity memory + EMA risk ⟶ memory replaced by I's type memory.
3. H, decision B: memory + fixed lookup + bounded adapter ⟶ J5 shows the lookup does not close the loop on NERVA.
4. **J, decision E (current):** keep explicit type memory + L0. The decision-theoretic mapping is the candidate,
   pending K's frozen-task confirmation.

## Conventions used throughout

- Every confirmatory test was preregistered, run once from a committed tree, and reported, including failures.
- Deviations are logged per programme (`DEVIATIONS.md`).
- Seeds are partitioned per programme (`seeds.py`).
- Raw data are git-ignored and hashed in manifests.
