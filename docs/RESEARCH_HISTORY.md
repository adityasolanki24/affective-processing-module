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
| K (v0.7) | 2026-10-18 | recovery from negative experience; frozen-task confirmation; continuous demonstration; other bodies | Recovery was limited by information (IMU at a safe distance) **and** by a memory defect (D1: no change step between consecutive meetings). Fixed memory + value of information + a modelled camera: **recovery in both directions on NERVA and a wheeled robot, 0 falls**. Decision-theoretic mapping **replaces L0** (4/4 frozen tasks). Continuous two-robot demo with controls | `experiments/K_recovery/REPORT.md` |
| L (v0.8) | 2026-10-18 | independent audit corrections; genuine vision; stress tests; fair controller comparison; integrated runtime | Audit confirmed: unenforced risk cap (5–29% over-cap decisions), recovery-metric and target-calibration defects, proxy task success, abstract camera. **Corrected on fresh seeds, 6/7 K claims survive** (caution-return claim fails for wheeled IMU). Rendered depth vision passes the perception gate on NERVA/wheeled; recovers on NERVA, too slowly on wheeled (K's camera result depended on the idealised sensor). Memory halves contacts under stress but its cost gain is significant in 6/15 scenarios. Runtime passes F1–F6 on 3 bodies; continuous demo with vision | `experiments/L_audit_perception/REPORT.md` |
| M (v1.0) | 2026-10-19 | final release: Isaac Sim, three robots, realistic scenes, preregistered A–F | Isaac Sim 5.1: Nova Carter and T1 run APM in warehouse, home and held-out hospital with physically present animated people and rendered depth eyes; NERVA imported (gates pass) but fails the home scene, so its comparative data are MuJoCo (D3). 624 X lifetimes: **B1 and C1–C4 on all three robots**; B2 and D1–D3 on two; A1 on T1 only; **A2 (identity-specific memory) not demonstrated**. Explicit memory beats a GRU. 0 executed over-cap approaches; T1 16 falls (all systems, pushing person). Not merged | `experiments/M_release/REPORT.md` |
| M remediation (v1.0 snapshot) | 2026-10-19/20 | fair comparisons, visible expression, T1 gait, simulator audit | **D6:** a person-contact artefact had thrown Nova Carter in ~9–12% of encounters, fixed and all affected Isaac comparisons rerun on fresh seeds. **Identity-specific memory supported on all three robots** (beats pooled, swapped, shuffled). APM beats rule, lookup, fixed-style and no-memory baselines but **not neutral motion with the same supervisor**. Experience-dependent movement measured on all three robots; recovery incomplete. Booster's official T1 controller smoother but **failed the frozen stability gate**; v1 kept | `experiments/M_release/REMEDIATION_REPORT.md` |

## Architecture recommendations over time

Only the last item is current.

1. v0.1: PAD affect core behind AffectLink ⟶ superseded by G (PAD harmful).
2. G, decision A: predictor + cumulative entity memory + EMA risk ⟶ memory replaced by I's type memory.
3. H, decision B: memory + fixed lookup + bounded adapter ⟶ J5 shows the lookup does not close the loop on NERVA.
4. J, decision E: keep explicit type memory + L0 ⟶ superseded by K.
5. K: explicit type memory with the D1 fix + decision-theoretic expression mapping + non-myopic information
   gathering ⟶ L found the risk cap unenforced and the camera abstract.
6. L: the same memory and mapping with an **enforced risk cap**, **IMU + rendered depth vision**, an independent
   supervisor, in one runtime behind AffectLink; value of information optional.
7. **M, v1.0 (current):** L's architecture unchanged, greedy (no value of information) by default, on three bodies
   in Isaac Sim and MuJoCo, with each robot's own controller and physical supervisor as the second safety layer.
   The remediation added a learned, label-free experience state and a designed multi-phase expressive program
   (observe, approach, interact, depart) realised through each robot's posture and timing channels.

## Conventions used throughout

- Every confirmatory test was preregistered, run once from a committed tree, and reported, including failures.
- Deviations are logged per programme (`DEVIATIONS.md`).
- Seeds are partitioned per programme (`seeds.py`).
- Raw data are git-ignored and hashed in manifests.
