# Programme G experiment matrix

Status legend: **done** (results committed) · **ready** (code + tests + pilot, awaiting compute approval or lock) ·
**designed** (preregistered at programme level; addendum and code pending) · **gated** (runs only if a gate passes).

| exp | question | systems / factors | partitions | primary family (Holm) | depends on | status |
|---|---|---|---|---|---|---|
| G0 | fair comparisons possible? | instrumentation, equal-budget tuning, integrity tests | P | gate: all integrity tests pass | — | **done** (14 tests) |
| G1 | PAD necessary, or simple persistent risk regulation? | R0–R8, 2×2 negative controls, B0; conditions nominal / miscal ×3 / dense / sparse | P pilot (done) · V tuning (K = 48 per family, N = 30 × 10 episodes) · T (R5 ES) · X (N = 280) · S (4 × 100) | P1 R6 > R1 · P2 R_best ≥ R6 (NI) · P3 R_best > R1 | G0 | **done**: P1 fails (PAD worse, +26%), P2 holds, P3 fails (−7%); see reports/G1_REPORT.md |
| G2 | which memory adapts and revises? | M0–M6 × evidence model (all vs exposed-only) × probing (none / uncertainty-guided / matched random); 9 scenario families | P, V, X | (i) adaptation ≤ 6 episodes (KM), (ii) collision NI vs M0, (iii) false blame ±0.03 | G1 regulator (fixed) | **done**: no memory meets the adaptation criterion; exposure gating −10–14% collisions; G2b: time decay + uncertainty probing fastest (`reports/G2_REPORT.md`) |
| G3 | what persistent state is needed? | none / EMA / LDS / GRU / fast+slow / PAD / learned latent; controlled-event episodes; reset, shuffle, identity-shuffle, clamp | P, T, V, X | any state > none; slow state > best single timescale | G1 | **done**: no persistent state helps the predictor (Q1 fails; Q2 holds only vs a weak EMA) (`reports/G3_REPORT.md`) |
| G4 | which architecture family? | A0 (bar), A1, A2 GRU, A3 GRU+memory, A4 residual regulator, A5 fast/slow (if G3), A6 short-horizon MPC (if justified) | T, V, X; 5 seeds | each shortlisted neural candidate > A0 | G1–G3 | **done** (with G5): no neural candidate beats A0 (`reports/G45_REPORT.md`) |
| G5 | is training the bottleneck? | PPO, BC, BC→PPO, DAgger, model-based, residual; matched pretraining; all data counted | T, V, X; 5 seeds | best method > PPO (≤ interactions); best-trained GRU ≥ A0 (NI) | G4 shortlist | **done**: BC→PPO 17% safer than PPO; best GRU not non-inferior to A0 |
| G6 | robust under shift? | held-out procedural variations (≥ 10); no retuning | S (+ new held-out families) | eligibility: NI to A0 in ≥ 80% conditions, never > 2× A0 collisions | G4/G5 | **done**: only A0_mem eligible; neural candidates not robust (`reports/G6_REPORT.md`) |
| G7 | continual adaptation, cross-embodiment? | frozen + memory / periodic / shadow + replay / alt. CL; NERVA comparable signals only | new phase partitions | per addendum | G6 survivors | **done** (predictor updates): gate blocked all updates; ungated replay learns but regresses safety (`reports/G7_REPORT.md`) |
| G8 | minimal architecture | ablations of every component; Pareto (behaviour vs complexity, compute, robustness) | X/S reuse only for reporting ablations of the already-selected model; no reselection | per addendum | G7 | **done**: memory and neural prediction needed; combined minimal model not confirmed (`reports/G8_REPORT.md`) |

| G9 | validity re-test of G1/G3/G4–G5/G6/G8 on avoidable collisions (testbed defect D8) | same frozen systems/models | fresh X/S indices 10000+ | as the original families | G1–G8 | running |

Programme F items accounted for in `reports/F_ACCOUNTING.md` (E1 completed under `F/prereg/E1_ADDENDUM.md`; E5/E6 reduced versions superseded by G7).
