# J4 addendum: revisable experience on NERVA (committed before any J4-X lifetimes, 2026-10-18)

This addendum supplements `MASTER_PREREGISTRATION.md` §4.2. J4 runs after J5 because the first bank was lost to a
spot preemption. J4's result does not change J5's, which used its own type model.

## Frozen inputs (git-normalised SHA-256, first 16 hex)

| file | hash | contents |
|---|---|---|
| bank `X_bank.jsonl` (job J4bank2, not committed: `runs/`) | `cd01f589d5fa140c` | 4,000 physically simulated NERVA standing encounters at 0.4 m, 1,000 per type, J3 seeds partition X |
| `configs/J4_types.json` | `9a5f249fccdebedd` | J3-T EM model, K = 2, detection rates 0.21 / 0.52 |
| `configs/J4_edges.json` | `f89ed63dc918634b` | acc8 edges, J3-T |
| `configs/J4_calibration.json` | `db734d5a79d2323d` | per memory: μ_pushy, μ_calm, threshold, from J4-V |
| `configs/J4_gru.json` | `6eb2d2339682849f` | recurrent baseline weights, trained on J4-V |

**Partitions of the bank.** Rows 0–299 of each type are validation (λ, calibration, GRU training). Rows 300–999 are
the test. J4-V lifetimes use seed 20261019; J4-X lifetimes use seed 20261018. Lifetimes are generated from the bank
without physics, so J4-X runs locally.

**λ = 0.005 per encounter.** Selected on J4-V by the self-supervised predictive log-likelihood over the fixed grid
(`results/J4/J4_V_lambda.json`). The objective is nearly flat: −2.036 (λ ≤ 0.01) to −2.073 (λ = 0.5).

## Recurrent baseline (pre-specified now; the master preregistration requires it if both explicit memories fail)

`nerva_memory.GruMemory`, trained by `j4_gru.py`:
- per identity, a GRU (hidden 8) over that identity's own acc8 category and detection;
- objective: predict the identity's next detection, self-supervised on J4-V lifetimes;
- no type labels, no type model, no kind pooling;
- validation log loss 0.586 vs 0.611 for a constant rate.

It runs in every J4-X lifetime. **Its result is interpreted only if triggered:** H_A0 and type_kind both fail at least
one of C1, C3 or C4. It is reported either way.

## Validation check (disclosed)

The criteria below were drafted before any J4-V lifetime existed and are unchanged. On J4-V (`results/J4/J4_V_check.json`;
the GRU numbers there are in-sample):
- type_kind: AUROC 1.00, caution effect +0.285 vs H_A0 +0.392;
- revision change_B (pushy → calm) not reached within 4 blocks by any memory;
- revision late_B: type_kind 3 blocks, gru 2, H_A0 and count not reached;
- restart exact.

C2 and C3 are therefore expected to fail on X for type_kind.

## Test

- J4-X: `j4_lifetimes.py --rows 300:1000 --n 200 --seed 20261018 --lam 0.005 --gru configs/J4_gru.json`;
- 7 histories × 200 lifetimes = 1,400 lifetimes × 48 encounters;
- memories: H_A0, count, type, type_kind, type_kind_restart, gru.

## Criteria (`j4_analyze.py test`; primary memory: `type_kind`; reference: `H_A0` on the robot-observable detector)

| criterion | requirement |
|---|---|
| C1 experience dependence | history AUROC (risky_B vs safe, end-of-lifetime B1 p) ≥ 0.90 |
| C2 history effect | L0 caution effect (risky_B − safe, B1) non-inferior to H_A0's: the paired difference of effects has a 95% bootstrap lower bound ≥ −0.05 |
| C3 revision | change_B (pushy → calm) **and** late_B (calm → pushy): Kaplan–Meier median ≤ 2 checkpoint blocks of 6 encounters after the switch (horizon 4 blocks); B1 is met about once in 6 encounters |
| C4 persistence | restart: max \|readout difference\| = 0 (exact) |
| C5 false attribution | zone_B: normalised B1-plain blame ≤ 0.5. **Expected to fail** for every identity-keyed memory, since none is keyed by context. Reported either way |

**J4 passes** if C1–C4 pass.

C3 reads the master preregistration's template "adaptation ≤ 6 encounter-blocks" as at most 2 blocks of 6
encounters. That allows roughly 2 meetings with B1 after the switch.

**Secondary (exploratory):**
- generalisation to the novel B3 (only kind-pooling memories can show it);
- risky_A specificity (B1's belief should not rise);
- C1–C5 for H_A0, count, type and gru.

C2 for the other memories uses the same paired difference against H_A0.
