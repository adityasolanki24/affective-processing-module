# Programme K deviations

Dated entries only. `PROTOCOL.md` is never edited after its freeze (commit 13b7ef2).

## D1 (2026-10-18, after K3-X, K5-X and K3b ran, before K3-X was analysed): the change model skipped a transition

**Found by:** the K4 continuous demonstration, which meets one entity every encounter. Its belief stuck at exactly
1.0 for 20 encounters of calm camera evidence. A replay of the recorded observations reproduced it.

**The defect** (`nerva_memory.TypeStateModel._at`, in use since programme J). The belief relaxes toward the prior
by exp(−λ · (n − k)), where n is the encounter clock and k is the identity's last update.
- n advances only inside the update, so an identity met in consecutive encounters gets **no** change step (gap 0).
- A Markov change model must apply one transition per encounter, including the one being entered (predict, then
  update). With 3 entities met at random the effective λ was lower than stated; with one entity it was zero.
- Without that step, a few decisive observations (likelihood ratios > 10⁴) drive the posterior to exactly 0/1 in
  floating point. From there no evidence can revise it.

**The fix:** `TypeStateModel(transition_each_meeting=True)` uses gap n − k + 1. With it, the calm type keeps at
least π_calm · (1 − e^(−λ)) ≈ 1.6% per encounter, and calm evidence can revise the belief
(`tests/test_k_infoseek.py`). The default stays `False`, so J and the frozen K runs reproduce exactly.

**What stands:**
- K3-X, K5-X and K3b are reported as run, with the defective memory, as the frozen protocol specified.
  - K3b is unaffected in substance: its histories have no change.
- The K1 diagnosis (information per meeting) does not depend on the memory.

**Programme J:**
- J4, J5 and J6 used the same class, so their effective forgetting was weaker than the stated λ.
- J's λ was selected empirically with the same code, so the runs are self-consistent.
- But J's "never revises pushy → calm" may be partly caused by this defect, not only by missing information.
- J's records are not edited. This note is the correction.

**Corrective test (fixed before it runs):**
- K3-X2 (NERVA) and K5-X2 (wheeled), on fresh indices 200–249.
- Same systems, histories, task, type models and criteria P1–P5 as `PROTOCOL.md` §4 and §5b, with
  `--transition-fix`.
- Both the original and the corrective results are reported. Recommendations use the corrective one, because the
  defect is in the implementation, not in the design being tested.
