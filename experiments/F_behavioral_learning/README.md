# F — behavioural learning programme (APM v0.2, E0–E7)

| file | content |
|---|---|
| `AUDIT.md` | repository audit, verified defects, blockers |
| `MASTER_PREREGISTRATION.md` | frozen hypotheses, systems, metrics, criteria, seeds, compute limits |
| `prereg/` | per-experiment addenda (sample sizes from pilots), committed before each confirmatory run |
| `DEVIATIONS.md` | every deviation from the preregistration |
| `seeds.py` | the only source of v0.2 world seeds (disjoint partitions) |
| `scripts/` | runners; `configs/` their configurations; `manifests/` data/source hashes; `results/` aggregate results |

Large raw data are written under `runs/` (git-ignored) and described by manifests.
