# Programme I: APM v0.5, autonomous revision under self-selected exposure

**Question:** can a robot that chooses its own exposure keep its beliefs about individual people accurate when they
change? This covers the *avoidance trap*: a person the robot avoids can no longer show it that they have changed.

## Documents

| file | contents |
|---|---|
| `MASTER_PREREGISTRATION.md` | frozen design: pilot basis, systems, tuning, metrics, criteria, decisions |
| `prereg/` | addenda committed before the runs they govern |
| `DEVIATIONS.md` | departures from the preregistration |
| `RESEARCH_LOG.md` | what was done, when |
| `reports/` | results, `COMPUTE_LOG.md` (cloud spend) |
| `seeds.py` | the only source of I world seeds |

## Code

| path | role |
|---|---|
| `src/apm_revise/` | evidence coding, EM person types, change-aware beliefs, revising memory, keep-away + exploration, S10 (removable package) |
| `scripts/i_run.py`, `i_common.py` | runner on G's testbed and rule; passive evidence logger identical for every system |
| `scripts/i_fit.py`, `i_tune.py`, `i_systems.py` | type models (T), tuning (V), system files |
| `scripts/i_metrics.py`, `i_analyze.py` | metrics and preregistered criteria |
| `scripts/i0_evidence_pilot.py` | the pilot's information measurement |
| `cloud/icloud.py`, `cloud/runner.sh` | capped spot VMs with a bucket job queue (no SSH) |

Raw lifetimes are written under `runs/`, which is git-ignored.
