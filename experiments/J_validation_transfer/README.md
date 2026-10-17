# Programme J: APM v0.6, validation, NERVA transfer and closed-loop expressive development

The programme does three things:
- validates programme I's revising memory: independent calibration (J1) and distribution shifts (J2);
- transfers the memory-learning principle to the NERVA biped, using robot-observable evidence (J3, J4);
- tests the full loop: experience → belief → expression → changed interaction → evidence → revised belief (J5, J6).

| file | contents |
|---|---|
| `MASTER_PREREGISTRATION.md` | frozen design, gates, criteria, decision options |
| `EXPERIMENT_MATRIX.md` | experiments, status, gates |
| `prereg/` | addenda committed before the runs they govern |
| `DEVIATIONS.md`, `RESEARCH_LOG.md` | departures; what was done and when |
| `reports/` | phase reports, `COMPUTE_LOG.md`, the final recommendation |
| `seeds.py` | the only source of J world seeds |
| `scripts/` | `j_common.py`/`j_run.py` (wheeled runner), `j1_*` (J1), and later `j2_*`, `j3_*`, … |
| `cloud/` | `jcloud.py` + `runner.sh`: capped spot VMs with a bucket job queue (no SSH: the organization enforces OS Login) |

**Code:** `src/apm_transfer/`, a removable package. It builds on `apm_revise` and `apm_express` without changing them.

## Reproducing a result

Each run's `run_meta.json` records the commit, source hashes and addendum. Check out that commit and rerun the
command in the addendum. Local:

```
PYTHONPATH=src python experiments/J_validation_transfer/scripts/j_run.py ...
```

Cloud:

```
python experiments/J_validation_transfer/cloud/jcloud.py submit NAME "<command>" --commit
```

This needs the env vars `APM_GCP_PROJECT`, `APM_GCP_BUCKET` and `APM_GCP_SERVICE_ACCOUNT`, which are never committed.
