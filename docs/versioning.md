# Versioning and release policy

- **Package:** semantic versioning `MAJOR.MINOR.PATCH`. Before 1.0, minor versions may change Python APIs; the
  changelog says so.
- **AffectLink:** independent version `affectlink/<major>.<minor>` in every message header.
  - Minor: only additive optional fields; receivers ignore unknown fields; field numbers are never reused.
  - Major: breaking change; a receiver rejects other majors; adapters migrate explicitly.
- **Models:** every affect model or learned predictor has a `model_version` string reported in outputs. A learned
  model is promoted only after frozen evaluation (docs/learning.md) and keeps a rollback target.
- **Saved state:** `state_version`; incompatible state is never loaded partially (`load_or_new`).
- **Releases:** tagged from `main` after CI passes; private until a licence and public release are decided.
