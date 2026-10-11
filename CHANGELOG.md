# Changelog

All notable changes; the project follows docs/versioning.md.

## [Unreleased]
### Added
- AffectLink v1 schema (`proto/affectlink/v1/affectlink.proto`) and the Python reference implementation: messages,
  JSON codec, validation, stream gate (ordering, duplicates, staleness, reset epochs, size limits).
- APM-0 core: bounded context with calibrated margins, transparent appraisal frames, attractor affect with phasic
  impulses, per-target tendencies, attention, expression intent, diagnostics.
- Outcome-grounded memory (entity and context statistics, ambiguous identity credit, episodes) with atomic
  persistence and corrupt-state fallback.
- Standalone synthetic example; CI (ruff, mypy, pytest).
