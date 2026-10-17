# Programme H status (2026-10-13)

## Done

- H0 → H0b, H1 → H1b, H3, H4 → H4b, H5, H6, H7 and H8. All failures and deviations are on record.
- **Decision B:** `EXPRESSIVE_ARCHITECTURE_DECISION.md`.

## Open

1. **H2 human study.**
   - Ready: protocol, power simulation, 72 stimulus clips (`videos/h1/`).
   - Needs an ethics determination by the researcher's institution and the user's written authorization.
   - No participants have been contacted.
   - A researcher self-pilot (2026-10-17, not blind; `RESEARCH_LOG.md`) found the styles hard to tell apart
     (2/6 recognised) and exposed instrument problems: midpoint answers, slider anchoring, fatigue.
   - Survey instrument v0.1 is in `survey/`. Before recruiting, a richer NERVA motion vocabulary is probably
     needed, not just a better survey.
2. **NERVA controller push robustness while standing or just stopped.** The certified envelope during expression is
   0.10 m/s. This is reported for NERVA and was not changed here.
3. **Combination weight for dispositions in the memory-centred lookup.**
4. **Context keying's pooled-shuffle control** was slightly non-null in H8 (+0.09 [0.01, 0.16]). Re-check before
   adopting it.
5. **Closed-loop experience** (expression changing exposure) and other outcome types.

## Environment notes

- The NERVA simulator runs from a separate Windows virtual environment (Open Duck inference lock), with `NERVA_ROOT`
  set.
- Large evaluation JSON files are committed gzip-compressed. Loaders accept either form.
- No cloud resources were used in programme H.
- One pre-existing Cloud Router/NAT gateway (not created by programmes F, G or H) was found in the configured
  project and reported; it was not touched.
