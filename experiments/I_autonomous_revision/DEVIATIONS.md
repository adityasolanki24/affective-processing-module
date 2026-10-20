# Programme I deviations

Dated entries only; the preregistration itself is never edited.

## D1 (2026-10-17, V stage, before any X data): p_random range widened to [0.02, 1.0]

**What happened.** §4 requires the random-probing control to be *budget-matched* to the selected VOI: its mean probes
per lifetime should be as close as possible to VOI's. Within the preregistered range [0.02, 0.6], the best match was
p = 0.575, with 10.8 probes against VOI's 21.2 (`runs/I/cloud/V/out/tune_explore.json`).

**Why the range was too narrow.** Probing is decided once per encounter. On validation, VOI probes at almost every
keep-away encounter for every ν in its grid (20.2–21.7 probes). Matching that needs p close to 1.

**What changes.** p_random is re-selected with the same rule (closest mean probes to VOI's 21.2) over 16 uniform values
in [0.02, 1.0], on the same validation lifetimes. Nothing else changes, and no test data exist.

**Consequence for interpretation.** The tuning objective was nearly flat in ν. VOI here therefore behaves almost like
"probe whenever keep-away would stop the robot", within the budget. P3, targeted vs random, tests whether that
residual targeting matters. Both the original (0.575) and the re-selected value are reported.
