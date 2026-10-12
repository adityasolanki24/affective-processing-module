# Programme H experiment matrix

| exp | question | arms | partitions | compute | status |
|---|---|---|---|---|---|
| H0 | safe expressive-action interface | NERVA gate reproduction (neutral, E1′); E1′ style reproduction; switching dwell; pitch steps; head envelope; fallback; nominal adapter | P pilot → X confirmatory (`prereg/H0_ADDENDUM.md`) | local | **gate FAILED** (d standing pushes, e arrival); reproductions byte-identical; see `reports/H0_REPORT.md` |
| H0b | interface re-test after fixes | standing-push map; fallback at m*; nominal with progress guarantee | X, fresh indices (`prereg/H0b_ADDENDUM.md`) | local | **gate PASSES** (m* = 0.15 m/s standing); see `reports/H0b_REPORT.md` |
| H1 | movement repertoire | X0, X1 (+ matched speed), X2, push, switch, shift, E1′ arm; X3 not run (no reference data) | V calibration, X test, S shift | local | **gate FAILED on G1** (1/112 push falls at 0.15 m/s after a stop); G2–G5 pass (distinctness 1.00); `reports/H1_REPORT.md` |
| H1b | push re-test after the stop-transition finding | transition push sweep; push arm at m** | X, fresh indices (`prereg/H1b_ADDENDUM.md`) | local | **passes** (m** = 0.10 m/s; 0/168 falls) → **H1 passes**; `reports/H1b_REPORT.md` |
| H2 | human perception | protocol (`prereg/H2_PROTOCOL.md`), power simulation, anonymised stimulus videos | — | local rendering | protocol drafted; **no participants; pending ethics determination and user authorization** |
| H3 | internal-state representations | S0–S7, explicit memories, oracle control | T train, V tune, X test (`prereg/H34_ADDENDUM.md`) | local | **done**: selected M_ctx (smallest non-inferior); nothing beats A0 memory; PAD weakest; `reports/H34_REPORT.md` |
| H4 | experience shapes expression | paired lifetimes, causal controls, physical pairs | X | local | **preregistered gate FAILED** (mis-specified shuffle control) |
| H4b | re-test with pooled-shuffle control | fresh X streams, physical pairs | X fresh (`prereg/H4b_ADDENDUM.md`) | local | **passes** for M_id, S5, S6, S7 (claim C in simulation) |
| H5 | learned expression mapping | L0–L3 (L4 conditional, L5 not run); supervision = designer references from simulator truth | T/V/X (`prereg/H5_ADDENDUM.md`) | local | addendum committed; waits for H3/H4 data |
| H6 | dispositions | 96-encounter lifetimes (joined X streams), D1 unknown-kind caution | X (`prereg/H6_ADDENDUM.md`) | local | **done**: revisable dispositions from global states (scalar EMA simplest); weak in A0 memory |
| H7 | cross-embodiment | biped (H1) vs kinematic wheeled base, shared intent | X (`prereg/H7_ADDENDUM.md`) | local | **correspondence supported** (C1 1.00, median ρ 0.71, transfer 0.79); interpretability untested; `reports/H7_REPORT.md` |
| H8 | integrated evaluation | A0 + memory + fixed lookup; A0 + learned mapping; gate-passing system; ablations | X | local | gated |
