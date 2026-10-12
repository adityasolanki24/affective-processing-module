# Programme H experiment matrix

| exp | question | arms | partitions | compute | status |
|---|---|---|---|---|---|
| H0 | safe expressive-action interface | NERVA gate reproduction (neutral, E1′); E1′ style reproduction; switching dwell; pitch steps; head envelope; fallback; nominal adapter | P pilot → X confirmatory (`prereg/H0_ADDENDUM.md`) | local | **gate FAILED** (d standing pushes, e arrival); reproductions byte-identical; see `reports/H0_REPORT.md` |
| H0b | interface re-test after fixes | standing-push map; fallback at m*; nominal with progress guarantee | X, fresh indices (`prereg/H0b_ADDENDUM.md`) | local | **gate PASSES** (m* = 0.15 m/s standing); see `reports/H0b_REPORT.md` |
| H1 | movement repertoire | X0, X1 (+ matched speed), X2, push, switch, shift, E1′ arm; X3 not run (no reference data) | V calibration, X test, S shift | local | **gate FAILED on G1** (1/112 push falls at 0.15 m/s after a stop); G2–G5 pass (distinctness 1.00); `reports/H1_REPORT.md` |
| H1b | push re-test after the stop-transition finding | transition push sweep; push arm at m** | X, fresh indices (`prereg/H1b_ADDENDUM.md`) | local | addendum committed; running |
| H2 | human perception | protocol (`prereg/H2_PROTOCOL.md`), power simulation, anonymised stimulus videos | — | local rendering | protocol drafted; **no participants; pending ethics determination and user authorization** |
| H3 | internal-state representations | S0–S7 | T train, V tune, X test | local | not started |
| H4 | experience shapes expression | paired lifetimes with different histories + 8 causal controls | T/V/X | local; cloud only if approved | not started (needs H1) |
| H5 | learned expression mapping | L0–L5 | T/V/X | local | gated by H1 and H4 |
| H6 | dispositions | long lifetimes, stable vs changing worlds | X | local | gated |
| H7 | cross-embodiment | biped vs wheeled adapters; 4 system variants | X | local | gated |
| H8 | integrated evaluation | A0 + memory + fixed lookup; A0 + learned mapping; gate-passing system; ablations | X | local | gated |
