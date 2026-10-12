# Programme H research log

Chronological and factual. Pilots (partition P) are exploratory. Confirmatory runs are named with their
addendum and commit.

## 2026-10-12

### Setup and audit
- Branch `research/apm-v0.4-expressive-intelligence` created from `research/apm-v0.3-architecture-selection`
  (`5fb2dd7`).
- Audit of APM v0.3 and NERVA (`reports/H0_AUDIT.md`).
- **NERVA environment.** NERVA's Linux venv is unusable on Windows, so a fresh Windows venv was built outside the
  repository, pinned to Open Duck's inference lock:
  - mujoco 3.3.0, onnxruntime 1.21.0, numpy 2.5.3, jax 0.5.3;
  - Open_Duck_Playground `b9be205a` (editable install), NERVA `726c8c6` (editable install; the tree stays clean).
  - The first import cloned MuJoCo Menagerie (1.7 GB, ≈ 35 min, one-off).
- **Benchmark.** 20 s of neutral-candidate walking takes 1.3–1.7 s on one core. Pools of 3 workers give the best
  throughput.
- **Cloud audit (read-only):**
  - no instances, disks, addresses, snapshots or images;
  - one Router + Cloud NAT gateway that programme F/G did not create, reported to the user and not touched;
  - no programme-H spending authorization.

### Gate reproductions (NERVA's own runner and protocol, unchanged)
- Neutral turn-translation candidate and E1′ at e = 0 both pass the full neutral long motor gate:
  - 160 trials each;
  - no falls, all pushes recovered, both latency conditions.
- **All 320 trial records are byte-identical to NERVA's recorded Linux results.**

### E1′ style reproduction
- E1′ style evaluation reproduction started (NERVA `e1prime_eval.py`).

### `apm_express` package
Written as a removable layer, with 11 unit tests:
- `ExpressionIntent`;
- `BipedExpressionAdapter`;
- `SafetyMonitor`;
- `NervaBiped` controller wrapper;
- kinematic stand-in;
- episode runner.

### H0 pilots (partition P)
- **Switching** (48 trials):
  - random command switching every 0.2–1.0 s, monitor in shadow;
  - 0 falls for the neutral candidate;
  - **1 fall for E1′** (dwell 0.4 s, latency on; tilt reached 25.5° and the trip came too late in shadow mode).
- **Head channel:**
  - NERVA's contract rejects any non-zero head offset in `set_behaviour`, so the wrapper validates the command with
    zero offsets and then sets offsets on the motor targets only;
  - smoke test at rest: yaw +0.4 rad turns the face +0.45 rad, with no drift;
  - smoke test walking forward: the neutral policy almost stops (0.005 m/s), and E1′ slows to 0.046 m/s and turns
    (yaw RMS 0.25).
  - This matches NERVA's RQ1c on the upstream policy.
  - **Design response:** a head channel used only while resting, with the head returned to neutral before any
    locomotion command, plus controller-level rate limits on head and pitch changes.

### H0 confirmatory run and outcome (later the same day)
- **H0 confirmatory run on X** (commit `2ecad67`, 1,240 trials).
- **Gate FAILS** on (d) standing pushes and (e) slow-probe arrival. Passes on (a) reproduction, (b) dwell 0.4 s and
  (c) channels.
- **Head box admitted:** ±0.4 × ±0.4 rad at rest. Face-down 0.4 with yaw −0.8 falls.
- **E1′:** 2/40 falls at 0.4 s dwell; its pitch changes were safe in 160/160 trials.
- **E1′ style evaluation reproduced:** 378/378 trials identical.
- **Diagnosis:**
  - all 0.30 m/s falls happened while standing during expressive pauses, pushed from behind;
  - a standing-push sweep (P) shows the neutral controller falls from ≥ 0.20 m/s backward pushes at rest — never
    covered by NERVA's gate;
  - the adapter had no progress guarantee.
- **Fixes, under H0b (`prereg/H0b_ADDENDUM.md`):** progress guarantee, minimum burst, push requirement at the
  standing-validated magnitude.
- **Spatial-feature sway bug fixed.**
- **H2 protocol drafted** (pending ethics).
- **Power simulation run** (design only).
