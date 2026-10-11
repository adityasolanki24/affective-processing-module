# AffectLink v1

AffectLink is the contract between a robot and an Affective Processing Module. It standardizes the **meaning** of
the information exchanged (entities, self state, goals, predictions, outcomes, cues, and the module's advisory
outputs), not sensors, joint names, simulators, or motor protocols.

- Normative schema: [`proto/affectlink/v1/affectlink.proto`](../proto/affectlink/v1/affectlink.proto).
- Python reference implementation: `apm.protocol` (`messages.py`, `codec.py`, `validation.py`).
- `tests/test_proto_sync.py` checks that the two match field by field.

## Why Protocol Buffers as the canonical schema

- **Explicit field numbers** make additive evolution safe: new optional fields don't break old receivers, and
  numbers are never reused.
- **Language-neutral code generation:** C/C++, Python and Rust.
- **A compact binary wire format** for constrained links.
- **Mature embedded runtimes** such as nanopb, for microcontrollers and CAN/UART bridges.
- **A direct mapping onto ROS 2 `.msg`/IDL types.**
- The canonical proto3 JSON mapping gives a readable local format for logs and debugging.

The local Python transport uses that JSON mapping and needs no protobuf runtime. Alternatives (JSON Schema,
FlatBuffers, ROS IDL alone) either lack compact binary encoding with field numbering or tie the core to one
middleware.

## Message families

| direction | message | meaning |
|---|---|---|
| in | `EmbodimentProfile` | capability metadata: which self fields and outcome kinds this robot provides, margin calibrations |
| in | `EntityObservation` | a tracked entity: temporary track id, optional persistent identity and its confidence, type, bearing/distance/closing speed, detection confidence, provenance |
| in | `RelationObservation` | subject–relation–object fact with confidence |
| in | `SelfState` | energy, thermal, stability/control margins, actuation stress, contact slip, sensor health, mode; each a `Quantity` |
| in | `GoalState` | goal id, type, priority, target, progress, active |
| in | `OutcomeHypothesis` | a **prediction**: kind, target, affected goal/self variable, probability, horizon, expected magnitude, confidence, action context |
| in | `OutcomeSignal` | an **observation**: kind, magnitude, target, affected variable, evidence quality, confirmed/estimated, action context |
| in | `SemanticCue` | an interpreted speech/scene meaning with speaker, target, confidence and model provenance |
| in | `Reset` | the sender restarted (new epoch) |
| out | `AffectState` | persistent PAD readout, baseline, uncertainty, model version |
| out | `AppraisalFrame` | appraisal of one referenced hypothesis, outcome or memory prediction |
| out | `ActionTendency` | per-target advisory approach/avoid/orient/explore/disengage/seek-assistance in [0, 1] |
| out | `AttentionBias` | target, priority, expiry |
| out | `ExpressionIntent` | activation, posture, movement energy, openness, urgency |
| out | `MemoryExpectation` | per target: expected adverse/benign outcome probability, familiarity, predictability, confidence |
| out | `Diagnostics` | input ages, saturation, missing capabilities, memory updates, health, rejected messages |

Outcome kinds come from an embodiment-agnostic catalog (`apm.core.outcomes`), for example:
`collision`, `fall`, `stability_loss`, `control_margin_degraded`, `resource_critical`, `task_interrupted`,
`task_completed`, `task_progress`, `recovery`, `benign_contact`, `assistance_received`.

A robot declares which kinds it can detect. The core rejects undeclared kinds rather than guessing.

## Protocol rules

1. **Versioning.** `Header.schema_version` is `affectlink/<major>.<minor>`.
   - Receivers reject a different major version.
   - Unknown fields from newer minor versions are ignored.
   - Field numbers are never reused.
2. **Timestamps.** `stamp_s` is seconds on the session clock, monotonic within an epoch.
   - Adapters convert their own clocks before sending.
   - The APM integrates affect over stamp differences, so its dynamics don't depend on message rate.
3. **Units and uncertainty.** Every physical value is a `Quantity` carrying a value, a unit (SI symbol or
   `fraction`), an optional standard deviation and a status.
4. **Missing data.** If a status is not `VALID`/`ESTIMATED`, the value must be absent; validation rejects a value
   attached to `MISSING`.
   - The core treats missing as unknown, raising uncertainty, and never as zero or as nominal.
   - `UNSUPPORTED` marks fields an embodiment never provides.
5. **Embodiment calibration.** A raw quantity, for example battery volts, becomes a margin fraction only through
   the profile's `MarginCalibration`, which gives the nominal value, the critical value and their meaning for that
   body.
   - A raw value without a matching calibration is ignored, not guessed.
   - A margin of 0.5 means "halfway between nominal and critical *for this robot*", not the same physical state on
     a biped and an arm.
6. **Prediction ≠ outcome.** These are different message types.
   - Memory learns only from `OutcomeSignal`.
   - A hypothesis is never treated as an observed adverse outcome.
7. **Ordering and duplicates.** For each `(source, stream)`, `(epoch, seq)` must strictly increase.
   - Duplicates and out-of-order messages are rejected and counted.
   - Outcome ids are also deduplicated, so a resent outcome counts once, including after a save/restore.
8. **Staleness.** Messages older than `max_age_s`, default 2 s, are rejected on arrival, as are messages stamped
   more than `max_age_s` in the future.
   - Context has TTLs: entities 1.5 s, self state 1 s, hypotheses until their horizon.
   - Only the periodic `SelfState` stream determines the `stale_inputs` health state; event streams may
     legitimately go quiet.
9. **Reset semantics.** A sender that restarts increments its epoch and sends `Reset`.
   - The receiver clears that source's ordering state and its context.
   - Learned memory is kept.
   - Messages from older epochs are rejected.
10. **Size limits.**
    - Local/Ethernet profile: 16 KiB per encoded message.
    - Compact profile for CAN FD/UART: messages must fit 512 bytes encoded. That requires binary protobuf, short
      ids and a bounded number of repeated fields.

## Transports

Implemented now: the **in-process Python API** (`APM.ingest` / `APM.step`) and the JSON codec (`encode`/`decode`)
for logs, files and any byte transport. The others are documented mappings, not implementations.

| transport | mapping | realistic constraints |
|---|---|---|
| Native C/C++ ABI | C structs generated from the `.proto` (nanopb or protobuf-c); `apm_ingest(const uint8_t* buf, size_t n)` / `apm_step(double now, apm_outputs* out)` with binary protobuf buffers | single-threaded entry points or an internal queue; caller owns clocks; no allocation in the hot path on MCUs |
| ROS 2 | one topic per message family (`/affectlink/self_state`, …); `.msg` files generated from the `.proto`; a bridge node owns an APM instance | must not be required by the core; QoS: reliable/keep-last for state, reliable for outcomes; SROS2 security is opt-in and must be configured |
| Ethernet (UDP/TCP, ZeroMQ or similar) | length-prefixed binary protobuf frames carrying `Header` | TCP for outcomes (must not be lost), UDP acceptable for periodic state; authenticate and integrity-check frames on shared networks |
| CAN FD | ISO-TP (ISO 15765-2) segmentation of compact-profile binary messages; one CAN id per family | 64-byte frames → multi-frame messages; ~1–5 Mbit/s data phase shared with motor traffic: send self state ≤ 20 Hz, entities only on change; outcome messages need acknowledgement |
| UART | COBS framing + CRC-32 around binary protobuf | 115 200 baud ≈ 11 kB/s: compact profile only, low rates; no implicit retransmission, so outcomes need an ack/resend scheme |

## Replaceable models behind a stable contract

AffectLink is independent of how the APM computes its outputs.
- The transparent baseline (appraisal → attractor PAD) can be replaced, or augmented, by a learned predictor
  (GRU today; a Transformer, an active-inference model or a neuromorphic implementation later) without any change
  to robots or adapters.
- `AffectState.model_version` and `Diagnostics.model_version` say which model produced a given output.
