# Embedded hardware track (design only: nothing purchased, no board tested)

**Status:** no embedded board was available, so nothing here is a hardware measurement.
- Resource numbers marked **[measured, dev PC]** come from `experiments/E_hardware/benchmark.py` on the
  development PC (`experiments/E_hardware/results.json`).
- Board figures marked **[estimate]** are scaled from those measurements.
- Specifications marked **[vendor]** are published vendor claims, not verified here.

## What the software needs

Measured with `experiments/E_hardware/benchmark.py` on the development PC: x86-64, 12 logical CPUs, Windows 11,
Python 3.12, PyTorch 2.14 CPU, single thread for the GRU.

| workload | measured on dev PC | embedded estimate |
|---|---|---|
| APM runtime step: busy scene with 10 tracked entities, 6 live hypotheses, 18 messages per tick, populated memory | median **0.32 ms**, p99 0.73 ms, max 1.3 ms | Cortex-A76 (Pi 5): ~2–4× slower → ≈ 1–3 ms **[estimate]**; Cortex-A53 (i.MX 8M Plus): ~6–12× slower → ≈ 2–9 ms **[estimate]**. Either is ample for a 10–20 Hz affect loop |
| message ingest (validation, ordering, routing) | **≈ 107,000 messages/s** | ≥ 10,000 messages/s on A53 **[estimate]**; robots send ~10²/s |
| APM memory | Python objects allocated by APM over a busy 300 s run: **0.17 MB** (tracemalloc; excludes the interpreter and NumPy); saved state **76 kB** with 51 entity records and 150 episodes | process RSS dominated by the Python runtime (tens of MB) **[estimate]**; a C++/Rust daemon would be a few MB |
| GRU inference, 17,857 parameters, batch 1 | median **0.52 ms**, p99 1.8 ms, PyTorch eager, 1 thread. In the Milestone C run, with 10 threads under load: 2.3 ms | A76: ≈ 1–2 ms; A53: ≈ 3–6 ms with PyTorch; ONNX Runtime/TFLite expected lower **[estimate]** |
| GRU training (shadow update) | **≈ 3,400 samples/s**, 1 thread. A Milestone C shadow update (≈ 18k samples × 5 epochs) ≈ 26 s | A76: ≈ 1–2 min per update; A53: ≈ 4–8 min **[estimate]**: acceptable for an offline or idle-time learner |
| model size | 75 kB serialized state | fits any candidate's storage trivially |

## Candidate classes

| | Raspberry Pi 5-class | NXP i.MX 8M Plus-class SoM | Jetson Orin Nano-class |
|---|---|---|---|
| CPU | 4× Cortex-A76 @ 2.4 GHz **[vendor]** | 4× Cortex-A53 @ up to 1.8 GHz + Cortex-M7 **[vendor]** | 6× Cortex-A78AE **[vendor]** |
| accelerator | none on board (optional inference HATs) | 2.3 TOPS NPU, **INT8 inference only** **[vendor]** | Ampere GPU, CUDA; supports PyTorch training **[vendor]** |
| on-device training of the GRU | CPU: yes **[estimate]** | CPU only (A53, ~3–5× slower than A76) **[estimate]** | GPU or CPU: yes |
| RAM | 4–16 GB LPDDR4X **[vendor]** | module-dependent, commonly 2–8 GB LPDDR4 **[vendor]** | 4 / 8 GB LPDDR5 **[vendor]** |
| storage / endurance | microSD (poor endurance) or NVMe via PCIe **[vendor]**; use NVMe/eMMC with wear-aware logging | eMMC on most SoMs **[vendor]**; industrial-grade eMMC options | NVMe **[vendor]** |
| power | roughly 3–12 W depending on load **[estimate]** | roughly 2–5 W for the module **[estimate]** | 7–25 W modes **[vendor]** |
| heat / cooling | active cooler recommended under sustained load **[vendor]** | passive feasible at the module's typical power **[estimate]** | heatsink + fan **[vendor]** |
| size / mass | 85 × 56 mm board **[vendor]** | small SoM on a custom or vendor carrier | 70 × 45 mm module on a carrier **[vendor]** |
| boot to service | Linux: ~10–20 s **[estimate]** | Yocto: ~5–15 s **[estimate]**; M7 can be up in < 1 s for a link heartbeat **[estimate]** | ~20–30 s **[estimate]** |
| Ethernet / CAN FD / UART | GbE, UART native; CAN FD via SPI controller (e.g. MCP2518FD HAT) **[vendor]** | 2× GbE (TSN-capable), 2× CAN FD, UART native **[vendor]** | GbE, UART; CAN availability is carrier-dependent (verify) |
| security / model integrity | no secure element by default; add a TPM/secure element | secure boot (HAB) and crypto engine on the SoC **[vendor]** | secure boot support **[vendor]** |
| maintained runtimes | Raspberry Pi OS / Debian; PyTorch CPU, ONNX Runtime | Yocto/Debian BSPs; ONNX Runtime/TFLite with NPU delegates | JetPack: CUDA, PyTorch, TensorRT |

**Do not choose by TOPS.** The APM workload is small, latency-tolerant (10–50 ms) and needs **on-device training**
(the shadow learner), which NPUs don't provide.

- **Pi 5:** fastest, easiest bench development path.
- **i.MX 8M Plus:** best embedded prototype for power, native CAN FD/TSN, secure boot and the M7 for a link
  heartbeat. Training runs on its CPU.
- **Jetson Orin Nano:** research fallback for much larger learned models, at higher power.

Not designed now: a custom ASIC.

## Block diagram

```
 Robot host (owns task + motor control + safety; works without APM)
     │  AffectLink frames (Ethernet: protobuf/TCP+UDP │ CAN FD: ISO-TP compact profile │ UART: COBS+CRC)
     ▼
 ┌─────────────────────────── APM module (SoM + carrier) ───────────────────────────┐
 │ link interface (+ heartbeat; on i.MX: M7 core)                                     │
 │ APM runtime: StreamGate · context · appraisal · affect · tendencies · diagnostics  │
 │ validated predictor (ONNX/TFLite)  ◄── promotion ──  shadow learner (CPU)          │
 │ memory store (entity/context statistics, episodes; atomic writes)                 │
 └──────────────┬───────────────────────────────────────────────────────────────────┘
                │ eMMC/NVMe (A/B model slots, journaling, signed models)
 power: wide-input DC/DC from robot bus → reverse-polarity + over-current protection → brown-out detect
        (hold-up capacitance sized for one atomic state write)
```

## Failure behaviour

Required: the robot keeps its ordinary safe behaviour without APM. See docs/safety.md for the full table.

| event | design response |
|---|---|
| APM loses power | the host times out on missing heartbeats/advisory outputs (≤ 0.5 s) and ignores APM. On power-up the APM restores the last atomically saved state, opens a new epoch and waits for fresh self state |
| network/CAN cable unplugged | the host behaves as above; the APM marks inputs stale, affect relaxes, and no memory updates come from absent data |
| APM crash | watchdog restart; same as power loss |
| late messages | rejected beyond `max_age_s`; counted in diagnostics |
| storage corruption | versioned, checksummed state; a failed load starts with empty memory and reports it (`load_or_new`); A/B slots for models |
| neural learner fails | the validated model keeps serving; shadow discarded; rollback available |
| host reboots | `Reset` with a new epoch; APM clears that context, keeps memory |

## Bench test procedure (for when hardware exists)

1. Flash the image; measure boot-to-first-AffectState time and idle/typical/peak power at the DC input.
2. Replay recorded AffectLink logs (NERVA replay, wheeled demo) at 20 Hz for 1 h. Record step latency
   (median/p99), CPU %, RSS and SoC temperature.
3. Run one shadow update on the stored experience buffer. Record wall time, peak memory and energy.
4. Fault injection: pull power during a state write (×20); unplug the link; inject corrupt state; kill the learner.
   Verify the robot-side timeout behaviour and the APM recovery behaviour.
5. Interface tests: CAN FD ISO-TP at the compact-profile rate under simulated motor-bus load; UART at 115,200 baud.

## Remaining unverified

Everything about physical boards: power, thermals, boot, real CAN/UART throughput, ONNX/TFLite conversion of the
GRU, and on-device training speed.
