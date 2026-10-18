#!/bin/bash
# K6: train a walking (joystick) policy for the Booster T1 humanoid with mujoco_playground's own PPO recipe, on a GPU VM.
# Output: out/humanoid/{t1_params.pkl, t1_meta.json, train_log.txt}. Never part of the APM core; a motor controller.
set -e
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done   # driver install can lag the boot
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv /work/hvenv
/work/hvenv/bin/pip install -q --upgrade pip
/work/hvenv/bin/pip install -q "jax[cuda12]==0.5.3" playground==0.0.4 2>&1 | tail -2 || /work/hvenv/bin/pip install -q "jax[cuda12]" playground
/work/hvenv/bin/python -c "import jax; d = jax.devices(); print('devices', d); assert d[0].platform == 'gpu'"
mkdir -p out/humanoid
/work/hvenv/bin/python experiments/K_recovery/humanoid/train_t1.py --out out/humanoid 2>&1 | tee out/humanoid/train_log.txt
