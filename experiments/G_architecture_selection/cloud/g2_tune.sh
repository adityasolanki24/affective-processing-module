#!/bin/bash
# G2 phase A (partitions T and V only): M6 training logs on G2-T, M6 training, memory tuning and validation selection.
set -x
S=experiments/G_architecture_selection/scripts
C=experiments/G_architecture_selection/configs
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G2 runs/G2/results_out
python - <<'PY'
import json
reg = json.load(open("experiments/G_architecture_selection/configs/G1_selected.json"))
best = {k: v for k, v in reg["systems"][reg["best_simple"]].items() if k not in ("config_id", "label")}
json.dump({"systems": {"logger": {**best, "memory": {"model": "M0", "evidence": "exposed"}}}},
          open("runs/G2/logger_specs.json", "w"))
PY
python $S/g2_run.py --partition T --n 20 --systems runs/G2/logger_specs.json --workers $P > runs/logs_G2/T.log 2>&1
python $S/g2_m6_train.py --logs runs/G2/T/lifetimes.jsonl > runs/logs_G2/m6.log 2>&1
python $S/g2_tune.py --regulator $C/G1_selected.json --k 16 --n 6 --nv 12 --m6 \
  experiments/G_architecture_selection/results/G2/m6_weights.json --workers $P > runs/logs_G2/tune.log 2>&1
cp -r experiments/G_architecture_selection/results/G2 runs/G2/results_out/ && cp $C/G2_selected.json runs/G2/results_out/
echo g2_tune done
