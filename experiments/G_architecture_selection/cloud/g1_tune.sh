#!/bin/bash
# Phase 1a: G1 equal-budget tuning (V) and R5 ES (T, final evaluation on V), concurrently. No X/S data.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G1
python $S/g1_es.py --k 48 --n 30 --episodes 10 --workers 16 > runs/logs_G1/es.log 2>&1 &
python $S/g1_tune.py --k 48 --n 30 --episodes 10 --workers $((P - 16)) > runs/logs_G1/tune.log 2>&1
wait
mkdir -p runs/G1/results_out && cp -r experiments/G_architecture_selection/results/G1 runs/G1/results_out/ \
  && cp experiments/G_architecture_selection/configs/G1_selected_partial.json runs/G1/results_out/
echo g1_tune done
