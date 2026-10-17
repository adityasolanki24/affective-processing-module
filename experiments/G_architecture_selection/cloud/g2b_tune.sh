#!/bin/bash
# G2b tuning and validation selection (partition V only).
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G2b runs/G2b/results_out
python $S/g2b_tune.py --k 16 --n 6 --nv 12 --workers $P > runs/logs_G2b/tune.log 2>&1
cp experiments/G_architecture_selection/results/G2/g2b_tuning.json experiments/G_architecture_selection/configs/G2b_selected.json runs/G2b/results_out/
echo g2b_tune done
