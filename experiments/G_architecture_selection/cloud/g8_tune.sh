#!/bin/bash
# G8 equal lambda retuning of every A0 variant (partition V only), prereg/G8_ADDENDUM.md.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G8/results_out
python $S/g8_tune.py --variants experiments/G_architecture_selection/configs/G8_variants.json --k 48 --n 30 --workers $P > runs/G8/tune.log 2>&1
cp experiments/G_architecture_selection/configs/G8_selected.json experiments/G_architecture_selection/results/G8/tuning_table.json runs/G8/results_out/
echo g8_tune done
