#!/bin/bash
# G8 stage 1 (partition X, N = 280, 10 episodes), prereg/G8_ADDENDUM.md.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G8
git status --porcelain --untracked-files=no
python $S/g1_run.py --partition X --n 280 --systems experiments/G_architecture_selection/configs/G8_test_systems.json \
  --no-controls --experiment G8 --addendum G8_ADDENDUM.md --out runs/G8 --workers $P > runs/G8/X.log 2>&1
echo g8_test done
