#!/bin/bash
# G2b test (partition X, indices 1000-1059 per scenario family), prereg/G2b_ADDENDUM.md.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G2b
git status --porcelain --untracked-files=no
python $S/g2_run.py --partition X --first 1000 --n 60 --systems experiments/G_architecture_selection/configs/G2b_selected.json \
  --addendum G2b_ADDENDUM.md --out runs/G2b --workers $P > runs/logs_G2b/X.log 2>&1
echo g2b_test done
