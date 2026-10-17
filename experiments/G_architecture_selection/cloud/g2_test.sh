#!/bin/bash
# G2 confirmatory test (partition X, N = 60 per scenario family), as fixed in prereg/G2_ADDENDUM.md.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G2
git status --porcelain --untracked-files=no
python $S/g2_run.py --partition X --n 60 --systems experiments/G_architecture_selection/configs/G2_test_systems.json \
  --workers $P > runs/logs_G2/X.log 2>&1
echo g2_test done
