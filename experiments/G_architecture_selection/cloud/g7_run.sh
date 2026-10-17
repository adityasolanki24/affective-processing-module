#!/bin/bash
# G7 lifelong study (prereg/G7_ADDENDUM.md).
set -x
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G7
git status --porcelain --untracked-files=no
python experiments/G_architecture_selection/scripts/g7_lifelong.py --partition X --n 150 --workers $P > runs/G7/X.log 2>&1
echo g7 done
