#!/bin/bash
# Phase 1b: G1 confirmatory test (X, N = 280) and shift conditions (S, 4 × 100), as fixed in prereg/G1_ADDENDUM.md.
set -x
S=experiments/G_architecture_selection/scripts
C=experiments/G_architecture_selection/configs/G1_selected.json
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G1
git status --porcelain --untracked-files=no
python $S/g1_run.py --partition X --n 280 --systems $C --workers $P > runs/logs_G1/X.log 2>&1
python $S/g1_run.py --partition S --n 100 --systems $C --conditions nominal,miscal_x3,dense,sparse --no-controls \
  --workers $P > runs/logs_G1/S.log 2>&1
echo g1_test done
