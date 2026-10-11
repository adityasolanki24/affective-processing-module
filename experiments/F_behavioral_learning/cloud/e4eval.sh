#!/bin/bash
# E4 test evaluation only (partition X, N = 400, prereg/E4_ADDENDUM.md) on the 40 networks trained on the first
# confirmatory VM (metadata `apm-input-run`). Split off from confirmatory2.sh so it does not race that VM's hard cap
# (DEVIATIONS.md); only one complete evaluation is used.
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
L=runs/logs_X; mkdir -p $L
git status --porcelain --untracked-files=no
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
IN=gs://$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)/apm-f/$(curl -sf -H Metadata-Flavor:Google $MD/apm-input-run)/out/runs/E4/X
mkdir -p runs/E4/X && gcloud storage cp "$IN/*_s?.json" "$IN/*_s?.pt" runs/E4/X/
ls runs/E4/X | wc -l
python $F/e4_eval.py --partition X --n 400 --models runs/E4/X --seeds 0,1,2,3,4 --workers $P > $L/E4_eval_X.log 2>&1
echo e4eval done
