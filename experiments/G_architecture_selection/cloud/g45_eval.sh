#!/bin/bash
# G4/G5 test evaluation (partition X, N = 400, prereg/G45_ADDENDUM.md). Models from the training runs listed in
# metadata `apm-input-run` (run ids joined by '+').
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G45/models runs/logs_G45
git status --porcelain --untracked-files=no
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
for run in $(curl -sf -H Metadata-Flavor:Google $MD/apm-input-run | tr '+' ' '); do
  gcloud storage cp "gs://$BUCKET/apm-g/$run/out/runs/G45/models/*_s?.pt" "gs://$BUCKET/apm-g/$run/out/runs/G45/models/*_s?.json" runs/G45/models/
done
ls runs/G45/models/*.pt | wc -l
python $S/g45_eval.py --partition X --n 400 --models runs/G45/models --workers $P > runs/logs_G45/X.log 2>&1
echo g45_eval done
