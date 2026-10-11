#!/bin/bash
# G6 (partition X, 150 lifetimes x 14 conditions), prereg/G6_ADDENDUM.md. Models from the runs in `apm-input-run`.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G45/models runs/logs_G6
git status --porcelain --untracked-files=no
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
for run in $(curl -sf -H Metadata-Flavor:Google $MD/apm-input-run | tr '+' ' '); do
  gcloud storage cp "gs://$BUCKET/apm-g/$run/out/runs/G45/models/*_s?.pt" "gs://$BUCKET/apm-g/$run/out/runs/G45/models/*_s?.json" runs/G45/models/
done
python $S/g6_run.py --partition X --n 150 --systems experiments/G_architecture_selection/configs/G6_systems.json \
  --models runs/G45/models --workers $P > runs/logs_G6/X.log 2>&1
echo g6 done
