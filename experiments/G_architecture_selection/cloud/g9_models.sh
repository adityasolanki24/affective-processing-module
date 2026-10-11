#!/bin/bash
# G9 validity re-test, model-based families (G4/G5 shortlist, G6) on fresh lifetimes (prereg/G9_ADDENDUM.md).
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G9 runs/G45/models
git status --porcelain --untracked-files=no
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
for run in 20261012-023336-g45a 20261012-023826-g45b; do
  gcloud storage cp "gs://$BUCKET/apm-g/$run/out/runs/G45/models/*_s?.pt" "gs://$BUCKET/apm-g/$run/out/runs/G45/models/*_s?.json" runs/G45/models/
done
python $S/g45_eval.py --partition X --first 10000 --n 400 --only A2_bcppo,A3_bcppo,A4_ppo,A3_ppo --addendum G9_ADDENDUM.md \
  --models runs/G45/models --out runs/G9/G45 --workers $P > runs/G9/G45.log 2>&1
python $S/g6_run.py --partition X --first 10000 --n 150 --systems experiments/G_architecture_selection/configs/G6_systems.json \
  --addendum G9_ADDENDUM.md --models runs/G45/models --out runs/G9/G6 --workers $P > runs/G9/G6.log 2>&1
echo g9_models done
