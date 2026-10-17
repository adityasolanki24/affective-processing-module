#!/bin/bash
# G3 stage B (partition X), prereg/G3_ADDENDUM.md (incl. amendment): predictive test on exploration lifetimes and the
# closed-loop test. Trained models come from stage A (input run in metadata `apm-input-run`).
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G3 runs/G3/models
git status --porcelain --untracked-files=no
python -c "from apm_select.provenance import confirmatory_lock; confirmatory_lock('experiments/G_architecture_selection/prereg/G3_ADDENDUM.md')" || exit 1
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
IN=gs://$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)/apm-g/$(curl -sf -H Metadata-Flavor:Google $MD/apm-input-run)/out/runs/G3/models
gcloud storage cp "$IN/*_s?.pt" "$IN/*_s?.json" runs/G3/models/
python $S/g3_data.py --partition X --n 100 --out runs/G3/data --workers $P > runs/logs_G3/dataX.log 2>&1
python $S/g3_predict.py --data runs/G3/data --models runs/G3/models --part X --workers 30 > runs/logs_G3/predictX.log 2>&1 &
python $S/g3_closed.py test --partition X --models runs/G3/models --n 200 --workers $((P - 30)) > runs/logs_G3/X.log 2>&1
wait
rm -f runs/G3/data/X.pkl
echo g3_stageB done
