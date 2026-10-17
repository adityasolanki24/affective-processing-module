#!/bin/bash
# Second confirmatory VM (DEVIATIONS.md 2026-10-11, "E3/E7/E4-eval refused on the first VM"): E7, E3, E3 retention and
# the E4 test evaluation with the sizes of the committed addenda, unchanged. The E4 networks are the ones trained on
# the first VM (run id in metadata `apm-input-run`); evaluation starts when all 40 training records are in the bucket.
# Logs go to runs/logs_X/ (untracked): writing to a force-added log under runs/ dirtied the tree on the first VM.
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
N_E3=60; N_E3_RET=60; N_E7=100; N_E4=400
L=runs/logs_X; mkdir -p $L
git status --porcelain --untracked-files=no
python $F/e7_run.py --partition X --n $N_E7 --workers $P > $L/E7_X.log 2>&1
python $F/e3_run.py --partition X --n $N_E3 --workers $P > $L/E3_X.log 2>&1
python $F/e3_run.py --partition X --n $N_E3_RET --mode retention --workers $P > $L/E3_retention_X.log 2>&1
IN=gs://$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/apm-bucket)/apm-f/$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/apm-input-run)/out/runs/E4/X
until [ "$(gcloud storage ls "$IN/*_s?.json" 2>/dev/null | wc -l)" -ge 40 ]; do sleep 30; done
sleep 150  # one more sync period of the first VM so every final checkpoint has landed
mkdir -p runs/E4/X && gcloud storage cp "$IN/*_s?.json" "$IN/*_s?.pt" runs/E4/X/
python $F/e4_eval.py --partition X --n $N_E4 --models runs/E4/X --seeds 0,1,2,3,4 --workers $P > $L/E4_eval_X.log 2>&1
echo confirmatory2 done
