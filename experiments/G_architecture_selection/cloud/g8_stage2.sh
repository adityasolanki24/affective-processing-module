#!/bin/bash
# G8 stage 2 (prereg/G8_ADDENDUM.md): retune A0_min's lambda (K = 48) on G8-V lifetimes 100-129, then A0 vs A0_min on
# G8-S lifetimes 0-279 (non-inferiority). A0 = the stage-1 selection (configs/G8_selected.json, committed).
set -x
S=experiments/G_architecture_selection/scripts
C=experiments/G_architecture_selection/configs
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G8
git status --porcelain --untracked-files=no
python $S/g8_tune.py --variants $C/G8_min_variants.json --k 48 --n 30 --first 100 --tag _min --out runs/G8/tune_min --workers $P > runs/G8/tune_min.log 2>&1
python - <<'PY'
import json
c = "experiments/G_architecture_selection/configs/"
a0 = json.load(open(c + "G8_selected.json"))["systems"]["A0"]
mn = json.load(open(c + "G8_selected_min.json"))["systems"]["A0_min"]
json.dump({"systems": {"A0": a0, "A0_min": mn}}, open("runs/G8/stage2_systems.json", "w"), indent=1)
PY
cp $C/G8_selected_min.json experiments/G_architecture_selection/results/G8/tuning_table_min.json runs/G8/
python $S/g1_run.py --partition S --n 280 --systems runs/G8/stage2_systems.json --no-controls --experiment G8 \
  --addendum G8_ADDENDUM.md --out runs/G8 --workers $P > runs/G8/S.log 2>&1
echo g8_stage2 done
