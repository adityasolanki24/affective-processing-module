#!/bin/bash
# G9 validity re-test, rule-based families (G1, G3, G8 stages 1-2) on fresh lifetimes (prereg/G9_ADDENDUM.md).
set -x
S=experiments/G_architecture_selection/scripts
C=experiments/G_architecture_selection/configs
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/G9 runs/G3/models
git status --porcelain --untracked-files=no
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
gcloud storage cp "gs://$BUCKET/apm-g/20261012-020614-g3a/out/runs/G3/models/*_s?.pt" runs/G3/models/
A="--addendum G9_ADDENDUM.md --first 10000 --n 400 --no-controls --workers $P"
python $S/g1_run.py --partition X --systems $C/G1_selected.json --out runs/G9/G1 $A > runs/G9/G1.log 2>&1
python $S/g1_run.py --partition X --systems $C/G8_test_systems.json --experiment G8 --out runs/G9/G8 $A > runs/G9/G8.log 2>&1
python - <<'PY'
import json
c = "experiments/G_architecture_selection/configs/"
json.dump({"systems": {"A0": json.load(open(c + "G8_selected.json"))["systems"]["A0"],
                       "A0_min": json.load(open(c + "G8_selected_min.json"))["systems"]["A0_min"]}},
          open("runs/G9/g8s2_systems.json", "w"), indent=1)
PY
python $S/g1_run.py --partition S --systems runs/G9/g8s2_systems.json --experiment G8 --out runs/G9/G8 $A > runs/G9/G8S.log 2>&1
python $S/g3_closed.py test --partition X --models runs/G3/models --out runs/G9/G3 $A > runs/G9/G3.log 2>&1
echo g9_rules done
