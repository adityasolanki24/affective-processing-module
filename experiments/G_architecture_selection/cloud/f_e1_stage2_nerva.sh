#!/bin/bash
# Programme F, E1 stage 2, NERVA arm (rerun): the first stage-2 VM lost NERVA partition T to a race (16 workers each
# cloning MuJoCo Menagerie on first import of mujoco_playground). Same procedure, with a single-process warm-up import
# before the pool. Experience generation is deterministic (stage 1 reproduced F's files bit for bit).
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/E1v2 runs/nerva_policy /work/nerva_in /work/NERVA
python -c "import json; from apm_select.provenance import confirmatory_lock; \
json.dump(confirmatory_lock('experiments/F_behavioral_learning/prereg/E1_ADDENDUM.md'), \
open('runs/E1v2/stage2_nerva_provenance.json', 'w'), indent=1)" || { echo "LOCK REFUSED"; exit 1; }
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
gcloud storage cp "gs://$BUCKET/apm-g/inputs/nerva/*" /work/nerva_in/
tar xzf /work/nerva_in/nerva.tar.gz -C /work/NERVA
cp /work/nerva_in/candidate.onnx runs/nerva_policy/candidate.onnx
git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/Open_Duck_Playground
git -C /work/Open_Duck_Playground checkout -q b9be205ac64488c23504ca42e5ec790337adeec3
python3 -m venv /work/nvenv
/work/nvenv/bin/pip install -q --upgrade pip
/work/nvenv/bin/pip install -q -r /work/nerva_in/requirements-inference.lock.txt
/work/nvenv/bin/pip install -q --no-deps -e /work/Open_Duck_Playground -e /work/NERVA
export OPEN_DUCK_ROOT=/work
/work/nvenv/bin/python -c "import mujoco_playground, playground.open_duck_mini_v2.mujoco_infer" || exit 1
for spec in T:480 V:160 X:160 S:120; do
  p=${spec%%:*}
  /work/nvenv/bin/python $F/nerva_generate.py --nerva /work/NERVA --policy runs/nerva_policy/candidate.onnx \
    --partition $p --trials ${spec##*:} --out runs/E1v2/nerva/$p --workers $P || exit 1
  python $F/e1_data.py --embodiment nerva --partition $p --experience runs/E1v2/nerva/$p --out runs/E1v2 --workers $P
done
python - <<'PY'
import json, pathlib
for p in ("T", "V"):
    a = json.loads(pathlib.Path(f"runs/E1/nerva_{p}.manifest.json").read_text())
    b = json.loads(pathlib.Path(f"runs/E1v2/nerva_{p}.manifest.json").read_text())
    print("MANIFEST nerva", p, "sha_equal", a["sha256"] == b["sha256"])
PY
python $F/e1_train.py --embodiment nerva --target stability_loss --data runs/E1v2 --out runs/E1v2/predsX \
  --parts T,V,X,S --workers $((P / 2))
rm -f runs/E1v2/*.pkl
echo e1_stage2_nerva done
