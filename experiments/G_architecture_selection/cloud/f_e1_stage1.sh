#!/bin/bash
# Programme F, E1 completion, stage 1 (no partition X): regenerate the E1 datasets P/T/V (wheeled and NERVA) with
# F's committed scripts and seeds, compare their hashes with F's committed manifests, and train every E1 arm on T
# (early stopping on V) with predictions on P, for the pilot power rule of e1_analyze.py. Outputs under runs/E1v2/
# so F's tracked manifests and records are never overwritten. NERVA: observe-only simulator experience from the
# validated policy (SHA-256 610b1c59…), Open Duck Playground at NERVA's pinned commit.
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
mkdir -p /work/nerva_in /work/NERVA runs/E1v2 runs/nerva_policy
gcloud storage cp "gs://$BUCKET/apm-g/inputs/nerva/*" /work/nerva_in/
tar xzf /work/nerva_in/nerva.tar.gz -C /work/NERVA
cp /work/nerva_in/candidate.onnx runs/nerva_policy/candidate.onnx
sha256sum runs/nerva_policy/candidate.onnx
git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/Open_Duck_Playground
git -C /work/Open_Duck_Playground checkout -q b9be205ac64488c23504ca42e5ec790337adeec3
pip install -q mujoco==3.3.0 onnxruntime==1.21.0
export OPEN_DUCK_ROOT=/work
for spec in P:40 T:100 V:30; do
  python $F/e1_data.py --embodiment wheeled --partition ${spec%%:*} --n ${spec##*:} --out runs/E1v2 --workers $P
done
for spec in P:120 T:480 V:160; do
  p=${spec%%:*}
  python $F/nerva_generate.py --nerva /work/NERVA --policy runs/nerva_policy/candidate.onnx --partition $p \
    --trials ${spec##*:} --out runs/E1v2/nerva/$p --workers $P
  python $F/e1_data.py --embodiment nerva --partition $p --experience runs/E1v2/nerva/$p --out runs/E1v2 --workers $P
done
python - <<'PY'
import json, pathlib
for f in sorted(pathlib.Path("runs/E1").glob("*.manifest.json")):
    old = json.loads(f.read_text()); new = json.loads((pathlib.Path("runs/E1v2") / f.name).read_text())
    print("MANIFEST", f.name, "sha_equal", old["sha256"] == new["sha256"], "samples", old["samples"], new["samples"],
          "events_equal", old["event_counts"] == new["event_counts"])
for p in ("P", "T", "V"):
    old = json.loads((pathlib.Path("runs/E1/nerva") / p / "manifest.json").read_text())
    new = json.loads((pathlib.Path("runs/E1v2/nerva") / p / "manifest.json").read_text())
    same = sum(a["sha256"] == b["sha256"] for a, b in zip(old["trials"], new["trials"]))
    print("NERVA", p, "trials", len(old["trials"]), len(new["trials"]), "identical_trial_files", same)
PY
python $F/e1_train.py --embodiment wheeled --target collision --data runs/E1v2 --out runs/E1v2/preds --parts T,V,P --workers $P
python $F/e1_train.py --embodiment wheeled --target stability_loss --data runs/E1v2 --out runs/E1v2/preds --parts T,V,P --workers $P
python $F/e1_train.py --embodiment nerva --target stability_loss --data runs/E1v2 --out runs/E1v2/preds --parts T,V,P --workers $P
rm -f runs/E1v2/*.pkl  # large; regenerated deterministically in stage 2 (hashes in the manifests)
echo e1_stage1 done
