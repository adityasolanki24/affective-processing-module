#!/bin/bash
# Programme F, E1 completion: the NERVA arm (stage 1 = partitions P/T/V; stage 2 adds X when E1_STAGE=2).
# NERVA experience is generated in a separate venv pinned to Open Duck's inference lock (jax/etils/mujoco/onnxruntime;
# numpy 2.5.3), so the APM environment stays at its own pins; the experience files (npz) are then read by F's
# e1_data.py in the APM environment. Observe-only; the validated policy (SHA-256 610b1c59…) is not modified.
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
PARTS=${E1_PARTS:-"P:120 T:480 V:160"}
mkdir -p /work/nerva_in /work/NERVA runs/E1v2 runs/nerva_policy
gcloud storage cp "gs://$BUCKET/apm-g/inputs/nerva/*" /work/nerva_in/
tar xzf /work/nerva_in/nerva.tar.gz -C /work/NERVA
cp /work/nerva_in/candidate.onnx runs/nerva_policy/candidate.onnx
sha256sum runs/nerva_policy/candidate.onnx
git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/Open_Duck_Playground
git -C /work/Open_Duck_Playground checkout -q b9be205ac64488c23504ca42e5ec790337adeec3
python3 -m venv /work/nvenv
/work/nvenv/bin/pip install -q --upgrade pip
/work/nvenv/bin/pip install -q -r /work/nerva_in/requirements-inference.lock.txt
/work/nvenv/bin/pip install -q --no-deps -e /work/Open_Duck_Playground -e /work/NERVA
export OPEN_DUCK_ROOT=/work
for spec in $PARTS; do
  p=${spec%%:*}
  /work/nvenv/bin/python $F/nerva_generate.py --nerva /work/NERVA --policy runs/nerva_policy/candidate.onnx \
    --partition $p --trials ${spec##*:} --out runs/E1v2/nerva/$p --workers $P
  python $F/e1_data.py --embodiment nerva --partition $p --experience runs/E1v2/nerva/$p --out runs/E1v2 --workers $P
done
python - <<'PY'
import json, pathlib
for p in ("P", "T", "V"):
    o, n = pathlib.Path("runs/E1/nerva") / p / "manifest.json", pathlib.Path("runs/E1v2/nerva") / p / "manifest.json"
    if o.exists() and n.exists():
        old, new = json.loads(o.read_text()), json.loads(n.read_text())
        same = sum(a["sha256"] == b["sha256"] for a, b in zip(old["trials"], new["trials"]))
        print("NERVA", p, "trials", len(old["trials"]), len(new["trials"]), "identical_trial_files", same)
    m_old, m_new = pathlib.Path(f"runs/E1/nerva_{p}.manifest.json"), pathlib.Path(f"runs/E1v2/nerva_{p}.manifest.json")
    if m_old.exists() and m_new.exists():
        a, b = json.loads(m_old.read_text()), json.loads(m_new.read_text())
        print("MANIFEST", m_old.name, "sha_equal", a["sha256"] == b["sha256"], "samples", a["samples"], b["samples"],
              "events_equal", a["event_counts"] == b["event_counts"])
PY
python $F/e1_train.py --embodiment nerva --target stability_loss --data runs/E1v2 --out runs/E1v2/preds \
  --parts ${E1_TRAIN_PARTS:-T,V,P} --workers $((P / 2))
rm -f runs/E1v2/*.pkl
echo e1_nerva done
