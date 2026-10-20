#!/bin/bash
# K6: one-time humanoid environment on a job VM (MuJoCo 3.16, the version the T1 model was compiled with).
set -e
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
if [ ! -x /work/hvenv/bin/python ]; then
  python3 -m venv /work/hvenv
  /work/hvenv/bin/pip install -q --upgrade pip
  /work/hvenv/bin/pip install -q "mujoco==3.16.0" numpy pillow
fi
T=experiments/K_recovery/humanoid/trained
[ -f $T/t1_model.mjb ] || gcloud storage cp -q "gs://$BUCKET/apm-j/inputs/t1_model.mjb" $T/t1_model.mjb
(cd $T && sha256sum -c t1_model.mjb.sha256)
export HPY=/work/hvenv/bin/python OMP_NUM_THREADS=1 PYTHONPATH="$PWD/src"
