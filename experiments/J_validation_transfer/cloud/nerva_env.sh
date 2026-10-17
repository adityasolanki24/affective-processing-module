#!/bin/bash
# Programme J: one-time NERVA simulation environment on a job VM (sourced by NERVA jobs; idempotent).
# Uses the NERVA package and frozen policies uploaded by `jcloud.py nerva-inputs`
# (gs://BUCKET/apm-j/inputs/nerva.tgz), Open Duck Playground at NERVA's pinned commit, and NERVA's inference lock.
# NERVA's code, contract and policies are used read-only.
set -e
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=$(curl -sf -H Metadata-Flavor:Google $MD/apm-bucket)
if [ ! -x /work/nvenv/bin/python ]; then
  mkdir -p /work/NERVA
  gcloud storage cp "gs://$BUCKET/apm-j/inputs/nerva.tgz" /work/nerva.tgz
  tar xzf /work/nerva.tgz -C /work/NERVA
  git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/Open_Duck_Playground
  git -C /work/Open_Duck_Playground checkout -q b9be205ac64488c23504ca42e5ec790337adeec3
  python3 -m venv /work/nvenv
  /work/nvenv/bin/pip install -q --upgrade pip
  /work/nvenv/bin/pip install -q -r /work/NERVA/env/open_duck_inference.lock.txt
  /work/nvenv/bin/pip install -q --no-deps -e /work/Open_Duck_Playground -e /work/NERVA
fi
export NERVA_ROOT=/work/NERVA OPEN_DUCK_ROOT=/work PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
export NPY=/work/nvenv/bin/python
export PYTHONPATH="$PWD/src:/work/NERVA"
