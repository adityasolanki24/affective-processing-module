# NERVA's MuJoCo reference on an Isaac VM (sourced after isaac_env.sh): a separate Python 3.12 venv with NERVA's
# inference lock (numpy 2.5 needs 3.12; the Isaac venv is left untouched). NERVA and Open Duck Playground read-only.
source experiments/M_release/cloud/setup_nerva_inputs.sh
mkdir -p /work/odroot && ln -sfn /work/odp /work/odroot/Open_Duck_Playground
if [ ! -x /work/nvenv312/bin/python ] || ! /work/nvenv312/bin/python -c "import numpy, jax, nerva" 2>/dev/null; then
  /work/isaac/uv/uv venv -q --python 3.12 --allow-existing /work/nvenv312
  /work/isaac/uv/uv pip install -q -p /work/nvenv312/bin/python -r /work/NERVA/env/open_duck_inference.lock.txt
  /work/isaac/uv/uv pip install -q -p /work/nvenv312/bin/python --no-deps -e /work/odp -e /work/NERVA
fi
export NERVA_ROOT=/work/NERVA OPEN_DUCK_ROOT=/work/odroot ODP_ROOT=/work/odp OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
export NPY=/work/nvenv312/bin/python
