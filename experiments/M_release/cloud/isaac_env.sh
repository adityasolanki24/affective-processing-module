# APM v1.0: pinned Isaac Sim 5.1.0 + Isaac Lab v2.3.2 environment on a GCP L4 VM (Deep Learning VM, Ubuntu 22.04,
# NVIDIA driver 580). Source this from a job: `source experiments/M_release/cloud/isaac_env.sh`.
#
# - This VM already has the env: reuse it.
# - A snapshot exists in the bucket: restore it (minutes).
# - Otherwise: install the pinned versions and upload a snapshot for the next VM.
#
# Exports ISAAC_PY (the env's python), ISAACLAB (Isaac Lab checkout) and OMNI_KIT_ACCEPT_EULA=YES.
set -e
ISAAC_SIM_VERSION=5.1.0
ISAAC_LAB_TAG=v2.3.2
TORCH_SPEC="torch==2.7.0 torchvision==0.22.0"
ROOT=/work/isaac
SNAP=gs://${APM_BUCKET:-$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/apm-bucket)}/apm-m/env/isaac-${ISAAC_SIM_VERSION}-lab-${ISAAC_LAB_TAG}.tar.zst
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export DEBIAN_FRONTEND=noninteractive
if ! command -v zstd >/dev/null; then apt-get update -qq && apt-get install -y -qq zstd libglu1-mesa libxt6 libxrandr2 \
  libxi6 libxinerama1 libxcursor1 libsm6 libice6 >/dev/null; fi
if [ ! -x $ROOT/venv/bin/python ]; then
  mkdir -p $ROOT
  if gcloud storage ls "$SNAP" >/dev/null 2>&1; then
    echo "[isaac_env] restoring $SNAP"
    gcloud storage cat "$SNAP" | zstd -d -T0 | tar -x -C /
  else
    echo "[isaac_env] installing Isaac Sim $ISAAC_SIM_VERSION and Isaac Lab $ISAAC_LAB_TAG"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$ROOT/uv sh >/dev/null
    UV=$ROOT/uv/uv
    $UV venv --seed --python 3.11 $ROOT/venv
    $UV pip install --python $ROOT/venv/bin/python $TORCH_SPEC --index-url https://download.pytorch.org/whl/cu128
    $UV pip install --python $ROOT/venv/bin/python "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
      --extra-index-url https://pypi.nvidia.com
    git clone -q --depth 1 --branch $ISAAC_LAB_TAG https://github.com/isaac-sim/IsaacLab.git $ROOT/IsaacLab
    (cd $ROOT/IsaacLab && . $ROOT/venv/bin/activate && ./isaaclab.sh --install rsl_rl)
    $ROOT/venv/bin/pip install -q mujoco==3.3.0 onnxruntime imageio imageio-ffmpeg pillow matplotlib
    echo "[isaac_env] saving snapshot $SNAP"
    tar -C / -cf - work/isaac | zstd -T0 -3 | gcloud storage cp - "$SNAP"
  fi
fi
export ISAAC_PY=$ROOT/venv/bin/python ISAACLAB=$ROOT/IsaacLab
$ISAAC_PY -c "import isaacsim, torch; print('[isaac_env] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
