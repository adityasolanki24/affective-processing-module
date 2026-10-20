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
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N
# startup-script jobs run without HOME; Kit derives its cache and data paths from it (null -> RTX init crash)
export HOME=${HOME:-/root} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/root/.cache} XDG_DATA_HOME=${XDG_DATA_HOME:-/root/.local/share}
export USER=${USER:-root}
export DEBIAN_FRONTEND=noninteractive
LIBS="zstd libglu1-mesa libxt6 libxrandr2 libxi6 libxinerama1 libxcursor1 libsm6 libice6 libgl1 libegl1 libvulkan1"
if ! dpkg -s $LIBS >/dev/null 2>&1; then apt-get update -qq && apt-get install -y -qq $LIBS >/dev/null; fi
# RTX rendering needs NVIDIA's GL/Vulkan user-space libraries matching the kernel driver exactly. The Deep Learning
# VM's driver install provides CUDA only, so add them from NVIDIA's runfile for the same version (no kernel module).
if ! ldconfig -p | grep -q libGLX_nvidia.so.0 || ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq libgl1 libegl1 libglvnd0 libvulkan1 vulkan-tools pkg-config >/dev/null
  V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  curl -fsSLo /tmp/nv.run https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run     || curl -fsSLo /tmp/nv.run https://us.download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V.run
  sh /tmp/nv.run --no-kernel-module --silent --no-questions --accept-license --no-backup --no-nouveau-check     --no-x-check --install-libglvnd > /tmp/nv_install.log 2>&1 || { tail -20 /tmp/nv_install.log; }
fi
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion|apiVersion" | head -3 || echo "[isaac_env] vulkan not available"
# an incomplete local install (an interrupted restore, or a truncated snapshot restored on 2026-10-19) is wiped and
# restored again: it must have the venv's pip and the Isaac Lab checkout
if [ -d $ROOT ] && { [ ! -d $ROOT/IsaacLab/source ] || ! $ROOT/venv/bin/python -c "import pip" >/dev/null 2>&1; }; then
  echo "[isaac_env] local install incomplete: removing it and restoring again"
  rm -rf $ROOT
fi
if [ ! -x $ROOT/venv/bin/python ]; then
  mkdir -p $ROOT
  if gcloud storage ls "$SNAP" >/dev/null 2>&1; then
    echo "[isaac_env] restoring $SNAP"
    set -o pipefail
    gcloud storage cat "$SNAP" | zstd -d -T0 | tar -x -C / || { echo "[isaac_env] restore failed"; exit 3; }
    set +o pipefail
    if [ ! -x $ROOT/venv/bin/python ]; then  # uv's interpreter lived outside the snapshot: reinstall that version
      PYV=$(awk -F' = ' '/^version_info/ {print $2}' $ROOT/venv/pyvenv.cfg | cut -d. -f1-3)
      $ROOT/uv/uv python install "$PYV" >/dev/null 2>&1
      [ -x $ROOT/venv/bin/python ] || $ROOT/uv/uv venv --seed --python "$PYV" --allow-existing $ROOT/venv >/dev/null
    fi
  else
    echo "[isaac_env] installing Isaac Sim $ISAAC_SIM_VERSION and Isaac Lab $ISAAC_LAB_TAG"
    export UV_PYTHON_INSTALL_DIR=$ROOT/python  # keep the interpreter inside the snapshot
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$ROOT/uv sh >/dev/null
    UV=$ROOT/uv/uv
    $UV venv --seed --python 3.11 $ROOT/venv
    $UV pip install --python $ROOT/venv/bin/python $TORCH_SPEC --index-url https://download.pytorch.org/whl/cu128
    $UV pip install --python $ROOT/venv/bin/python "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
      --extra-index-url https://pypi.nvidia.com
    git clone -q --depth 1 --branch $ISAAC_LAB_TAG https://github.com/isaac-sim/IsaacLab.git $ROOT/IsaacLab
    $ROOT/venv/bin/pip install -q mujoco==3.3.0 onnxruntime imageio imageio-ffmpeg pillow matplotlib
    echo "[isaac_env] saving snapshot $SNAP"
    tar -C / -cf - work/isaac | zstd -T0 -3 | gcloud storage cp - "$SNAP"
  fi
fi
export PATH=$ROOT/uv:$PATH  # isaaclab.sh uses uv when the venv was made by uv
LAB_OK='import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl_rl")) else 1)'
# checked on every job: the marker travels in the snapshot, so a restore that came out incomplete (VMs h4, h2 of the
# remediation: editable-install finders missing) would otherwise be trusted
if (cd / && $ROOT/venv/bin/python -c "$LAB_OK" 2>/dev/null); then touch $ROOT/.lab_ok; else rm -f $ROOT/.lab_ok; fi
if [ ! -f $ROOT/.lab_ok ]; then
  echo "[isaac_env] (re)installing Isaac Lab $ISAAC_LAB_TAG extensions"
  # flatdict's sdist needs pkg_resources at build time, which recent setuptools in pip's build isolation lacks
  $ROOT/venv/bin/python -m pip install -q "setuptools<70" wheel
  $ROOT/venv/bin/python -m pip install -q --no-build-isolation flatdict==4.0.1
  (cd $ROOT/IsaacLab/source && $ROOT/venv/bin/python -m pip install -q -e isaaclab -e isaaclab_assets      -e isaaclab_tasks -e "isaaclab_rl[rsl_rl]" > /tmp/isaaclab_install.log 2>&1) || tail -40 /tmp/isaaclab_install.log
  (cd / && $ROOT/venv/bin/python -c "import isaaclab, isaaclab_tasks, rsl_rl; print('[isaac_env] isaaclab ok', isaaclab.__file__)")
  (cd / && $ROOT/venv/bin/python -c "$LAB_OK") && touch $ROOT/.lab_ok
  # never overwrite the shared snapshot from a repair: a working one already exists (it restored on other VMs)
  if [ -f $ROOT/.lab_ok ] && ! gcloud storage ls "$SNAP" >/dev/null 2>&1; then
    echo "[isaac_env] saving repaired snapshot $SNAP"
    tar -C / -cf - work/isaac | zstd -T0 -3 | gcloud storage cp - "$SNAP"
  fi
fi
export ISAAC_PY=$ROOT/venv/bin/python ISAACLAB=$ROOT/IsaacLab
$ISAAC_PY -c "import isaacsim, torch; print('[isaac_env] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
