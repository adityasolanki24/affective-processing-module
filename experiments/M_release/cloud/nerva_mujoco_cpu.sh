# NERVA's MuJoCo backend on a CPU VM (programme J/L setup): NERVA simulation venv + OSMesa for rendered vision.
source experiments/J_validation_transfer/cloud/nerva_env.sh
apt-get install -y -qq libgl1 libosmesa6 libgl1-mesa-dri >/dev/null 2>&1 || \
  { sleep 60; apt-get -o DPkg::Lock::Timeout=600 install -y -qq libgl1 libosmesa6 libgl1-mesa-dri >/dev/null 2>&1; }
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
