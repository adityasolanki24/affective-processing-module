set -x
source experiments/M_release/cloud/isaac_env.sh || true
P=/work/isaac/venv/bin/python
ls -la /work/isaac/venv/bin/python*; $P -c "import sys; print(sys.version, sys.prefix)"; $P -m pip --version
cd /work/isaac/IsaacLab/source && ls && $P -m pip install -e isaaclab 2>&1 | tail -25
$P -c "import isaaclab; print(isaaclab.__file__)" 2>&1 | tail -5
$P -m pip install -e isaaclab_assets -e isaaclab_tasks -e "isaaclab_rl[rsl_rl]" 2>&1 | tail -15
$P -c "import isaaclab, isaaclab_tasks, rsl_rl; print('OK')" 2>&1 | tail -5
