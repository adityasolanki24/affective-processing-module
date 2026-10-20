# Learning-and-recovery demonstration (expression_v2.py --demo): one continuous lifetime, person B1 pushy for
# PUSHY_N meetings, then calm for CALM_N; filmed with the learned state per frame (env sourced: isaac_env.sh,
# setup_nerva_inputs.sh).   bash experiments/M_release/cloud/expression_demo.sh ROBOT
set -u
R=$1
OUT=out/expr_demo/$R
mkdir -p "$OUT"
MA=$($ISAAC_PY -c "import sys; sys.path.insert(0,'experiments/M_release/cloud'); from m6_plan import model_args; print(model_args('$R'))")
EXTRA=""
[ "$R" = nerva ] && EXTRA="--near-miss 0.03 --looming-near 1.5 --looming-horizon 1.0 --looming-upright --camera-calibration robust"  # as in expression_v2.sh
# legged robots: meetings start at the approach point, as in the histories (v1 T1 fell turning at 2 rad/s on the
# long walk from home, outside its 1 rad/s training range)
[ "$R" != nova_carter ] && EXTRA="$EXTRA --demo-reposition"
APM_CACHE=/work/apm_cache_demo $ISAAC_PY experiments/M_release/isaac/expression_v2.py --robot $R --history pushy --demo \
  --pushy-n ${PUSHY_N:-6} --calm-n ${CALM_N:-7} --out "$OUT" --seed ${SEED:-73401} $MA $EXTRA > "$OUT/proc_demo.log" 2>&1
grep -hE "^(CALIBRATED|DEMO)|Traceback|Error:" "$OUT/proc_demo.log" | cut -c1-240
