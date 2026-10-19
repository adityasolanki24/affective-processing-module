# Expression v2 pilot on one VM: both physical histories for one robot, in parallel (env sourced: isaac_env.sh,
# setup_nerva_inputs.sh).   bash experiments/M_release/cloud/expression_v2.sh ROBOT
set -u
R=$1
OUT=out/expr_v2/$R
mkdir -p "$OUT"
MA=$($ISAAC_PY -c "import sys; sys.path.insert(0,'experiments/M_release/cloud'); from m6_plan import model_args; print(model_args('$R'))")
EXTRA=""
[ "$R" = humanoid ] && EXTRA="--policy-dir experiments/M_release/policies/t1_booster_candidate"
for h in pushy calm; do
  $ISAAC_PY experiments/M_release/isaac/expression_v2.py --robot $R --history $h --out "$OUT" $MA $EXTRA \
    > "$OUT/proc_$h.log" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60
  echo "$(date -u +%H:%M) $(grep -hE '^(CALIBRATED|HISTORY|LEARNED|PRESENT|EXPRESSION DONE)' $OUT/proc_*.log | tail -2 | tr '\n' '|')"
done
wait
grep -hE "^(CALIBRATED|HISTORY|LEARNED|PRESENT|EXPRESSION DONE)|Traceback|Error:" $OUT/proc_*.log | cut -c1-240
