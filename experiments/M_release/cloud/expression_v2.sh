# Expression v2 pilot on one VM: both physical histories for one robot, in parallel (env sourced: isaac_env.sh,
# setup_nerva_inputs.sh).   bash experiments/M_release/cloud/expression_v2.sh ROBOT
set -u
R=$1
OUT=out/expr_v2/$R
mkdir -p "$OUT"
MA=$($ISAAC_PY -c "import sys; sys.path.insert(0,'experiments/M_release/cloud'); from m6_plan import model_args; print(model_args('$R'))")
EXTRA=""
# T1: the retained v1 locomotion controller unless T1_POLICY names another policy directory
[ "$R" = humanoid ] && [ -n "${T1_POLICY:-}" ] && EXTRA="--policy-dir $T1_POLICY"
[ "$R" = nerva ] && EXTRA="--near-miss 0.03"  # NERVA falls under contact in Isaac (D3): sudden approaches, no touch
for h in ${HISTORIES:-pushy calm}; do
  # each process its own asset cache (two NERVA imports racing on one cache broke the first pilot)
  APM_CACHE=/work/apm_cache_$h $ISAAC_PY experiments/M_release/isaac/expression_v2.py --robot $R --history $h \
    --out "$OUT" --seed ${SEED:-73001} $MA $EXTRA > "$OUT/proc_$h.log" 2>&1 &
  # two NERVA simulations side by side crash Isaac (null USD stage): NERVA's histories run one after another
  if [ "$R" = nerva ]; then wait; fi
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60
  echo "$(date -u +%H:%M) $(grep -hE '^(CALIBRATED|HISTORY|LEARNED|PRESENT|EXPRESSION DONE)' $OUT/proc_*.log | tail -2 | tr '\n' '|')"
done
wait
grep -hE "^(CALIBRATED|HISTORY|LEARNED|PRESENT|EXPRESSION DONE)|Traceback|Error:" $OUT/proc_*.log | cut -c1-240
