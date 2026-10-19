# T1 gait battery on one VM: both controller stacks for one environment (env sourced: isaac_env.sh).
#   bash experiments/M_release/cloud/gait_battery.sh ENV
# Optional: SEEDS (default 72001-72005), PROFILE (candidate deployment profile JSON file) and LABEL (its name).
set -u
ENV=$1
SEEDS=${SEEDS:-72001-72005}
OUT=out/gait_battery
mkdir -p "$OUT"
$ISAAC_PY experiments/M_release/isaac/gait_battery.py --controller v1 --env "$ENV" --seeds "$SEEDS" \
  --video-seed "${SEEDS%-*}" --out "$OUT" > "$OUT/proc_${ENV}_v1.log" 2>&1 &
sleep 90  # staggered starts (simultaneous Isaac starts corrupted one process)
EXTRA=""
[ -n "${PROFILE:-}" ] && EXTRA="--profile $PROFILE --label ${LABEL:-}"
$ISAAC_PY experiments/M_release/isaac/gait_battery.py --controller candidate --env "$ENV" --seeds "$SEEDS" \
  --video-seed "${SEEDS%-*}" --out "$OUT" $EXTRA > "$OUT/proc_${ENV}_candidate.log" 2>&1 &
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60
  echo "$(date -u +%H:%M) $(grep -h '^RUN' $OUT/proc_${ENV}_*.log 2>/dev/null | wc -l) runs, falls: $(grep -h '^RUN.* fell' $OUT/proc_${ENV}_*.log 2>/dev/null | wc -l)"
done
wait
grep -h "BATTERY DONE\|Traceback\|Error:" $OUT/proc_${ENV}_*.log | head
