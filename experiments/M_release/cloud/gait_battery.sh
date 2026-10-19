# T1 gait battery on one VM: both controller stacks for one environment, in parallel (env sourced: isaac_env.sh).
#   bash experiments/M_release/cloud/gait_battery.sh ENV
set -u
ENV=$1
OUT=out/gait_battery
mkdir -p "$OUT"
for c in v1 candidate; do
  $ISAAC_PY experiments/M_release/isaac/gait_battery.py --controller $c --env "$ENV" --seeds 72001-72005 \
    --video-seed 72001 --out "$OUT" > "$OUT/proc_${ENV}_$c.log" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do
  sleep 60
  echo "$(date -u +%H:%M) $(grep -h '^RUN' $OUT/proc_${ENV}_*.log 2>/dev/null | wc -l) runs, falls: $(grep -h '^RUN.* fell' $OUT/proc_${ENV}_*.log 2>/dev/null | wc -l)"
done
wait
grep -h "BATTERY DONE\|Traceback\|Error:" $OUT/proc_${ENV}_*.log | head
