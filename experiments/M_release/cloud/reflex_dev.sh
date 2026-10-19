# Push-reflex development (flat, development seeds 72101-72105; never the battery seeds): v1, the plain
# candidate and the reflex variants of configs/t1_push_reflex_variants.json, in parallel.
set -u
OUT=out/reflex_dev
mkdir -p "$OUT"
SC="push_walk_0.5,push_walk_0.3,push_stand_0.3,startstop,speed_0.2,turn_walk_+"
G="experiments/M_release/isaac/gait_battery.py --env flat --seeds 72101-72105 --scenarios $SC --out $OUT"
$ISAAC_PY $G --controller v1 > "$OUT/p_v1.log" 2>&1 &
$ISAAC_PY $G --controller candidate > "$OUT/p_cand.log" 2>&1 &
for v in R1 R2 R3; do
  P=$($ISAAC_PY -c "import json; print(json.dumps(json.load(open('experiments/M_release/configs/t1_push_reflex_variants.json'))['$v']))")
  $ISAAC_PY $G --controller candidate --label "$v" --profile "$P" > "$OUT/p_$v.log" 2>&1 &
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do sleep 60; echo "$(date -u +%H:%M) $(grep -h '^RUN' $OUT/p_*.log | wc -l) runs"; done
wait
grep -h "BATTERY DONE\|Traceback\|Error:" $OUT/p_*.log
