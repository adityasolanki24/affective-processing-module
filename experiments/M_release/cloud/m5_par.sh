# Run lifetimes in parallel Isaac processes on one VM (env sourced: isaac_env.sh, setup_nerva_inputs.sh).
#   bash experiments/M_release/cloud/m5_par.sh NPROC FIRST N OUTDIR -- <m5_lifetime.py args without --first/--n/--out>
# Process j runs a contiguous block of indices (one scene load per process) and writes OUTDIR/part_j.jsonl.
NPROC=$1; FIRST=$2; N=$3; OUT=$4; shift 5
mkdir -p "$OUT"
B=$(( (N + NPROC - 1) / NPROC ))
for j in $(seq 0 $((NPROC - 1))); do
  F=$((FIRST + j * B)); M=$(( N - j * B < B ? N - j * B : B ))
  [ $M -le 0 ] && continue
  $ISAAC_PY experiments/M_release/isaac/m5_lifetime.py "$@" --first $F --n $M --out "$OUT/part_$j.jsonl" \
    --logs "$OUT/logs" > "$OUT/proc_$j.log" 2>&1 &
done
# heartbeat for the streamed job log: lifetimes completed per process, every minute
( while sleep 60; do
    printf "%s progress:" "$(date -u +%H:%M)"
    for f in "$OUT"/part_*.jsonl; do [ -f "$f" ] && printf " %s=%s" "$(basename $f .jsonl)" "$(wc -l < $f)"; done
    echo
  done ) &
HB=$!
wait $(jobs -p | grep -v "^$HB$")
kill $HB 2>/dev/null
grep -hE "^[A-Z_]+ [a-z_A-Z]+ [A-Z][0-9]+:" "$OUT"/proc_*.log
echo "m5_par done: $(cat $OUT/part_*.jsonl 2>/dev/null | wc -l) lifetimes"
