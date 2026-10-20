# Run lifetimes in parallel Isaac processes on one VM (env sourced: isaac_env.sh, setup_nerva_inputs.sh).
#   bash experiments/M_release/cloud/m5_par.sh NPROC FIRST N OUTDIR -- <m5_lifetime.py args without --first/--n/--out>
# Process j runs a contiguous block of indices (one scene load per process) and writes OUTDIR/part_j.jsonl.
# A heartbeat prints lifetimes completed per process every minute; a process that exits prints its log tail.
NPROC=$1; FIRST=$2; N=$3; OUT=$4; shift 5
mkdir -p "$OUT"
B=$(( (N + NPROC - 1) / NPROC ))
PIDS=()
for j in $(seq 0 $((NPROC - 1))); do
  F=$((FIRST + j * B)); M=$(( N - j * B < B ? N - j * B : B ))
  [ $M -le 0 ] && continue
  $ISAAC_PY experiments/M_release/isaac/m5_lifetime.py "$@" --first $F --n $M --out "$OUT/part_$j.jsonl" \
    --logs "$OUT/logs" > "$OUT/proc_$j.log" 2>&1 &
  PIDS+=($!)
done
declare -A REPORTED
while :; do
  alive=0
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      alive=$((alive + 1))
    elif [ -z "${REPORTED[$i]}" ]; then
      REPORTED[$i]=1
      echo "process $i exited; log tail:"
      grep -vE "\[Warning\]|\[Info\]|^\s*$" "$OUT/proc_$i.log" | tail -8 | cut -c1-240
    fi
  done
  printf "%s progress:" "$(date -u +%H:%M)"
  for f in "$OUT"/part_*.jsonl; do [ -f "$f" ] && printf " %s=%s" "$(basename $f .jsonl)" "$(wc -l < $f)"; done
  echo " alive=$alive"
  for i in "${!PIDS[@]}"; do  # where each live process is
    kill -0 "${PIDS[$i]}" 2>/dev/null && echo "  p$i: $(grep -vE '\[Warning\]|\[Info\]|^\s*$' "$OUT/proc_$i.log" | tail -1 | cut -c1-160)"
  done
  [ $alive -eq 0 ] && break
  sleep 60
done
grep -hE "^[A-Z_]+[a-zA-Z0-9:=./,_-]* [a-z_A-Z]+ [A-Z][0-9]+:" "$OUT"/proc_*.log
echo "m5_par done: $(cat $OUT/part_*.jsonl 2>/dev/null | wc -l) lifetimes"
