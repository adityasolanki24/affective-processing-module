# APM v1.0 job wrapper: run a command with a hard timeout, streaming its log to the bucket every 30 s.
#   bash experiments/M_release/cloud/run.sh TAG SECONDS cmd args...
# Live log: gs://<bucket>/apm-m/live/TAG.log (also kept in out/TAG.log).
TAG=$1; LIMIT=$2; shift 2
BUCKET=${APM_BUCKET:-$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/apm-bucket)}
mkdir -p out
LOG=out/$TAG.log
( while sleep 30; do gcloud storage cp -q "$LOG" "gs://$BUCKET/apm-m/live/$TAG.log" >/dev/null 2>&1; done ) &
STREAM=$!
timeout --kill-after=30 "$LIMIT" "$@" > "$LOG" 2>&1
RC=$?
kill $STREAM 2>/dev/null
gcloud storage cp -q "$LOG" "gs://$BUCKET/apm-m/live/$TAG.log" >/dev/null 2>&1
echo "[run.sh] $TAG exit $RC (124 = timeout)" | tee -a "$LOG"
grep -vE "\[Warning\]|^\s*$" "$LOG" | tail -60
exit $RC
