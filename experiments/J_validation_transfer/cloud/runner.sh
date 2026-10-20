#!/bin/bash
# Programme J job runner (VM startup script, root, Ubuntu 24.04). Installs Python + numpy once, then takes job
# archives from gs://BUCKET/apm-j/NAME/queue/ in name order. Each archive holds apm/ (code) and apm/job.sh. The job
# runs with PYTHONPATH=src in the venv; apm/out/ is uploaded to gs://BUCKET/apm-j/NAME/done/ID/ with the log and exit
# code. The VM deletes itself after IDLE_MIN minutes without a job; --max-run-duration is the hard backstop.
set -euo pipefail
MD=http://metadata.google.internal/computeMetadata/v1/instance
get() { curl -sf -H Metadata-Flavor:Google "$MD/$1"; }
BUCKET=$(get attributes/apm-bucket); IDLE_MIN=$(get attributes/apm-idle-min); NAME=$(get name)
ZONE=$(get zone | awk -F/ '{print $NF}')
Q=gs://$BUCKET/apm-j/$NAME
W=/work; mkdir -p $W/jobs
exec > >(tee -a $W/runner.log) 2>&1
echo "=== start $(date -u +%FT%TZ) nproc=$(nproc)"
export DEBIAN_FRONTEND=noninteractive
# GPU images install the NVIDIA driver with apt at first boot: wait for the dpkg lock instead of failing
for i in 1 2 3; do
  apt-get -o DPkg::Lock::Timeout=900 update -qq && \
    apt-get -o DPkg::Lock::Timeout=900 install -y -qq python3-venv git >/dev/null && break
  sleep 30
done
python3 -m venv $W/venv && $W/venv/bin/pip install -q --upgrade pip && $W/venv/bin/pip install -q numpy==2.2.6
$W/venv/bin/python -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
echo ready | gcloud storage cp - $Q/ready
last=$(date +%s)
while true; do
  job=$(gcloud storage ls "$Q/queue/*.tgz" 2>/dev/null | sort | head -1 || true)
  if [ -n "$job" ]; then
    id=$(basename "$job" .tgz); d=$W/jobs/$id; mkdir -p $d
    gcloud storage cp "$job" $d/job.tgz && gcloud storage rm "$job"
    echo "=== job $id $(date -u +%FT%TZ)"
    (cd $d && tar xzf job.tgz && cd apm && { . $W/venv/bin/activate 2>/dev/null || true; } \
      && export PYTHONPATH=src && GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*' bash job.sh) \
      > $d/log.txt 2>&1 && rc=0 || rc=$?
    echo "$rc" > $d/rc
    if [ -d $d/apm/out ]; then tar czf $d/out.tgz -C $d/apm out; else tar czf $d/out.tgz -T /dev/null; fi
    gcloud storage cp $d/out.tgz $d/log.txt $Q/done/$id/ && gcloud storage cp $d/rc $Q/done/$id/rc
    echo "=== done $id rc=$(cat $d/rc) $(date -u +%FT%TZ)"
    rm -rf $d
    last=$(date +%s)
  else
    if [ $(( $(date +%s) - last )) -gt $(( IDLE_MIN * 60 )) ]; then
      echo "=== idle, deleting"; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
    fi
    sleep 10
  fi
done
