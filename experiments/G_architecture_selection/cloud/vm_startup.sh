#!/bin/bash
# Programme G VM startup (runs as root on Ubuntu 24.04). Clones the uploaded git bundle (so the confirmatory lock
# sees a clean checkout of the exact commit), installs pinned packages, runs cloud/$JOB.sh, syncs outputs to the
# bucket every 2 minutes and at exit, then deletes this VM. The hard cap (--max-run-duration) is the backstop.
set -uo pipefail
MD=http://metadata.google.internal/computeMetadata/v1/instance
get() { curl -sf -H Metadata-Flavor:Google "$MD/$1"; }
BUCKET=$(get attributes/apm-bucket); RUN=$(get attributes/apm-run-id); SHA=$(get attributes/apm-code-sha)
BRANCH=$(get attributes/apm-branch); JOB=$(get attributes/apm-job); NAME=$(get name); ZONE=$(get zone | awk -F/ '{print $NF}')
OUT=gs://$BUCKET/apm-g/$RUN/out
W=/work; mkdir -p $W/out; LOG=$W/out/job.log
exec > >(tee -a $LOG) 2>&1
sync_out() { gcloud storage rsync --recursive --quiet $W/out $OUT >/dev/null 2>&1 || true; }
finish() {
  echo "=== exit $(date -u +%FT%TZ)"; cp -r $W/repo/runs/. $W/out/runs/ 2>/dev/null || true; sync_out
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
}
trap finish EXIT
echo "=== start $(date -u +%FT%TZ) run=$RUN sha=$SHA nproc=$(nproc)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-venv git >/dev/null
gcloud storage cp gs://$BUCKET/apm-g/$RUN/code.bundle $W/code.bundle
git clone -q --branch "$BRANCH" $W/code.bundle $W/repo && cd $W/repo
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "commit mismatch"; exit 1; }
python3 -m venv $W/venv && . $W/venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy==2.5.4 && pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q -e . --no-deps
python -c "import numpy, torch, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__)"
mkdir -p $W/out/runs
( while true; do sleep 120; mkdir -p $W/out/runs; cp -r runs/. $W/out/runs/ 2>/dev/null; sync_out; done ) &
bash experiments/G_architecture_selection/cloud/$JOB.sh
echo "=== job exit code $?"
