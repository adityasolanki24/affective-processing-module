# Build the release videos on a VM from demonstration archives already in the bucket (sourced env: isaac_env.sh).
#   bash experiments/M_release/cloud/m7_build.sh gs://.../done/<job>/out.tgz ...
# Each archive is streamed and unpacked under /work/m7d; only the composed videos go to out/videos.
set -e
D=/work/m7d
mkdir -p "$D"
for uri in "$@"; do
  name=$(basename "$(dirname "$uri")")
  if [ ! -d "$D/$name" ]; then
    mkdir -p "$D/$name"
    gcloud storage cp "$uri" - | tar xzf - -C "$D/$name"
  fi
done
$ISAAC_PY -m pip install -q imageio-ffmpeg >/dev/null 2>&1 || true
$ISAAC_PY experiments/M_release/scripts/m7_videos.py --d "$D" --out out/videos
ls -la out/videos
