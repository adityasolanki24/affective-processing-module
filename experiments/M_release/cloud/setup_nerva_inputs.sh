# NERVA inputs on an Isaac VM (read-only use): NERVA package + frozen policies, Open Duck Playground at NERVA's pin.
BUCKET=${APM_BUCKET:-$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/apm-bucket)}
if [ ! -d /work/NERVA/nerva ]; then
  mkdir -p /work/NERVA && gcloud storage cp -q "gs://$BUCKET/apm-j/inputs/nerva.tgz" /work/nerva.tgz && tar xzf /work/nerva.tgz -C /work/NERVA
fi
if [ ! -d /work/odp/playground ]; then
  git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/odp
  git -C /work/odp checkout -q b9be205ac64488c23504ca42e5ec790337adeec3
fi
