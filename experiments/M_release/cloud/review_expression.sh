# Re-render only the matched present task after restoring actual physical sensor histories.
# REVIEW_ARCHIVE_URI must point to the completed EXPRESSION_PILOT output archive.
set -euo pipefail
source experiments/M_release/cloud/isaac_env.sh
source experiments/M_release/cloud/setup_nerva_inputs.sh
mkdir -p /work/review_history
gcloud storage cat "${REVIEW_ARCHIVE_URI:?set the completed physical-history archive}" |
  tar xzf - -C /work/review_history --wildcards 'out/expression/*/logs/*.jsonl' 'out/expression/*/lifetimes.jsonl'
for robot in ${REVIEW_ROBOTS:-humanoid nerva}; do
  model_args=$($ISAAC_PY -c "import sys; sys.path.insert(0,'experiments/M_release/cloud'); from m6_plan import model_args; print(model_args('$robot'))")
  for mode in matched neutral_matched; do
    histories=change_B,safe
    if [ "$mode" = neutral_matched ]; then histories=safe; fi
    # model_args consists only of repository-generated option/value tokens without whitespace in paths.
    $ISAAC_PY experiments/M_release/isaac/m5_lifetime.py \
      --robot "$robot" --scene warehouse --experiment M7 --partition P --first 71001 \
      --n 1 --n-enc 1 --switch 0 --systems DT --histories "$histories" \
      --review-person-b --review-after 0 --expression-mode "$mode" --supervision matched \
      --sensor-expectation \
      --review-replay-dir "/work/review_history/out/expression/$robot/logs" \
      --video --logs "out/review/$robot/$mode/logs" \
      --out "out/review/$robot/$mode/lifetimes.jsonl" $model_args
  done
done
