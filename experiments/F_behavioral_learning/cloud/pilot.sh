#!/bin/bash
# Pilot runs (partition P / training partitions only; no confirmatory data). Sizes the E3, E4 and E7 addenda and
# measures VM throughput. Run from the repository root inside the VM's venv.
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
mkdir -p runs/E4/pilot
for v in B3 B4; do
  python $F/e4_train.py --variant $v --seed 0 --budget 1200000 --collect-workers 8 --out runs/E4/pilot \
    > runs/E4_pilot_$v.log 2>&1 &
done
python $F/e7_run.py --partition P --n 20 --workers 14 > runs/E7_pilot.log 2>&1 &
python $F/e3_run.py --partition P --n 10 --mode retention --workers 10 > runs/E3_retention_pilot.log 2>&1
python $F/e3_run.py --partition P --n 20 --workers 24 > runs/E3_pilot.log 2>&1
wait
python $F/e4_eval.py --partition P --n 40 --models runs/E4/pilot --variants B3,B4 --seeds 0 --out runs/E4 \
  --workers "$(nproc)" > runs/E4_eval_pilot.log 2>&1
echo pilot done
