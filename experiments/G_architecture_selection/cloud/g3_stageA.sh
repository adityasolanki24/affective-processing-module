#!/bin/bash
# G3 stage A (partitions T, V, P only): exploration data, training (6 kinds x 5 seeds), equal lambda tuning on V,
# closed-loop pilot on P for the sample-size amendment.
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)
mkdir -p runs/logs_G3 runs/G3/results_out
for spec in T:150 V:40 P:40; do
  python $S/g3_data.py --partition ${spec%%:*} --n ${spec##*:} --out runs/G3/data --workers $P >> runs/logs_G3/data.log 2>&1
done
python $S/g3_train.py --data runs/G3/data --out runs/G3/models --parts P --workers 30 > runs/logs_G3/train.log 2>&1
python $S/g3_closed.py tune --models runs/G3/models --k 16 --n 30 --workers $P > runs/logs_G3/tune.log 2>&1
python $S/g3_closed.py test --partition P --models runs/G3/models --n 40 --workers $P > runs/logs_G3/pilot.log 2>&1
cp experiments/G_architecture_selection/configs/G3_selected.json experiments/G_architecture_selection/results/G3/tuning_table.json runs/G3/results_out/
rm -f runs/G3/data/T.pkl runs/G3/data/V.pkl
echo g3_stageA done
