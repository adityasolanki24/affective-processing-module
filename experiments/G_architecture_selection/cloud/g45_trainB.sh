#!/bin/bash
# G4/G5 training, VM B: bcppo (2.4M), bc and dagger (0.3M) cells for A2, A3 x seeds 0-4 (prereg/G45_ADDENDUM.md).
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
mkdir -p runs/G45/models
for a in A2 A3; do for s in 0 1 2 3 4; do
  python $S/g45_train.py --arch $a --method bcppo --seed $s --budget 2400000 --collect-workers 3 \
    --out runs/G45/models > runs/G45/models/train_${a}_bcppo_s$s.log 2>&1 &
  for m in bc dagger; do
    python $S/g45_train.py --arch $a --method $m --seed $s --budget 300000 --collect-workers 1 \
      --out runs/G45/models > runs/G45/models/train_${a}_${m}_s$s.log 2>&1 &
  done
done; done
wait
echo g45_trainB done
