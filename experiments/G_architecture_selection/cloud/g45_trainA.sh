#!/bin/bash
# G4/G5 training, VM A: PPO-from-scratch cells (A2, A3, A4) x seeds 0-4, 2.4M decisions each (prereg/G45_ADDENDUM.md).
set -x
S=experiments/G_architecture_selection/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
mkdir -p runs/G45/models
for a in A2 A3 A4; do for s in 0 1 2 3 4; do
  python $S/g45_train.py --arch $a --method ppo --seed $s --budget 2400000 --collect-workers 3 \
    --out runs/G45/models > runs/G45/models/train_${a}_ppo_s$s.log 2>&1 &
done; done
wait
echo g45_trainA done
