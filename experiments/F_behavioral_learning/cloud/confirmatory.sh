#!/bin/bash
# Confirmatory runs (partition X) of E2, E3, E4 and E7 exactly as fixed in prereg/E*_ADDENDUM.md. Each runner checks
# that its addendum is committed and the checkout is clean (confirmatory_lock). Run from the repository root inside
# the VM's venv; outputs go to runs/ (git-ignored, so the tree stays clean).
set -x
F=experiments/F_behavioral_learning/scripts
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
P=$(nproc --all)  # plain nproc honours OMP_NUM_THREADS=1
# Sizes and budget from the addenda (do not edit after the addenda are committed).
N_E2=240; N_E3=60; N_E3_RET=60; N_E7=100; N_E4=400; BUDGET=1200000
E4_VARIANTS="B3 B4 B3_noMemory B3_noPersist B3_fastOnly B3_slowOnly B4_half B4_double"

# E4 training: 8 variants x 5 seeds, 2 collection processes each, in parallel with everything else.
mkdir -p runs/E4/X
for v in $E4_VARIANTS; do for s in 0 1 2 3 4; do
  python $F/e4_train.py --variant $v --seed $s --budget $BUDGET --collect-workers 2 --out runs/E4/X > runs/E4/X/train_${v}_s$s.log 2>&1 &
done; done
W=$((P - 80)); [ $W -lt 16 ] && W=16

python $F/e2_run.py --partition X --n $N_E2 --workers $W > runs/E2_X.log 2>&1
python $F/e7_run.py --partition X --n $N_E7 --workers $W > runs/E7_X.log 2>&1
python $F/e3_run.py --partition X --n $N_E3 --workers $W > runs/E3_X.log 2>&1
python $F/e3_run.py --partition X --n $N_E3_RET --mode retention --workers $W > runs/E3_retention_X.log 2>&1
wait
python $F/e4_eval.py --partition X --n $N_E4 --models runs/E4/X --seeds 0,1,2,3,4 --workers $P > runs/E4_eval_X.log 2>&1
# Analyses run locally on the fetched raw data (scripts/e*_analyze.py --partition X).
echo confirmatory done
