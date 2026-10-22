#!/bin/bash
# E20 HMC (prior N(0, 0.3^2)), 8 batched chains, then BBB under the same prior (experiments/E20_hmc_sp03/protocol.md).
# Usage: scripts/run_E20.sh <gpu> <n>
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; N=$2
CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.hmc_batch --n $N --chains 8 --prior-sigma 0.3 \
  --warmup 500 --iters 1000 --thin 10 --traj-len 0.471 --max-leapfrog 6000 --out results/E20
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method bbb_steps23550_sp0.3 --n $N --seed $seed \
    --hidden 100 --out results/E20 --save-model
done
