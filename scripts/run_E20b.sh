#!/bin/bash
# E20b: E20's HMC on data seeds 1 and 2 (experiments/E20b_hmc_seeds/protocol.md).
# Usage: scripts/run_E20b.sh <gpu> <data seed>   (runs n = 2000 then n = 500)
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; D=$2
for N in 2000 500; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.hmc_batch --n $N --data-seed $D --chains 8 --prior-sigma 0.3 \
    --warmup 500 --iters 1000 --thin 10 --traj-len 0.471 --max-leapfrog 6000 --out results/E20
done
