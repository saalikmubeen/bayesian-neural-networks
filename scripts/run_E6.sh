#!/bin/bash
# E6 training dynamics (experiments/E6_dynamics/protocol.md). Usage: scripts/run_E6.sh <gpu> <rho_init...>
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for rho in "$@"; do
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.dynamics --n 2000 --seed $seed --rho-init $rho \
      --steps 50000 --every 1000 --out results/E6
  done
done
