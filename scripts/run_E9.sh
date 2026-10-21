#!/bin/bash
# E9 checkpoint selection with synthetic outliers (experiments/E9_selection/protocol.md).
# Usage: scripts/run_E9.sh <gpu> "<n>:<rho>" ...      e.g. scripts/run_E9.sh 0 10000:-5 500:-5
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for cfg in "$@"; do
  n=${cfg%%:*}; rho=${cfg##*:}
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.dynamics --n $n --seed $seed --rho-init $rho \
      --steps 40000 --every 1000 --proxies --out results/E9
  done
done
