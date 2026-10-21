#!/bin/bash
# E3 budget-matched MAP baseline (experiments/E3_map_baseline/protocol.md).
# Writes into results/E2 so runs pair with the E2 runs. Resumable.
# Usage: scripts/run_E3.sh <gpu> <seeds...>
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for seed in "$@"; do
  for n in 500 2000 10000; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method map --n $n --seed $seed --out results/E2
  done
done
