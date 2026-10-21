#!/bin/bash
# E2 MNIST grid (experiments/E2_mnist_grid/protocol.md).
# Usage: scripts/run_E2.sh <gpu> <seeds...>     e.g. scripts/run_E2.sh 0 0 1 2 3 4
# Finished runs are skipped, so the script is resumable.
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
OUT=results/E2
GPU=$1; shift
for seed in "$@"; do
  for n in 500 2000 10000; do
    for m in dnn laplace mc_dropout ensemble bbb; do
      CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $m --n $n --seed $seed --out $OUT
    done
  done
  for m in bbb_beta0.1 bbb_beta0.3 bbb_beta3 bbb_beta10 bbb_beta30; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $m --n 2000 --seed $seed --out $OUT
  done
done
