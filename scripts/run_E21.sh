#!/bin/bash
# E21: Fashion-MNIST as in-distribution (experiments/E21_fashion_id/protocol.md).
# Usage: scripts/run_E21.sh <gpu> <seed> [<seed> ...]
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for seed in "$@"; do
  for m in dnn ensemble bbb bbb_steps23550 dnn_steps23550 ensemble_steps23550; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --id fashion --method $m --n 2000 --seed $seed --out results/E21
  done
  for m in dnn bbb; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --id fashion --method $m --n 10000 --seed $seed --out results/E21
  done
done
