#!/bin/bash
# E22: MNIST digits 0-4 in distribution, digits 5-9 held out (experiments/E22_heldout_digits/protocol.md).
# Usage: scripts/run_E22.sh <gpu> <seed> [<seed> ...]
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for seed in "$@"; do
  for m in dnn ensemble bbb bbb_steps23550 dnn_steps23550 ensemble_steps23550; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --id mnist04 --method $m --n 2000 --seed $seed --out results/E22
  done
done
