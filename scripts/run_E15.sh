#!/bin/bash
# E15 IVON with tempered likelihood (experiments/E15_ivon_inflation/protocol.md).
# Usage: scripts/run_E15.sh <gpu> "<n>:<seed>:<k>" ...
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for cfg in "$@"; do
  IFS=: read n seed k <<< "$cfg"
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.ivon_dynamics --n $n --seed $seed --lr 0.2 --hess-init 0.1 \
    --temp $k --steps 40000 --every 1000 --out results/E15 --tag temp$k
done
