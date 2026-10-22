#!/bin/bash
# E13 runs after the divergence fix. Usage: scripts/run_E13_resume.sh <gpu> "<n>:<lr_rho>:<tag>:<seed>" ...
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for cfg in "$@"; do
  IFS=: read n lr tag seed <<< "$cfg"
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.dynamics --steps 40000 --every 1000 --proxies \
    --rho-optimizer sgd --clip mu --rho-lr $lr --tag $tag --n $n --seed $seed --out results/E13
done
