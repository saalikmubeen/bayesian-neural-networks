#!/bin/bash
# E14 tuning: n = 2000, seed 99, 10k steps; choose (lr, hess_init) by validation NLL only.
# Usage: scripts/run_E14_tune.sh <gpu> <lr...>
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for lr in "$@"; do
  for h in 0.1 1.0; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.ivon_dynamics --n 2000 --seed 99 --lr $lr --hess-init $h \
      --steps 10000 --every 1000 --out results/E14/tune --tag lr${lr}_h${h}
  done
done
