#!/bin/bash
# E14 main runs with the (lr, hess_init) chosen in results/E14/tune/choice.txt.
# Usage: scripts/run_E14.sh <gpu> <lr> <hess_init> "<n>:<seed>" ...
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; LR=$2; H=$3; shift 3
for cfg in "$@"; do
  n=${cfg%%:*}; seed=${cfg##*:}
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.ivon_dynamics --n $n --seed $seed --lr $LR --hess-init $H \
    --steps 40000 --every 1000 --out results/E14
done
