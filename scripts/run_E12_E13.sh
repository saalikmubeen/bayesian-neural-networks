#!/bin/bash
# E12 step-matched baselines + E13 clean SGD arm. Usage: scripts/run_E12_E13.sh <gpu> <part>
#   part a: E12 at n = 2000 + E13 at n = 500     part b: E12 at n = 10000 + E13 at n = 10000
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; PART=$2
if [ "$PART" = a ]; then N12=2000; else N12=10000; fi
for seed in 0 1 2 3 4; do
  for m in dnn_steps23550 ensemble_steps23550; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $m --n $N12 --seed $seed --out results/E12 --save-model
  done
done
dyn() { CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.dynamics --steps 40000 --every 1000 --proxies \
          --rho-optimizer sgd --clip mu --out results/E13 "$@"; }
for seed in 0 1 2; do
  if [ "$PART" = a ]; then
    dyn --n 500 --seed $seed --rho-lr 0.05 --tag sgd_mu
  else
    dyn --n 10000 --seed $seed --rho-lr 0.05 --tag sgd_mu
    dyn --n 10000 --seed $seed --rho-lr 1.0 --tag sgd_scaled
  fi
done
