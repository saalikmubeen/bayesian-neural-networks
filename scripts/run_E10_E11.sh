#!/bin/bash
# E10 optimizer clock + E11 prior scale. Usage: scripts/run_E10_E11.sh <gpu> <part>
#   part a: E10 at n = 500 + E11     part b: E10 at n = 10000
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; PART=$2
run() { CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.dynamics --steps 40000 --every 1000 --proxies "$@"; }
if [ "$PART" = a ]; then NS=500; else NS=10000; fi
for seed in 0 1 2; do
  for n in $NS; do
    run --n $n --seed $seed --lr 5e-4 --tag lr5e-4 --out results/E10
    run --n $n --seed $seed --lr 2e-3 --tag lr2e-3 --out results/E10
    run --n $n --seed $seed --rho-optimizer sgd --rho-lr 0.05 --tag sgd --out results/E10
  done
  if [ "$PART" = a ]; then
    run --n 2000 --seed $seed --prior-sigma 0.1 --tag sp0.1 --out results/E11
    run --n 2000 --seed $seed --prior-sigma 0.3 --tag sp0.3 --out results/E11
  fi
done
