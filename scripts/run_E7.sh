#!/bin/bash
# E7 CIFAR-10 replication (experiments/E7_cifar/protocol.md). Usage: scripts/run_E7.sh <gpu> <seeds...>
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; shift
for seed in "$@"; do
  for m in dnn ensemble; do
    [ -f results/E7/${m}_n10000_s$seed.npz ] || CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.cifar $m --n 10000 --seed $seed --out results/E7
  done
  [ -f results/E7/dyn_n10000_rho-5_s$seed.csv ] || CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.cifar bbb --n 10000 --seed $seed --out results/E7
done
