#!/bin/bash
# E7b MLP on CIFAR-10 (experiments/E7b_cifar_mlp/protocol.md). Usage: scripts/run_E7b.sh <gpu> <n>
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; n=$2
for seed in 0 1 2; do
  [ -f results/E7b/dnn_n${n}_s$seed.npz ] || CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.cifar dnn --arch mlp --n $n --seed $seed --out results/E7b
  [ -f results/E7b/dyn_n${n}_rho-5_s$seed.csv ] || CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.cifar bbb --arch mlp --n $n --seed $seed --every 1000 --out results/E7b
done
