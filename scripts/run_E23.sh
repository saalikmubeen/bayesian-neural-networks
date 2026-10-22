#!/bin/bash
# E23 part A: MAP and MAP ensemble under N(0, 0.3^2), width 100, on the HMC subsets (experiments/E23_hmc_controls_r1_baselines/protocol.md).
# Usage: scripts/run_E23.sh <gpu> <n>
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; N=$2
for seed in 0 1 2; do
  for m in map_sp0.3 map_ensemble_sp0.3; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $m --n $N --seed $seed --hidden 100 --out results/E23
  done
done
