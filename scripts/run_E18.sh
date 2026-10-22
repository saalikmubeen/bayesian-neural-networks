#!/bin/bash
# E18 fresh BBB models (seeds 5-9). Usage: scripts/run_E18.sh <gpu> <method> <n>
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; M=$2; N=$3
for seed in 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $M --n $N --seed $seed --out results/E18 --save-model
done
