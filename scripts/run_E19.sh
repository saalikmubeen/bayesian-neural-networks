#!/bin/bash
# E19 HMC chains (experiments/E19_hmc/protocol.md). Usage: scripts/run_E19.sh <gpu> <chain>  (runs n = 2000 then n = 500)
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1; C=$2
for n in 2000 500; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.hmc --n $n --chain $C --out results/E19
done
