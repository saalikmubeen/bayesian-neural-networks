#!/bin/bash
# E19 width-100 BBB and DNN on the same subsets (seeds 0-2). Usage: scripts/run_E19_vi.sh <gpu>
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
GPU=$1
for n in 2000 500; do
  for seed in 0 1 2; do
    for m in dnn bbb_steps23550; do
      CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $m --n $n --seed $seed --hidden 100 --out results/E19 --save-model
    done
  done
done
