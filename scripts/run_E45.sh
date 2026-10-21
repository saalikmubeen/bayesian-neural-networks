#!/bin/bash
# E4 (steps vs n) and E5 (on-support OOD): experiments/E4_steps_vs_n, experiments/E5_on_support_ood.
# Usage: scripts/run_E45.sh <gpu> <seeds...>     Resumable: finished runs are skipped.
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
OUT=results/E45
GPU=$1; shift
run() { CUDA_VISIBLE_DEVICES=$GPU $PY -W ignore -m bnnstudy.train --method $1 --n $2 --seed $3 \
          --out $OUT --on-support --save-model; }
for seed in "$@"; do
  for n in 2000 10000; do
    for m in dnn map bbb ensemble mc_dropout laplace; do run $m $n $seed; done
  done
  run bbb_steps23550 2000 $seed
  run bbb_steps4800 10000 $seed
  run dnn 500 $seed
  run bbb 500 $seed
  run bbb_steps23550 500 $seed
done
