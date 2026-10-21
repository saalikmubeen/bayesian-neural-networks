#!/bin/bash
# E1 pilot: KL-weight bug and softness-matched baseline (experiments/E1_pilot/protocol.md)
set -e
PY=${PY:-~/miniconda3/envs/python_38_dvces_cu111/bin/python}
OUT=results/E1_pilot
for seed in 0 1 2; do
  for m in dnn bbb bbb_bug bbb_legacy; do
    $PY -m bnnstudy.train --method $m --n 2000 --seed $seed --out $OUT
  done
done
$PY -m bnnstudy.evaluate $OUT
