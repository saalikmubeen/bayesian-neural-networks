# Is the Bayesian OOD advantage just softness?: commands

Every command run so far, in order, so you can rerun everything yourself.
Run from the repo root:

```bash
cd /home/GTL/smubeen/CS8813/bayesian-neural-networks
PY=~/miniconda3/envs/python_38_dvces_cu111/bin/python   # torch 1.9 + CUDA 11.1
```

Status markers: ✅ done and verified · ⏸ prepared, not run yet · ⛔ stopped (reason given).

---

## 1. Smoke test of the new pipeline ✅

Trains one DNN (MNIST n=2000, seed 0) and saves per-sample outputs. Also
downloads MNIST, Fashion-MNIST and KMNIST into `data/` (gitignored).

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m bnnstudy.train --method dnn --n 2000 --seed 0 --out results/E1_pilot
```

Outputs: `results/E1_pilot/dnn_n2000_s0.{npz,json}`. Result: test acc 0.922
(earlier implementation: 0.924 ± 0.002).

---

## 2. E1 pilot: KL-weight bug and softness-matched baseline ✅

Protocol (committed before running): `experiments/E1_pilot/protocol.md`.
4 methods × 3 seeds at n=2000, then metrics and temperature variants.
~4 min on one RTX 3090.

```bash
CUDA_VISIBLE_DEVICES=1 scripts/run_E1.sh 2>&1 | tee logs/E1_pilot.log
# metrics only (re-runnable from the saved .npz files):
$PY -m bnnstudy.evaluate results/E1_pilot
```

Outputs: `results/E1_pilot/{metrics.csv,summary.txt}`. Result: fixing the KL
weight lifts BBB from acc 0.904 / ECE 0.185 to 0.936 / 0.022; a DNN with
matched softness (T = 2.05) equals or beats BBB on OOD AUROC (Fashion 0.866 vs
0.832, KMNIST 0.893 vs 0.886).

---

## 3. Repo cleanup and revised report (E1 only) ✅

Older exploratory scripts and outputs removed (they remain in git history).
The report is `report/bnn.tex`, covering E1 only; E2 is described as planned.
Figures are made from `results/E1_pilot/metrics.csv`.

```bash
$PY scripts/make_figures_E1.py                      # figures/E1_kl_bug.pdf, figures/E1_paired_auroc.pdf
cd report && pdflatex bnn && bibtex bnn && pdflatex bnn && pdflatex bnn && cd ..
cp report/bnn.pdf bnn.pdf
python3 scripts/check_claims.py report/bnn.tex CLAIMS.md
```

Result: 5-page report; 58 of 58 numbers traced in `CLAIMS.md`.

---

## 4. E2 MNIST grid ⏸ (running)

Protocol (committed before running): `experiments/E2_mnist_grid/protocol.md`.
Seeds 0–4 on GPU 0 and 5–9 on GPU 1, ~35 min. Resumable: finished runs are skipped.

```bash
scripts/run_E2.sh 0 0 1 2 3 4 > logs/E2_gpu0.log 2>&1 &
scripts/run_E2.sh 1 5 6 7 8 9 > logs/E2_gpu1.log 2>&1 &
# after both finish:
$PY -m bnnstudy.evaluate results/E2
$PY -m bnnstudy.compare results/E2
```

---

## 5. Report corrections after independent review ✅

Changes made after an independent review:
- The report now says pre-registered prediction P2 failed on Fashion-MNIST.
- The claims "best classifier" and "well calibrated" are removed.
- Henning et al. 2021, Kristiadi et al. 2020 and Liang et al. 2018 are cited.
- The mutual-information result is reported with its per-seed values.

```bash
cd report && pdflatex bnn && bibtex bnn && pdflatex bnn && pdflatex bnn && cd .. && cp report/bnn.pdf bnn.pdf
python3 scripts/check_claims.py report/bnn.tex CLAIMS.md
```

Result: 6 pages; 59 of 59 numbers traced.

---

## 6. E3 budget-matched MAP baseline ✅

Protocol (committed before running): `experiments/E3_map_baseline/protocol.md`.
This is a MAP network under BBB's N(0,1) prior, with BBB's epochs and gradient
clipping. Its runs go into `results/E2` so they pair with the E2 runs.

```bash
scripts/run_E3.sh 0 0 1 2 3 4 > logs/E3_gpu0.log 2>&1 &
scripts/run_E3.sh 1 5 6 7 8 9 > logs/E3_gpu1.log 2>&1 &
# then, once for E2 + E3:
$PY -m bnnstudy.evaluate results/E2
$PY -m bnnstudy.compare results/E2 dnn
$PY -m bnnstudy.compare results/E2 map
```

Result: BBB beats budget-matched MAP on accuracy by +1.1 pt (n = 500) and +0.5 pt
(n = 2000), and loses by 0.5 pt at n = 10000. On OOD it beats the
softness-matched MAP net only at n = 10000.

---

## 7. E4 steps vs n + E5 on-support OOD ✅

Protocols: `experiments/E4_steps_vs_n/`, `experiments/E5_on_support_ood/`.
Models are saved (`.pt`, gitignored). ~25 min on 2 GPUs.

```bash
scripts/run_E45.sh 0 0 1 2 > logs/E45_gpu0.log 2>&1 &
scripts/run_E45.sh 1 3 4   > logs/E45_gpu1.log 2>&1 &
$PY -m bnnstudy.evaluate results/E45
$PY -m bnnstudy.compare results/E45 dnn
$PY -m bnnstudy.compare results/E45 map
PYTHONPATH=. $PY scripts/analyze_E4.py         # sigma by pixel, figures/E4_sigma_maps.pdf
```

Result: the OOD gain follows optimizer steps, not n: +0.06 to +0.08 at 23550
steps, none at 4800, at every n. σ grows globally, not on dead pixels.
Removing dead-pixel ink changes nothing.

---

## 8. E6 training dynamics + E6b layer ablation ✅

Protocols: `experiments/E6_dynamics/`, `experiments/E6b_layer_ablation/`. E6: 12
runs × 50k steps, constant lr, eval every 1k steps (~25 min on 2 GPUs). E6b:
inference only on E4's saved models.

```bash
scripts/run_E6.sh 0 -5 -3 > logs/E6_gpu0.log 2>&1 &
scripts/run_E6.sh 1 -1 0  > logs/E6_gpu1.log 2>&1 &
$PY scripts/analyze_E6.py                      # results/E6/summary.txt, figures/E6_dynamics.pdf
PYTHONPATH=. $PY scripts/analyze_E6b.py        # results/E6b/layers.csv
```

Result: the OOD advantage is a transient window (peak +0.07 at 9k–14k steps,
gone by 25k). The ELBO keeps improving towards a near-prior, underfit
posterior. σ0 ≥ 0.31 never learns. The advantage needs noise in both hidden
layers.

---

## Where things are

| Path | What |
|---|---|
| `bnnstudy/train.py` | one run → `results/<id>/<method>_n<n>_s<seed>.npz` + `.json` run info |
| `bnnstudy/evaluate.py` | all metrics + `dnn_Tnll` / `dnn_Tmatch[ref]` variants from the saved outputs |
| `scripts/run_E1.sh` | E1 pilot |
| `logs/` | run logs |
