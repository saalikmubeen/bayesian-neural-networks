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

## 9. E9 checkpoint selection with synthetic outliers ✅

Protocol: `experiments/E9_selection/`. 12 dynamics runs with proxies, ~20 min on 2 GPUs.

```bash
scripts/run_E9.sh 0 10000:-5 500:-5 > logs/E9_gpu0.log 2>&1 &
scripts/run_E9.sh 1 2000:-5 2000:-3 > logs/E9_gpu1.log 2>&1 &
$PY scripts/analyze_window_scaling.py           # window vs n (exploratory)
```

The selection table is computed by the inline script recorded in the E9
protocol result (`results/E9/selection.csv`). Result: combined proxies recover
73% of the oracle advantage; NLL stopping lands before the window. The window
peak does not move reliably with n, but it closes later at n = 10000.

---

## 10. E7 CIFAR-10 CNN and E7b CIFAR-10 MLP ✅

Protocols: `experiments/E7_cifar/`, `experiments/E7b_cifar_mlp/`. E7 ~1 h, E7b ~25 min on 2 GPUs.

```bash
scripts/run_E7.sh 0 0 2 > logs/E7_gpu0.log 2>&1 &
scripts/run_E7.sh 1 1   > logs/E7_gpu1.log 2>&1 &
scripts/run_E7b.sh 0 2000  > logs/E7b_gpu0.log 2>&1 &
scripts/run_E7b.sh 1 10000 > logs/E7b_gpu1.log 2>&1 &
```

Result: no OOD window on CIFAR-10 with either architecture. The drift towards
the prior replicates. Summaries: `results/E7/summary.txt`, `results/E7b/summary.txt`.

---

## 11. Report rewrite ✅

```bash
PYTHONPATH=. $PY scripts/analyze_support_rule.py     # results/E45/support_rule.csv
$PY scripts/make_figures_paper.py                    # figures/paper_steps_vs_n.pdf, paper_cifar.pdf
cd report && pdflatex bnn && bibtex bnn && pdflatex bnn && pdflatex bnn && cd .. && cp report/bnn.pdf bnn.pdf
python3 scripts/check_claims.py report/bnn.tex CLAIMS.md
```

Result: 9 pages; every number traced in `CLAIMS.md`.

---

## 12. E10 optimizer clock + E11 prior scale ✅

Protocols: `experiments/E10_optimizer_clock/`, `experiments/E11_prior_scale/`.
These were written before running, and their SHA-256 hashes were recorded at
launch in `experiments/E10_E11_prereg.txt`. 24 dynamics runs, ~45 min on 2 GPUs.

```bash
scripts/run_E10_E11.sh 0 a > logs/E10_E11_gpu0.log 2>&1 &   # E10 at n = 500 + E11
scripts/run_E10_E11.sh 1 b > logs/E10_E11_gpu1.log 2>&1 &   # E10 at n = 10000
$PY scripts/analyze_E10_E11.py      # results/E10/summary.txt, results/E11/summary.txt, figures/E10_clock.pdf
```

Result: under Adam, posterior σ grows on a learning-rate × steps clock that
is independent of n (all four predictions confirmed). Under SGD, n = 10000
barely moves σ and shows no OOD advantage. The prior-scale kill test passed
(the MI advantage survives σ_p = 0.1 and 0.3). With σ_p = 0.3 the entropy
advantage no longer collapses. σ_p = 0.1 underfits.

---

## 13. Stronger baselines, report update ✅

No training. These scripts compute the DNN energy, max-logit and oracle-temperature
scores from the saved E45 logits, and recompute the stopping table with one
estimator (ratio of means).

```bash
PYTHONPATH=. $PY scripts/analyze_strong_baselines.py     # results/E45/strong_baselines.csv
pdftoppm -png -r 150 -singlefile figures/E10_clock.pdf figures/E10_clock
cd report && pdflatex bnn && bibtex bnn && pdflatex bnn && pdflatex bnn && cd .. && cp report/bnn.pdf bnn.pdf
python3 scripts/check_claims.py report/bnn.tex CLAIMS.md
```

The ratio-of-means stopping table was computed with the inline script whose
output is saved in `results/E9/selection_ratio_of_means.txt`. Result:
longer-trained BBB beats every deterministic score in 20/20 runs (by ≥ 0.04).
The report is now 12 pages, with all 201 numbers traced.

---

## 14. E12 controls + E13 clean SGD arm ✅

Protocols: `experiments/E12_controls/`, `experiments/E13_sgd_clean/`. Hashes
were recorded before launch in `experiments/E12_E13_prereg.txt`. E13's first
run diverged and crashed; `bnnstudy/dynamics.py` now stops cleanly on
non-finite values (deviation noted in the E13 protocol).

```bash
scripts/run_E12_E13.sh 0 a > logs/E12_E13_gpu0.log 2>&1 &     # step-matched DNN/ensemble n = 2000 (+ first E13 runs)
scripts/run_E12_E13.sh 1 b > logs/E12_E13_gpu1.log 2>&1 &     # step-matched DNN/ensemble n = 10000
scripts/run_E13_resume.sh 0 500:0.05:sgd_mu:0 500:0.05:sgd_mu:1 500:0.05:sgd_mu:2 10000:1.0:sgd_scaled:2
scripts/run_E13_resume.sh 1 10000:0.05:sgd_mu:1 10000:0.05:sgd_mu:2 10000:1.0:sgd_scaled:0 10000:1.0:sgd_scaled:1
PYTHONPATH=. $PY scripts/analyze_E12.py all     # weight noise, step-matched, Mahalanobis/kNN -> results/E12/
$PY scripts/analyze_E13.py                      # results/E13/summary.txt
```

Result: BBB beats weight noise (+0.10 to +0.15), the step-matched DNN and
ensemble, and Mahalanobis, all 5/5. kNN comes within 0.02. Under SGD without
joint clipping, the dependence on n holds (53×), but runs diverge once σ grows
large. The report is now 14 pages, with all 216 numbers traced.

---

## 15. E14 IVON, E15 tempered IVON ✅

Protocols: `experiments/E14_ivon/`, `experiments/E15_ivon_inflation/`; hashes in
`experiments/E14_prereg.txt`, `E15_prereg.txt`. Needs `pip install ivon-opt==0.1.3`.

```bash
scripts/run_E14_tune.sh 0 0.05 0.1 & scripts/run_E14_tune.sh 1 0.2 0.5     # tuning, seed 99, validation NLL only
$PY scripts/analyze_E14.py tune                                             # -> lr 0.2, hess_init 0.1
scripts/run_E14.sh 0 0.2 0.1 10000:0 10000:1 500:0 500:1 2000:2 &
scripts/run_E14.sh 1 0.2 0.1 10000:2 2000:0 2000:1 500:2
$PY scripts/analyze_E14.py score
scripts/run_E15.sh 0 2000:0:100 2000:1:100 2000:2:100 2000:0:10 2000:1:10 2000:2:10 &
scripts/run_E15.sh 1 10000:0:100 10000:1:100 10000:2:100 10000:0:10 10000:1:10 10000:2:10
$PY scripts/analyze_E15.py
```

Result: IVON's σ follows 1/√N (4.41× vs 4.47×) and it has no advantage over
the softened network. Tempering either collapses it or gives no gain.

---

## 16. E16 swap test, E17 mechanism, E18 confirmation ✅

Protocols: `experiments/E16_swap/`, `E17_mechanism/`, `E18_r1_confirm/`;
hashes in `E16_prereg.txt` to `E18_prereg.txt`.

```bash
PYTHONPATH=. $PY scripts/analyze_E16.py           # swap / shuffle (inference only)
PYTHONPATH=. $PY scripts/analyze_E17.py           # noise-to-signal R1, R2 (inference only)
scripts/run_E18.sh 0 bbb_steps23550 2000 & scripts/run_E18.sh 1 bbb 10000     # fresh BBB, seeds 5-9
PYTHONPATH=. $PY scripts/analyze_E18.py
PYTHONPATH=. $PY scripts/make_figures_mechanism.py
```

Result: BBB's means and per-weight σ must be paired (overpruned posterior).
R₁ reproduces BBB's sampled OOD detection within 0.02 on fresh models and
unseen OOD sets. The report is now 17 pages, with all 251 numbers traced.

---

## Where things are

| Path | What |
|---|---|
| `bnnstudy/train.py` | one run → `results/<id>/<method>_n<n>_s<seed>.npz` + `.json` run info |
| `bnnstudy/evaluate.py` | all metrics + `dnn_Tnll` / `dnn_Tmatch[ref]` variants from the saved outputs |
| `scripts/run_E1.sh` | E1 pilot |
| `logs/` | run logs |

---

## 17. E19 HMC reference posterior (not converged)

Protocol: `experiments/E19_hmc/protocol.md`; hash in `E19_prereg.txt`.

```bash
scripts/run_E19_vi.sh 0                                 # width-100 BBB and DNN, seeds 0-2
scripts/run_E19.sh 0 0 & scripts/run_E19.sh 1 1         # HMC chains 0,1 (then 2,3)
scripts/run_E19.sh 0 2 & scripts/run_E19.sh 1 3
PYTHONPATH=. $PY scripts/analyze_E19.py
```

Result: chains failed the R̂ check at both n (each stayed near its MAP start),
so E19 is exploratory and the exact-posterior question remains open.

---

## 18. E20 HMC that mixes: prior N(0, 0.3²), 8 batched chains

Protocol: `experiments/E20_hmc_sp03/protocol.md`; hash in `E20_prereg.txt`.
Pilots (sampler diagnostics only, n = 500): `logs/E20_pilot_c*.log`,
`logs/E20_pilot2.log`, `logs/E20_probe_w20.log`, `logs/E20_probe_sp03.log`.

```bash
scripts/run_E20.sh 0 500 & scripts/run_E20.sh 1 2000    # HMC, then BBB (prior 0.3) seeds 0-2
PYTHONPATH=. $PY scripts/analyze_E20.py
PYTHONPATH=. $PY scripts/analyze_E20_extra.py           # exploratory follow-ups (inference only)
PYTHONPATH=. $PY scripts/make_figures_mechanism.py      # adds figures/E20_hmc.{pdf,png}
```

Result: converged at both n (split-R̂ ≤ 1.02). The exact posterior beats the
softness-matched network (+0.07 to +0.11); BBB's MI matches it on KMNIST
within 0.006. Its per-weight marginals equal the prior, and its information
lies in weight correlations.

---

## 19. E20b same HMC comparison on data subsets 1 and 2

Protocol: `experiments/E20b_hmc_seeds/protocol.md`; hash in `E20b_prereg.txt`.
BBB (prior 0.3) and DNN for seeds 1–2 come from sections 17–18.

```bash
scripts/run_E20b.sh 0 1 & scripts/run_E20b.sh 1 2      # HMC, data seeds 1 and 2 (n = 2000, then 500)
PYTHONPATH=. $PY scripts/analyze_E20b.py
PYTHONPATH=. $PY scripts/make_figures_mechanism.py      # E20 figure: mean and range over 3 subsets
```

Result: all runs converged; Q1–Q4 confirmed on every subset.

---

## 20. E21 Fashion-MNIST as in-distribution

Protocol: `experiments/E21_fashion_id/protocol.md`; hash in `E21_prereg.txt`.

```bash
scripts/run_E21.sh 0 0 2 4 & scripts/run_E21.sh 1 1 3      # train.py --id fashion, seeds 0-4
PYTHONPATH=. $PY -m bnnstudy.evaluate results/E21
PYTHONPATH=. $PY -m bnnstudy.compare results/E21
PYTHONPATH=. $PY scripts/analyze_E21.py
```

Result: the long-training advantage replicates (P2, P4); BBB already wins at
the standard budget (P1 falsified); the step-matched ensemble ties on KMNIST.

---

## 21. E22 held-out digits (near OOD) and post-hoc review analyses

Protocol: `experiments/E22_heldout_digits/protocol.md`; hash in `E22_prereg.txt`.

```bash
scripts/run_E22.sh 0 0 2 4 & scripts/run_E22.sh 1 1 3      # train.py --id mnist04, seeds 0-4
PYTHONPATH=. $PY -m bnnstudy.evaluate results/E22
PYTHONPATH=. $PY -m bnnstudy.compare results/E22
PYTHONPATH=. $PY scripts/analyze_E22.py
PYTHONPATH=. $PY scripts/analyze_posthoc_review.py      # temperature sweep, accuracy-controlled Delta (exploratory)
```

Result: falsified as pre-registered; on held-out digits the advantage is +0.011,
no better than step-matched deterministic models. BBB beats even the
oracle-temperature DNN on every far-OOD setting (50/50).

---

## 22. TMLR paper

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
python3 ../scripts/check_claims.py main.tex ../CLAIMS.md
```

---

## 23. E23 HMC controls (MAP, MAP ensemble under the prior; BBB 800 samples) and R₁ baselines

Protocol: `experiments/E23_hmc_controls_r1_baselines/protocol.md`; hash and amendment in `E23_prereg.txt`.

```bash
scripts/run_E23.sh 0 500 & scripts/run_E23.sh 1 2000
PYTHONPATH=. $PY scripts/analyze_E23.py         # part A
PYTHONPATH=. $PY scripts/analyze_E23_r1.py      # part B (E18 models, inference only)
```

Result: HMC beats a MAP ensemble under the same prior on every subset; 50
samples understate BBB's entropy Δ at n = 500; R₁ does not beat a pixel rule as
a predictor of BBB's far-OOD MI (falsified), but beats input statistics on
rotated digits.
