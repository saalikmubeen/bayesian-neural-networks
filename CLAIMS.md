# Claims → evidence

Every number in `report/bnn.tex`, with the file it comes from. Protocol
files (`experiments/<id>/protocol.md`) contain result tables computed from the
listed result files. Check with
`python3 scripts/check_claims.py report/bnn.tex CLAIMS.md`.

## Abstract and introduction
| Claim | Values | Source |
|---|---|---|
| Window advantage vs softness-matched DNN | +0.06 to +0.08 (E4: 0.057–0.076) | `experiments/E4_steps_vs_n/protocol.md`; `results/E45/paired_dnn.csv` |
| Proxy selection recovers 73% of oracle | 73 | `experiments/E9_selection/protocol.md`; `results/E9/selection.csv` |
| MLP on CIFAR-10: CIFAR-100 gain at most +0.05 | 0.046 max while learning | `experiments/E7b_cifar_mlp/protocol.md`; `results/E7b/summary.txt` |

## Setup
| Claim | Values | Source |
|---|---|---|
| Training sizes; validation pool; MLP widths | 500, 2,000, 10,000; 5,000; 784–256–256–10 | `bnnstudy/data.py`, `bnnstudy/models.py` |
| DNN: Adam 1e-3, 40 epochs, batch 64; MC dropout p 0.1, 50 samples; ensemble 5 | 10, 3, 40, 64, 0.1, 50, 5 | `bnnstudy/train.py` METHODS |
| BBB: rho init −5 (σ0 ≈ 0.007), N(0,1), 150 epochs, 4,800 steps at n = 2,000 | 5, 0.007, 0, 1, 150, 4,800 | `bnnstudy/models.py`, `bnnstudy/train.py` |

## §3.1 Grid (Table 1) and accuracy
| Claim | Values | Source |
|---|---|---|
| Table 1, all Δ and CIs | −0.046 [−0.063, −0.032], −0.035 [−0.041, −0.028], +0.014 [0.003, 0.027], +0.010 [0.008, 0.012], −0.003 [−0.016, 0.009], −0.019 [−0.021, −0.018], −0.072 [−0.088, −0.055], −0.112 [−0.122, −0.102], −0.024 [−0.048, −0.001], −0.010 [−0.014, −0.007], +0.015 [0.007, 0.022], +0.015 [0.013, 0.018], +0.007 [0.002, 0.012], −0.010 [−0.012, −0.007], −0.040 [−0.047, −0.032], −0.043 [−0.047, −0.038], +0.048 [0.038, 0.059], +0.069 [0.065, 0.074], +0.016 [0.008, 0.026], +0.023 [0.021, 0.025], −0.010 [−0.018, −0.001], −0.007 [−0.010, −0.003], −0.014 [−0.023, −0.006], −0.012 [−0.016, −0.008]; 10 seeds; ensemble +0.01 to +0.02 | `experiments/E2_mnist_grid/protocol.md`; `results/E2/paired_dnn.csv` |
| BBB vs budget-matched MAP accuracy: +1.1 pt (9/10), +0.5 pt (10/10), −0.5 pt | 1.1, 0.5, 9/10, 10/10 | `experiments/E3_map_baseline/protocol.md`; `results/E2/metrics.csv` |

## §3.2 Steps vs n (Figure 1)
| Claim | Values | Source |
|---|---|---|
| Budgets 23,550 and 4,800 steps; Δ +0.06 to +0.08 in 5/5 seeds | 23,550, 4,800, 5/5, 95 (% CI) | `experiments/E4_steps_vs_n/protocol.md`; `figures/paper_steps_vs_n.pdf` from `scripts/make_figures_paper.py` |

## §3.3 Window (Figure 2)
| Claim | Values | Source |
|---|---|---|
| 50,000 steps, eval every 1,000, 3 seeds per init | 50,000, 1,000, 3 | `scripts/run_E6.sh`, `bnnstudy/dynamics.py` |
| Δ negative first ~3,000 steps; peak +0.067 to +0.074 at 9,000–14,000; σ 0.65–0.82 | 3,000, 0.067, 0.074, 9,000, 14,000, 0.65, 0.82 | `experiments/E6_dynamics/protocol.md`; `results/E6/summary.txt` |
| Negative again at 25,000–30,000; −ELBO 143 (4,000–5,000), 6.8 (24,000–25,000), 3.5 (50,000) | 25,000, 30,000, 143, 4,000, 5,000, 6.8, 24,000, 3.5 | `results/E6/summary.txt` (P2, P4, end state) |
| End state σ ≈ 0.98 hidden, 0.93 last; acc 0.89; ECE 0.19 | 0.98, 0.93, 0.89, 0.19 | `results/E6/summary.txt` (rho −5: 0.985, 0.982, 0.932; acc 0.889; ECE 0.186) |
| σ0 0.049 reaches window at 3,000 vs 6,000; σ0 ≥ 0.31 at chance; −ELBO 9.0–9.2 | 0.049, 3,000, 6,000, 0.31, 9.0, 9.2 | `results/E6/summary.txt` (P3, end state) |

## §3.4 Mechanism
| Claim | Values | Source |
|---|---|---|
| fc1 + fc2 noise keeps 73–82%; first layer alone 16% / −26% | 73, 82, 16, 26 | `experiments/E6b_layer_ablation/protocol.md`; `results/E6b/summary.txt` |
| Constant pixels on average 192 / 148 / 104 | 192.4, 147.8, 104.6 | `results/E45/sigma_by_pixel.csv` (dead_px, 5 seeds) |
| Model-free rule AUROC 0.967–1.000 over 15 subsets; means 0.990–1.000 | 0.967, 1.000, 15, 0.990 | `results/E45/support_rule.csv` from `scripts/analyze_support_rule.py` (5 seeds × 3 n) |
| On-support: +0.057 → +0.055, +0.070 → +0.070 | 0.057, 0.055, 0.070 | `experiments/E5_on_support_ood/protocol.md` |
| σ dead/live ratio 1.0–1.24 | 1.0, 1.24 | `experiments/E4_steps_vs_n/protocol.md`; `results/E45/sigma_by_pixel.csv` |

## §3.5 Selection (Table 2) and window vs n
| Claim | Values | Source |
|---|---|---|
| NLL stops at 1,000–3,000 | 1,000, 3,000 | `experiments/E6_dynamics/protocol.md` (exploratory) |
| Table 2 | 10,917 +0.067 1.00 0.922; 10,417 +0.049 0.73 0.924; 13,750 +0.045 0.67 0.923; 39,417 +0.034 0.52 0.906; 2,250 −0.032 −0.49 0.922 | `experiments/E9_selection/protocol.md`; `results/E9/selection.csv` |
| 12 runs; 69–77%; no single proxy ≥ 70% | 12, 69, 77, 70 | same |
| Window opens 5,000–8,000; medians 10,000 / 11,000 / 11,000; closes 14,000–19,000 (6 runs); open at 40,000 | 5,000, 8,000, 10,000, 11,000, 14,000, 19,000, 40,000 | `results/E9/window_scaling.csv`; `experiments/E9_selection/protocol.md` |

## §4 CIFAR-10 (Figure 3)
| Claim | Values | Source |
|---|---|---|
| CNN: 3×3 convs 32/64/128, fc 256/10; n = 10,000; SVHN 10,000 | 3, 32, 64, 128, 256, 10,000 | `bnnstudy/cifar.py` |
| CNN peak Δ_SVHN −0.044, +0.006, −0.056; CIFAR-100 ≤ +0.031 | as written | `results/E7/summary.txt`; `experiments/E7_cifar/protocol.md` |
| Ensemble vs softened DNN: SVHN 0.727–0.740 vs 0.759–0.807; CIFAR-100 +0.02 to +0.03 | as written (0.023–0.029) | `results/E7/summary.txt` |
| CNN σ_f1 → 0.996; acc 0.67–0.68 → 0.61–0.63; conv-1 σ 0.04 | 0.996, 0.67, 0.68, 0.61, 0.63, 0.04 | `results/E7/summary.txt` |
| MLP DNN accuracy 38% / 46% | 0.381–0.384, 0.461–0.466 | `logs/E7b_gpu*.log` (`test acc`) |
| MLP peak Δ_SVHN −0.031 to +0.030; CIFAR-100 +0.030 to +0.046 while learning; CNN +0.024 to +0.031 | as written | `results/E7b/summary.txt`; `results/E7/summary.txt` |
| MLP collapse: σ_f1 0.997–0.999, acc 0.10–0.29; post-collapse gains +0.09 to +0.21; BBB acc 37–47% | as written | `results/E7b/summary.txt` |

## Limitations, appendix
| Claim | Values | Source |
|---|---|---|
| CIFAR checkpoints every 2,000 (CNN) / 1,000 (MLP) steps | 2,000, 1,000 | `scripts/run_E7.sh`, `scripts/run_E7b.sh` |
| KL slip: 62.5×, B = 64; 3.2 points (0.904 vs 0.936); ECE 0.022 vs 0.185 | 62.5, 64, 3.2, 0.904, 0.936, 0.022, 0.185 | `experiments/E1_pilot/protocol.md`; `results/E1_pilot/metrics.csv` |
| Ledger: 29 entries; 9 confirmed, 16 falsified, 3 mixed, 1 none | 29, 9, 16, 3, 1 | tally of the appendix table (each row from its protocol) |
| Ledger thresholds and outcomes: ±0.02, ±0.03, 0.035, 0.005, 5×, 70%, 30%, 20%, 67%, 0.31 | as written | `experiments/*/protocol.md` predictions |
| Appendix table column widths (layout, not a result; the checker reads "8" from "8.6cm") | 0.9, 8, 8.6, 4.6 cm | `report/bnn.tex` tabular spec |

## Additions after review 2 (no new compute)
| Claim | Values | Source |
|---|---|---|
| Wilcoxon with 5 seeds cannot fall below p = 0.0625 | 0.0625 | exact two-sided minimum for n = 5 |
| E6 KMNIST MI-Δ +0.073 (10k), +0.071 (30k), +0.061 (50k); test entropy 0.30 → 0.94 | 0.073, 0.071, 0.061, 0.30, 0.94 | `results/E6/dyn_n2000_rho-5_s*.csv` (mean of 3 seeds; `delta_kmnist_mi`, `H_id`) |
| Entropy-Δ KMNIST at 40k: +0.033 (n = 500), +0.060 (n = 10,000) | 0.033, 0.060 | `results/E9/dyn_n{500,10000}_rho-5_s*.csv` (mean of 3 seeds) |
| Collapsed runs (σ0 ≥ 0.31): entropy-Δ ≈ −0.86, MI-Δ ≈ +0.10 | 0.86, 0.10 | `results/E6/dyn_n2000_rho{-1,0}_s*.csv` (mean over checkpoints) |
| Mean fc1 ρ ≈ softplus⁻¹(mean σ) at 1k / 3k / 5k: −3.54/−3.52/−3.62, −1.88/−1.85/−1.92, −1.01/−0.99/−1.09 (n = 500 / 2,000 / 10,000) | as written | `results/E9/dyn_n*_rho-5_s*.csv` (`sigma_fc1`, mean of 3 seeds) |
| Fixed stop at 8k / 10k: Δ̄ +0.048 / +0.047, 72% / 71% of oracle; MI at last checkpoint +0.049 | 0.048, 0.047, 0.72, 0.71, 72, 0.049 | recomputed from `results/E9/dyn_*.csv` (12 runs) |
| CIFAR CNN SVHN MI-AUROC 0.24–0.60 | 0.238, 0.599 | `results/E7/dyn_*.csv` (`auroc_svhn_mi`) |
| CIFAR accuracy range 37–68% | 0.367 (E7b BBB max), 0.683 (E7 BBB max) | `results/E7b/summary.txt`, `results/E7/summary.txt` |
