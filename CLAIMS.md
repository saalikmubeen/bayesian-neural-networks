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

## Additions with E10, E11 and the stronger-baseline analysis
| Claim | Values | Source |
|---|---|---|
| n = 2,000 Fashion: 3/10 seeds better, CI upper −0.001, Wilcoxon p = 0.16 | 0.16 | `results/E2/paired_dnn.csv` (bbb, n 2000, auroc_fashion_entropy) |
| Time to σ = 0.3: 10,000 / 5,000 / 3,000 (n = 500) and 12,000 / 6,000 / 3,000 (n = 10,000) for lr 5e-4 / 1e-3 / 2e-3; × lr within 12%; n changes it ≤ 20%; onset × lr within 7% | 10,000, 5,000, 3,000, 12,000, 6,000, 12, 20, 7 | `results/E10/summary.txt`, `results/E10/per_run.csv`; `experiments/E10_optimizer_clock/protocol.md` |
| SGD on ρ: σ at 20,000 steps 0.67 vs 0.017 (39×); peak Δ −0.007; accuracy 0.966 | 20,000, 0.67, 0.017, 39, 0.007, 0.966 | same |
| lr 2e-3 at n = 500 collapses (final acc 0.09–0.67) | 0.09, 0.67 | `results/E10/per_run.csv` |
| E10 setup: 3 seeds, 40,000 steps, SGD momentum 0.9 | 3, 40,000, 0.9 | `scripts/run_E10_E11.sh`, `bnnstudy/dynamics.py` |
| Prior scale: peak MI-Δ KMNIST +0.055 (0.1), +0.075 (0.3), +0.080 (1); entropy Δ at 40k +0.045 / +0.048 (0.3) vs −0.009 / +0.015 (1); σ reaches prior by ~10,000 steps; acc 0.870 vs 0.896 | as written | `results/E11/summary.txt`, `results/E11/per_run.csv`; `experiments/E11_prior_scale/protocol.md` |
| Stronger deterministic scores: best average 0.888–0.918; BBB 0.951–0.987 (entropy), 0.960–0.995 (MI); beats best in 20 of 20 pairs by ≥ 0.04 (min 0.0402) | 0.888, 0.918, 0.951, 0.987, 0.960, 0.995, 20, 0.04 | `results/E45/strong_baselines.csv` from `scripts/analyze_strong_baselines.py` |
| Stopping table (one estimator, ratio of means): proxies 0.74, fixed 8k 0.72, fixed 10k 0.71, −ELBO 0.51, NLL −0.48 | 74, 0.74, 0.51, 0.48 | `results/E9/selection_ratio_of_means.txt` |
| Ledger: 36 entries; 14 confirmed, 17 falsified, 4 mixed, 1 none | 36, 14, 17, 4, 1 | tally of the appendix table |
| SGD on ρ at n = 500: σ 0.67 at 20k, peak entropy Δ_K +0.03 (mean 0.0316), never ≥ 0.04, accuracy 0.882 | 0.67, 0.03, 0.04, 0.882 | `results/E10/summary.txt` (sgd (rho), n 500), `results/E10/per_run.csv` |
| DNN 40 epochs vs BBB 23,550 steps (budgets not matched) | 40, 23,550 | `bnnstudy/train.py` METHODS |

## Additions with E12 (controls), E13 (clean SGD) and the clock derivation
| Claim | Values | Source |
|---|---|---|
| Clock derivation: ρ* = softplus⁻¹(0.3) = −1.05; Δρ = 3.95; t* ≈ 3,950 / 7,900 / 1,975 steps at η = 1e-3 / 5e-4 / 2e-3; observed/predicted 1.27–1.52 | −1.05, 3,950, 7,900, 1,975, 1.27, 1.52, 0.3, 5 | arithmetic in `report/bnn.tex` §3.4 against `results/E10/per_run.csv` |
| Weight noise (softness-matched) AUROC 0.81–0.89; BBB beats by +0.10 to +0.15 (Fashion +0.11 to +0.14); oracle scale +0.07 to +0.10 | 0.81, 0.89, 0.10, 0.15, 0.11, 0.14, 0.07 | `results/E12/summary.txt`, `results/E12/noise.csv` |
| Step-matched softened DNN +0.040 to +0.087 (KMNIST +0.075 to +0.078); step-matched ensemble +0.041 to +0.075 (Fashion +0.041 to +0.058; KMNIST +0.058 to +0.075) | as written | `results/E12/summary.txt`, `results/E12/matched.csv` |
| Mahalanobis 0.86–0.94; BBB − Mahalanobis +0.021 to +0.109 (Fashion), +0.030 to +0.059 (KMNIST) | as written | `results/E12/features.csv` |
| kNN 0.93–0.97; BBB − kNN +0.006 to +0.023 (Fashion), +0.017 to +0.023 (KMNIST); tie at n = 10,000 Fashion (2/5) | as written | `results/E12/features.csv` |
| Abstract/finding: kNN within 0.02 | 0.02 | max BBB − kNN mean = 0.023 (rounded) — `results/E12/summary.txt` |
| E13: σ reaches 0.3 by 5,000 at n = 500 (3/3), never at n = 10,000; σ at 20k 0.945 vs 0.018 (53× = 0.9452 / 0.0177); n = 500 advantage +0.057 to +0.060; scaled runs t₀.₃ 5,000 in 2/3; 6 of 9 runs diverged (17,000–34,000 at n = 500; 5,000–9,000 scaled); scaled peak +0.022; ρ lr 1.0 | as written | `results/E13/summary.txt`, `results/E13/per_run.csv`; `experiments/E13_sgd_clean/protocol.md` |
| Ledger: 43 entries; 18 confirmed, 18 falsified, 6 mixed, 1 none | 43, 18, 6, 1 | scripted tally of the appendix rows |

## Additions after review 4 (no new compute)
| Claim | Values | Source |
|---|---|---|
| SGD clock: step ≈ 0.05·10/N = 5×10⁻⁵ at n = 10,000; after 20,000 steps less half of the 640-step warm-up, ρ ≈ −4.02, σ ≈ 0.018 (observed 0.0177 E13, 0.0173 E10); n = 500: 10⁻³ per step, σ = 0.3 after ≈ 4,300 steps (first checkpoint ≥ 0.3: 5,000) | 5, 10, −4.02, 0.018, 0.0177, 0.0173, 4,300, 640 | arithmetic (`python3 -c` in session; formula in report §3.4) against `results/E13/per_run.csv`, `results/E10/per_run.csv` |
| Scaled SGD runs crossed σ = 0.3 then diverged at 6,000 and 9,000 | 6,000, 9,000 | `results/E13/per_run.csv` (diverged_step) |
| Ledger after rescoring E13 P2: 43 entries; 18 confirmed, 19 falsified, 5 mixed, 1 none | 19, 5 | scripted tally of the appendix rows |

## Additions with E14–E18 (IVON, inflation, swap, mechanism, confirmation)
| Claim | Values | Source |
|---|---|---|
| Overpruning: hidden-layer median σ 0.71–0.84; |μ| > σ for 1.2–4.5% of hidden weights; fc2 corr(|μ|, σ) −0.64 to −0.72; last layer nearly deterministic | 0.71, 0.84, 1.2, 4.5, 0.64, 0.72 | `experiments/E16_swap/protocol.md` (exploratory follow-up, seed-0 models); `figures/E16_overpruning.pdf` |
| Swap test: plain-network means + BBB σ → accuracy 0.13–0.15 (13–15%); shuffled σ → accuracy 0.81 / 0.59, Δ −0.16 to −0.32 | 13, 15, 0.81, 0.59, 0.16, 0.32 | `results/E16/summary.txt`, `results/E16/variants.csv` |
| E17: second-layer ratio R₂ AUROC 0.45–0.69 (falsified) | 0.45, 0.69 | `results/E17/summary.txt` |
| E18 (fresh seeds 5–9): R₁ AUROC 0.960–0.991 on four far-OOD sets; within 0.003–0.019 of BBB MI; Spearman 0.71–0.83; rotated digits at n = 10,000: BBB 0.950, R₁ 0.863; ten fresh models; 90° rotation | 0.960, 0.991, 0.003, 0.019, 0.71, 0.83, 0.950, 0.863, 90 | `results/E18/summary.txt`, `results/E18/per_seed.csv`; `experiments/E18_r1_confirm/protocol.md` |
| E14 IVON: σ_fc1 0.170 / 0.086 / 0.039 (n = 500 / 2,000 / 10,000); ratio 4.41 vs √20 = 4.47; accuracy 0.930 / 0.969; peak Δ −0.005 / −0.008 vs BBB +0.07 / +0.065; BBB σ 0.96–0.98 | 0.170, 0.086, 0.039, 4.41, 4.47, 0.930, 0.969, 0.005, 0.008, 0.065 | `results/E14/summary.txt`, `results/E14/per_run.csv` |
| E15 tempered IVON: σ raised 3.1–9.3×; n = 10,000, k = 10: accuracy 0.951, Δ −0.004; k ∈ {10, 100} | 3.1, 9.3, 0.951, 0.004, 10, 100 | `results/E15/summary.txt`, `results/E15/per_run.csv` |
| Ledger: 61 entries; 26 confirmed, 24 falsified, 10 mixed, 1 none | 61, 26, 24, 10 | scripted tally of the appendix rows |
