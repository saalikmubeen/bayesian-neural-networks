# Protocol: E3 — budget-matched MAP baseline

Written: 2026-10-21, before running. Status: planned

Motivation: an independent review. BBB trains for 150
epochs with gradient clipping and an N(0,1) prior. The DNN trains for 40
epochs with no weight decay or clipping. BBB's accuracy lead over the DNN
(+1.1 points in E1) could come from extra regularisation and compute rather
than from the posterior.

- **Hypothesis (with mechanism):** BBB's posterior mean already behaves like a
  MAP estimate under the same prior. A deterministic network trained to MAP
  under N(0,1) (L2 = ||w||² / (2N)), with BBB's epochs, learning rate,
  schedule and clipping, matches BBB's accuracy. Any accuracy lead of BBB
  over the plain DNN is therefore regularisation, not Bayesian averaging.
- **Predictions** (paired over seeds 0–9 at each n ∈ {500, 2000, 10000}):
  P1: |acc(bbb) − acc(map)| < 0.005 at n = 2000 and 10000.
  P2: OOD AUROC (entropy): Δ = bbb − map_Tmatch[bbb] has a 95% CI including 0
  or below 0 on both OOD sets at every n (no advantage beyond softness against
  the stronger base either).
- **Falsified if:** P1 — BBB more accurate than MAP by ≥ 0.005 with Wilcoxon
  p < 0.05 at n = 2000 or 10000 (then averaging helps accuracy). P2 — BBB's Δ > 0
  with CI excluding 0 on either OOD set at any n.
- **Config:** method `map` = MLP 784-256-256-10, Adam 1e-3, cosine schedule,
  150 epochs at n ≥ 2000 (×4 at n = 500, as for BBB), batch 64, gradient
  clipping 1.0, L2 ||w||²/(2·1²·N) on all weights and biases. Runs saved in
  `results/E2/` next to the E2 runs (same seeds, subsets and splits) so they
  pair. Temperature variants `map_Tnll`, `map_Tmatch[<ref>]` are fitted on
  validation only, as for the DNN.
- **Metrics:** accuracy (primary for P1); OOD AUROC with entropy, paired with
  bootstrap CI and Wilcoxon (primary for P2); NLL, ECE.
- **Cost:** 30 runs, ~10–15 min on one RTX 3090. Go-ahead given by user
  2026-10-21 ("queue the budget-matched DNN after E2").
- **Decision it informs:** if P1 holds, the report drops any accuracy claim for
  BBB and uses `map` as the main deterministic baseline. If P1 fails, BBB's
  accuracy gain is a (small) genuine benefit of posterior averaging to report.

## Result (filled after running, 2026-10-21)
Files: `results/E2/map_*`, `results/E2/metrics.csv`, `paired_map.csv`, log `logs/E3_gpu*.log`.

Accuracy (10 seeds, Wilcoxon on bbb − map):

| n | bbb | map | dnn | bbb − map | bbb higher | p | ECE map / bbb |
|---|---|---|---|---|---|---|---|
| 500 | 0.8875 | 0.8763 | 0.8779 | +0.0112 | 9/10 | 0.004 | 0.044 / 0.032 |
| 2000 | 0.9385 | 0.9334 | 0.9264 | +0.0051 | 10/10 | 0.002 | 0.030 / 0.023 |
| 10000 | 0.9640 | 0.9689 | 0.9674 | −0.0049 | 0/10 | 0.002 | 0.016 / 0.045 |

OOD, Δ = AUROC_entropy(M) − AUROC_entropy(map_Tmatch[M]) [95% CI] (wins/10):

| n | bbb Fashion | bbb KMNIST | ensemble Fashion | ensemble KMNIST |
|---|---|---|---|---|
| 500 | +0.014 [0.001, 0.026] (7) | −0.022 [−0.028, −0.016] (0) | +0.059 [0.036, 0.086] (10) | +0.016 [0.012, 0.019] (10) |
| 2000 | −0.021 [−0.042, −0.007] (2) | −0.020 [−0.023, −0.017] (0) | +0.012 [−0.003, 0.028] (6) | +0.001 [−0.002, 0.004] (6) |
| 10000 | **+0.044 [0.034, 0.057] (10)** | **+0.054 [0.051, 0.057] (10)** | +0.010 [−0.001, 0.022] (7) | +0.006 [0.004, 0.008] (10) |

- **P1 falsified (narrowly) at n = 2000:** BBB beats budget-matched MAP by
  0.0051 (10/10, p = 0.002), just over the 0.005 bar. At n = 500 the gain is
  +0.011. At n = 10000 MAP is *better* by 0.005 and better calibrated.
  Posterior averaging buys a small accuracy gain only in the small-data regime.
  About half of BBB's lead over the plain DNN at n = 2000 (+0.012) is
  regularisation (MAP +0.007 over DNN).
- **P2 falsified at n = 10000** (as in E2), and marginally on Fashion at n = 500.
- Not predicted: against the stronger MAP base, the ensemble's OOD edge mostly
  vanishes at n = 2000 and is small at n = 10000. Part of the ensemble's E2
  advantage came from the weak DNN baseline.
- CONFIRMATORY (P1, P2 as stated).
