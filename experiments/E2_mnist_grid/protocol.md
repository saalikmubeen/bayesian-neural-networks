# Protocol: E2 — MNIST grid: which Bayesian advantages survive the softness control?

Written: 2026-10-21, before running. Status: planned

- **Hypothesis (with mechanism):**
  H2a (from E1): for mean-field VI (BBB), the OOD advantage over a DNN is
  explained by global softness, so it disappears against a DNN whose
  temperature matches BBB's mean validation entropy.
  H2b: deep ensembles carry information beyond softness (members disagree
  more on OOD inputs than on ID inputs, which no single temperature can
  reproduce), so their advantage survives.
  H2c: MC dropout (p = 0.1) and last-layer Laplace behave like BBB (≈ softness).
  H2d (β sweep): a larger KL weight β widens the posterior: accuracy falls and
  MI-based OOD AUROC rises.
- **Predictions** (paired over 10 seeds, Δ = AUROC_entropy(M) − AUROC_entropy(dnn_Tmatch[M])):
  P1: BBB — 95% CI of Δ includes 0 or lies below 0, on both OOD sets, at every n.
  P2: ensemble — Δ > 0 with CI excluding 0 on both OOD sets at n = 2000 and 10000.
  P3: MC dropout and Laplace — |mean Δ| < 0.02 at every n.
  P4: β sweep (n = 2000) — test accuracy decreases and KMNIST MI-AUROC
  increases monotonically (Spearman ρ with β: < 0 and > 0) over β ∈ {0.1, 0.3, 1, 3, 10, 30}.
- **Falsified if:** P1 — BBB Δ > 0 with CI excluding 0 on either OOD set at
  any n (a real Bayesian advantage beyond softness). P2 — ensemble CI includes
  0 or lies below 0 (then even ensembles are "just softness" here).
- **Config:** as E1 (`bnnstudy`, MNIST → Fashion-MNIST, KMNIST; 5k validation
  pool disjoint from training; temperature fitted on validation only).
  n ∈ {500, 2000, 10000}; seeds 0–9; methods dnn, bbb, mc_dropout (p = 0.1,
  50 samples), ensemble (5 members), laplace (last layer, full GGN, prior
  precision on validation NLL). Epochs as E1 at n = 2000; for n = 500 epochs
  ×4 (same number of steps as n = 2000); n = 10000 same epochs as n = 2000.
  β sweep: bbb_beta{0.1, 0.3, 3, 10, 30} and bbb (β = 1) at n = 2000, seeds 0–9.
  Hyperparameters are fixed, not tuned per n (a stated limitation).
- **Metrics:** primary — Δ AUROC (entropy score), paired by seed, mean with
  95% bootstrap CI and Wilcoxon p. Secondary — Δ with M's mutual information
  as score; accuracy, NLL, ECE, AURC.
- **Cost:** ~1 GPU-hour total, two RTX 3090s in parallel. Go-ahead given by
  user 2026-10-21.
- **Decision it informs:** if P1–P3 hold and P2 holds → paper message
  "only ensembles beat the softness control"; then HMC (E3) decides whether
  an exact posterior behaves like ensembles or like VI. If P2 fails → "no
  method beats a matched temperature on MNIST"; then CIFAR-10 (E4) is
  essential before any general claim.

## Result (filled after running, 2026-10-21)
Files: `results/E2/` (200 runs). Analysis with `bnnstudy.evaluate` and
`bnnstudy.compare` (base `dnn`); for this table the MAP runs added later by E3
were excluded. Δ = AUROC_entropy(M) − AUROC_entropy(dnn_Tmatch[M]), 10 seeds,
95% bootstrap CI, Wilcoxon p.

| n | method | Δ Fashion [CI] (wins/10) | Δ KMNIST [CI] (wins/10) |
|---|---|---|---|
| 500 | bbb | −0.046 [−0.063, −0.032] (0) | −0.035 [−0.041, −0.028] (0) |
| 500 | ensemble | +0.014 [0.003, 0.027] (8) | +0.010 [0.008, 0.012] (10) |
| 500 | mc_dropout | −0.003 [−0.016, 0.009] (4) | −0.019 [−0.021, −0.018] (0) |
| 500 | laplace | −0.072 [−0.088, −0.055] (0) | −0.112 [−0.122, −0.102] (0) |
| 2000 | bbb | −0.024 [−0.048, −0.001] (3) | −0.010 [−0.014, −0.007] (0) |
| 2000 | ensemble | +0.015 [0.007, 0.022] (8) | +0.015 [0.013, 0.018] (10) |
| 2000 | mc_dropout | +0.007 [0.002, 0.012] (9) | −0.010 [−0.012, −0.007] (0) |
| 2000 | laplace | −0.040 [−0.047, −0.032] (0) | −0.043 [−0.047, −0.038] (0) |
| 10000 | bbb | **+0.048 [0.038, 0.059] (10)** | **+0.069 [0.065, 0.074] (10)** |
| 10000 | ensemble | +0.016 [0.008, 0.026] (9) | +0.023 [0.021, 0.025] (10) |
| 10000 | mc_dropout | −0.010 [−0.018, −0.001] (3) | −0.007 [−0.010, −0.003] (2) |
| 10000 | laplace | −0.014 [−0.023, −0.006] (2) | −0.012 [−0.016, −0.008] (0) |

β sweep (n = 2000, mean of 10 seeds): β 0.1 / 0.3 / 1 / 3 / 10 / 30 → acc
0.937 / 0.938 / 0.939 / 0.937 / 0.933 / 0.920; KMNIST MI AUROC 0.890 / 0.889 /
0.896 / 0.908 / 0.925 / 0.942; Fashion MI AUROC 0.886 / 0.876 / 0.859 / 0.858 /
0.855 / 0.851. Spearman with β (per run): acc −0.63, KMNIST MI +0.88.

- **P1 FAILED at n = 10000:** BBB beats the softness control in 10/10 seeds on
  both OOD sets (CI excludes 0). It holds at n = 500 and 2000, where BBB is
  significantly *worse* than the control.
- **P2 CONFIRMED:** ensembles beat the control at every n, including n = 500.
- **P3 split:** MC dropout |mean Δ| ≤ 0.019 everywhere (holds). Last-layer
  Laplace is far *below* the control (−0.012 to −0.112), so P3 fails for Laplace.
- **P4 CONFIRMED** for accuracy (↓) and KMNIST MI (↑). Not predicted: Fashion
  MI *decreases* with β.
- CONFIRMATORY for P1–P4 as stated. The interpretation of the n = 10000 result
  (pixel-support shortcut switched on by step count) is EXPLORATORY and is
  tested in E4/E5.
