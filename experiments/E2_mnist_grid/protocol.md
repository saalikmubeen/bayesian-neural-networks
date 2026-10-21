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

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
