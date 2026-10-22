# Protocol: E20b — does the E20 exact-posterior result hold on other data subsets?

Written: 2026-10-22, after E20 and before any E20b run. Status: planned.
(SHA-256 recorded in `experiments/E20b_prereg.txt` before launch,
self-attested.)

Background: E20 (one HMC run per n, data seed 0, prior N(0, 0.3²), width 100)
converged and found that the exact posterior beats the softness-matched
network (Δ +0.07 to +0.11) and that BBB's MI-based Δ_K matches it within
0.006 on the same subset. Its main remaining limitation is one data subset
per n. BBB (σ_p = 0.3) and the width-100 DNN already exist for data seeds 1
and 2 (`results/E20/`, `results/E19/`); these were trained before E20's HMC
result was known and are not retrained.

- **Design:** exactly E20's HMC settings (`bnnstudy.hmc_batch`, 8 chains,
  identity mass, dual averaging towards 0.8 over 500 warm-up iterations,
  trajectory 0.471, ≤ 6000 leapfrog steps, 1000 sampling iterations, every
  10th kept), on data seeds 1 and 2 at n ∈ {500, 2000}. Comparators: BBB
  `bbb_steps23550_sp0.3` and the DNN with the same seed (same subset).
  Combined with E20 this gives 3 subsets per n.
- **Convergence:** per run, as in E20 (split-R̂ 90th percentile < 1.1 on
  p(true class) for 1000 test images). Only converged runs enter the
  predictions; non-converged runs are reported separately.
- **Predictions** (over the converged subsets at each n, seeds 0–2):
  Q1: HMC's entropy Δ_K and Δ_F are ≥ +0.03 on every subset (the E20 P1
  outcome replicates).
  Q2: the mean over subsets of |MI Δ_K(HMC) − MI Δ_K(BBB)| is ≤ 0.03.
  Q3: HMC's test accuracy is ≥ BBB's on every subset.
  Q4: HMC's mean entropy Δ_K exceeds BBB's mean entropy Δ_K by ≥ 0.02 (BBB's
  entropy recovers only part of the exact posterior's advantage).
- **Falsified if:** Q1 fails on ≥ 2 of the 4 new runs (then E20's headline
  was subset-specific).
- **Config:** `scripts/run_E20b.sh`, `scripts/analyze_E20b.py`; output
  `results/E20/hmcb_n<n>_d<seed>.pt`, `results/E20/seeds.csv`,
  `results/E20/seeds_summary.txt`.
- **Cost:** ~80 min on two RTX 3090s (n = 2000 then n = 500 on each GPU).
  Go-ahead: user delegated running the HMC follow-up overnight, 2026-10-22.
- **Decision it informs:** whether the paper can report the exact-posterior
  result with a spread over data subsets instead of a single run.

## Result (filled after running, 2026-10-22)
Files: `results/E20/hmcb_n{500,2000}_d{1,2}.pt`, `results/E20/seeds.csv`,
`results/E20/seeds_summary.txt`; logs `logs/E20b_d{1,2}.log`. Run time ~80 min.

- **Convergence: all 4 new runs converged** (split-R̂ 90th pct 1.004–1.005,
  max 1.015–1.022), so with E20 there are 3 converged subsets per n.

| n | subset | HMC acc | HMC Δ_F | HMC Δ_K | HMC MI Δ_K | BBB acc | BBB Δ_F | BBB Δ_K | BBB MI Δ_K |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 0 | 0.890 | +0.079 | +0.070 | +0.075 | 0.798 | −0.002 | −0.021 | +0.081 |
| 500 | 1 | 0.882 | +0.074 | +0.052 | +0.056 | 0.808 | +0.001 | −0.020 | +0.058 |
| 500 | 2 | 0.889 | +0.078 | +0.060 | +0.065 | 0.747 | −0.026 | −0.054 | +0.055 |
| 2000 | 0 | 0.929 | +0.115 | +0.074 | +0.077 | 0.909 | +0.077 | +0.033 | +0.072 |
| 2000 | 1 | 0.934 | +0.074 | +0.074 | +0.077 | 0.911 | +0.010 | +0.027 | +0.074 |
| 2000 | 2 | 0.930 | +0.075 | +0.077 | +0.080 | 0.899 | +0.018 | +0.029 | +0.069 |

- **Q1 CONFIRMED:** HMC Δ_K and Δ_F ≥ +0.03 on all 6 runs (Δ_K +0.052 to
  +0.077, Δ_F +0.074 to +0.115).
- **Q2 CONFIRMED:** mean |MI Δ_K(HMC) − MI Δ_K(BBB)| = 0.006 (n = 500) and
  0.007 (n = 2000); per subset 0.002–0.012.
- **Q3 CONFIRMED:** HMC more accurate than BBB on every subset (means 0.887
  vs 0.784; 0.931 vs 0.906).
- **Q4 CONFIRMED:** mean entropy Δ_K, HMC minus BBB: +0.092 (n = 500),
  +0.045 (n = 2000).
- Falsification condition not met (0 of 4 new runs fail Q1). CONFIRMATORY.
- Not predicted: on Fashion-MNIST BBB's MI falls clearly short of the exact
  posterior at n = 2000 (mean +0.047 vs +0.092), while on KMNIST it matches.
  HMC's Δ_F also varies more across subsets (+0.074 to +0.115) than its Δ_K.
- What it rules out: E20's headline being specific to one data subset.
