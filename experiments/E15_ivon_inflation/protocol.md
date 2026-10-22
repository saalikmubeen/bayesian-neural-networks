# Protocol: E15 — does inflating IVON's variances create the OOD advantage?

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E15_prereg.txt` before any E15 run, self-attested.)

Background (E14): IVON's posterior σ follows N and curvature (σ_fc1 0.039–0.17)
and shows no OOD advantage over a softness-matched DNN, while BBB under Adam
inflates σ to 0.6–0.98 and does show one (+0.06 to +0.08). Interpretation to
test: BBB's advantage comes from variance inflation, not from a more accurate
posterior.

- **Hypothesis (causal):** if IVON's variances are inflated to BBB-like scales,
  IVON's OOD advantage appears too. Inflation is done by tempering the
  likelihood: effective sample size λ = N/k and weight decay δ = 1/(λσ_p²), so
  the posterior is ∝ p(D|w)^{1/k} p(w) with the same N(0,1) prior, and
  σ ≈ 1/√((N/k)h + 1/σ_p²) grows by about √k when the likelihood term dominates.
- **Predictions** (MNIST MLP, n ∈ {2000, 10000}, seeds 0–2, k ∈ {10, 100},
  40k steps, the E14 hyperparameters lr 0.2 and hess_init 0.1 unchanged; Δ as
  in E14; IVON with k = 1 from E14 is the reference):
  P1: σ_fc1 at 40k is ≥ 2× the k = 1 value for k = 10, and ≥ 5× for k = 100,
  at both n.
  P2 (main, causal): for k = 100, IVON's peak entropy-based Δ_KMNIST is
  ≥ +0.03 (mean over seeds) at both n.
  P3: for k = 10, the test accuracy at 40k is within 0.02 of k = 1.
- **Falsified if:** for both k ∈ {10, 100}, peak Δ_KMNIST < +0.02 at both n
  (then inflating variances alone does not produce BBB's advantage; its
  specific σ pattern or the mean weights must matter).
- **Config:** `bnnstudy.ivon_dynamics --temp k` (new flag: ess = N/k,
  weight_decay = 1/(ess·σ_p²)), σ_p = 1, β1 0.9, β2 0.99999, 1 MC sample per
  step, batch 64, constant lr, eval every 1000 steps with 20 posterior
  samples, FPR95 included. DNN base `results/E45/dnn_n<n>_s<seed>.npz`.
  Output `results/E15/`, tags `temp10`, `temp100`.
- **Cost:** 12 runs × ~3 min, ~20 min on 2 RTX 3090s. Go-ahead: user, 2026-10-22.
- **Decision it informs:** if P2 holds, the paper can state "the OOD advantage
  is produced by inflated predictive variance, whatever the optimizer", and
  BBB's clock is just how Adam happens to inflate it. If P2 fails, the
  advantage depends on something more specific about BBB.

## Result (filled after running, 2026-10-22)
Files: `results/E15/ivon_n*_s*_temp*.csv`, `results/E15/per_run.csv`,
`results/E15/summary.txt`, `figures/E15_inflation.pdf`; script
`scripts/analyze_E15.py`; logs `logs/E15_gpu*.log`. Reference k = 1 from E14.

| n | k | runs usable | σ_fc1 at 40k (× k=1) | acc at 40k | peak entropy Δ_K | peak MI Δ_K |
|---|---|---|---|---|---|---|
| 2000 | 1 | 3 | 0.086 | 0.930 | −0.005 | +0.003 |
| 2000 | 10 | 3 | 0.264 (3.06×) | 0.113 (collapsed) | −0.590 | +0.072 |
| 2000 | 100 | 0 | — | — (2 runs non-finite at step 1000; 1 crashed in temperature matching) | — | — |
| 10000 | 1 | 3 | 0.039 | 0.969 | −0.008 | −0.005 |
| 10000 | 10 | 3 | 0.121 (3.14×) | 0.951 | −0.004 | +0.016 |
| 10000 | 100 | 2 (1 crashed) | 0.360 (9.33×) | 0.135 (collapsed) | −0.594 | +0.103 |

- **P1 holds** wherever measurable (σ ratios 3.06, 3.14, 9.33).
- **P2 FAILED:** k = 100 produced no usable model at n = 2000 and a collapsed,
  chance-level model at n = 10000 (peak entropy Δ −0.59).
- **P3 holds at n = 10000** (−0.018) and **fails at n = 2000** (collapse,
  −0.817).
- **Falsification condition MET:** no tempered IVON model reaches peak
  entropy Δ_K ≥ +0.02. The only tempered model that still learns
  (n = 10000, k = 10, σ 3× larger, accuracy 0.951) has no advantage (−0.004).
- Reading: inflating IVON's variances does not reproduce BBB's advantage;
  either the model collapses or there is no gain. Together with the E12
  weight-noise control (noise added to a trained DNN), this suggests that
  variance size alone is not the mechanism. BBB's advantage seems to need
  large σ with mean weights that co-adapted to that noise during joint
  training. The variance-inflation interpretation offered after E14 is not
  supported as stated.
- Caveats: lr 0.2 and hess_init 0.1 were tuned for k = 1 and kept fixed as
  pre-registered; the collapses may partly reflect untuned settings for
  tempered IVON. Collapsed runs again show positive MI Δ (+0.07 to +0.10) at
  chance accuracy, as in E6.
- CONFIRMATORY for P1–P3 and the falsification condition as stated.
