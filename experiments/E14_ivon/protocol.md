# Protocol: E14 — does natural-gradient VI (IVON) escape the optimizer clock?

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E14_prereg.txt` before any E14 run, self-attested.)

Background: under Adam, BBB's posterior σ grows on an N-independent clock
(t* ≈ (ρ* − ρ₀)/η; E10, report §3.4), and BBB's OOD advantage over a
softness-matched DNN switches on with it and, under an N(0,1) prior, can fade
as σ drifts to the prior (E6). IVON (Shen et al., ICML 2024) is a
natural-gradient variational method whose posterior is
N(m, 1/(λ(h + δ))) with λ = effective sample size (= N here), h a running
Hessian estimate and δ the weight decay. Its variances are set by N and the
loss curvature, not by a step count.

- **Hypotheses:**
  H1: IVON's posterior σ depends on N (σ² ∝ 1/(N(h + δ))), unlike BBB under
  Adam, and stays far below the prior scale instead of drifting to it.
  H2: IVON's OOD behaviour therefore does not depend on training length once
  the Hessian estimate has settled: no transient.
  H3: IVON's predictive uncertainty is curvature-based (like a Laplace or
  ensemble posterior), so its advantage over a softness-matched DNN is
  positive but smaller than BBB's in-window advantage.
- **Predictions** (MNIST MLP 784-256-256-10, n ∈ {500, 2000, 10000}, seeds
  0–2, 40k steps, constant lr; σ_fc1 = mean of 1/√(λ(h + δ)) over fc1
  weights and biases; Δ = entropy-AUROC vs the DNN softened to IVON's
  validation entropy, as in E6/E9):
  P1: mean σ_fc1 at 40k steps is ≥ 2× larger at n = 500 than at n = 10000.
  P2: mean σ_fc1 at 40k steps is ≤ 0.3 at n = 2000 and n = 10000 (BBB:
  0.98 under the same N(0,1) prior).
  P3: no transient: |Δ_K(40k) − Δ_K(10k)| ≤ 0.02 (mean over seeds) at every n.
  P4: IVON's peak Δ_K (entropy) is below BBB's peak Δ_K at the same n (from
  E9, rho −5, same seeds), and ≥ 0 (mean over seeds) at n ∈ {2000, 10000}.
  P5: IVON's final test accuracy is within 0.01 of the 40-epoch DNN's
  (`results/E45/dnn_n<n>_s<seed>`) at each n (mean over seeds).
- **Falsified if:** P1 ratio < 1.5 (then IVON's variances do not track N
  either), or P3 fails at n ≥ 2000 (IVON also has a training-length effect).
- **Hyperparameters (fixed rule, decided now):** λ (ess) = N; weight decay
  δ = 1/(N·σ_p²) with σ_p = 1, i.e. the same N(0,1) prior as BBB; β1 = 0.9,
  β2 = 0.99999; 1 MC sample per step; batch 64; no clipping. The learning rate
  and initial Hessian are chosen once, on a separate subset (n = 2000,
  seed 99), from lr ∈ {0.05, 0.1, 0.2, 0.5} × hess_init ∈ {0.1, 1.0}, 10k
  steps each, by minimum *validation* NLL at the last checkpoint (no OOD data
  used). The chosen pair is used for all main runs.
- **Evaluation:** every 1000 steps, 20 posterior samples (as in E6/E9):
  accuracy, ECE, NLL, entropy- and MI-AUROC on Fashion-MNIST and KMNIST, Δ vs
  the softness-matched DNN, **FPR at 95% TPR** (new metric, requested by
  reviewers; also reported for the DNN control), σ per layer, and validation
  NLL. Synthetic-outlier proxies as in E9.
- **Cost:** tuning 8 × ~1 min; main 9 runs × ~4 min; ~45 min on 2 RTX 3090s.
  Go-ahead: user ("work on next target"), 2026-10-22.
- **Decision it informs:** if P1–P3 hold, the paper can say the clock is a
  property of BBB under Adam that natural-gradient VI avoids, which is a
  practical recommendation. P4 tells whether that comes at the cost of OOD
  performance.

## Result (filled after running, 2026-10-22)
Tuning (seed 99, validation NLL only; `results/E14/tune/choice.txt`): lr 0.2,
hess_init 0.1 (val NLL 0.309; no configuration diverged).
Main runs: `results/E14/ivon_n*_s*.csv` (9 runs, none diverged),
`results/E14/per_run.csv`, `results/E14/summary.txt`,
`figures/E14_ivon_vs_bbb.pdf`; script `scripts/analyze_E14.py`; logs `logs/E14_*.log`.
BBB reference: E9 (rho −5, same seeds and subsets).

| n | IVON σ_fc1 10k / 40k | BBB σ_fc1 40k | IVON Δ_K 10k / 40k | IVON peak Δ_K (MI) | BBB peak Δ_K | IVON acc | DNN acc | FPR95 KMNIST IVON / DNN_T |
|---|---|---|---|---|---|---|---|---|
| 500 | 0.147 / 0.170 | 0.983 | −0.038 / −0.081 | −0.031 (−0.018) | +0.059 | 0.859 | 0.876 | 0.76 / 0.52 |
| 2000 | 0.074 / 0.086 | 0.982 | −0.024 / −0.021 | −0.005 (+0.003) | +0.071 | 0.930 | 0.925 | 0.61 / 0.55 |
| 10000 | 0.033 / 0.039 | 0.956 | −0.019 / −0.012 | −0.008 (−0.005) | +0.065 | 0.969 | 0.968 | 0.49 / 0.49 |

- **P1 CONFIRMED:** σ_fc1 at 40k is 4.41× larger at n = 500 than at n = 10000,
  close to the σ ∝ 1/√N expectation (√20 = 4.47).
- **P2 CONFIRMED:** IVON's σ stays at 0.086 (n = 2000) and 0.039 (n = 10000),
  far below the prior scale; BBB reaches 0.98 and 0.96.
- **P3 holds at n = 2000 and 10000** (|change| 0.004 and 0.007) and **fails at
  n = 500** (Δ_K drifts −0.038 → −0.081).
- **P4 mixed:** IVON's peak Δ is below BBB's at every n (as predicted), but it
  is *negative* at n ≥ 2000 (−0.005, −0.008), not ≥ 0: IVON has no OOD advantage
  over the softness-matched DNN, by entropy or MI. FPR95 agrees (IVON worse
  than or equal to the softened DNN).
- **P5 holds at n ≥ 2000** (+0.005, +0.001) and **fails at n = 500** (−0.017).
- Reading: IVON escapes the optimizer clock as derived (variances set by N and
  curvature, stable over training), but then shows no OOD advantage beyond
  softness. BBB's advantage under Adam coincides with σ inflated far beyond
  IVON's curvature-based values (0.6–0.98 vs 0.04–0.17), which suggests the
  advantage comes from optimizer-driven variance inflation rather than from a
  more accurate posterior. EXPLORATORY interpretation; one architecture, IVON
  with one tuned setting, no exact-posterior reference yet.
- CONFIRMATORY for P1–P5 as stated.
