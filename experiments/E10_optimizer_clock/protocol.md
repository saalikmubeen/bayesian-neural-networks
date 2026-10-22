# Protocol: E10 — does Adam set an N-independent clock for the posterior variances?

Written: 2026-10-22, before running. Status: planned. (Not committed before
running; the SHA-256 of this file and the launch time are recorded in
`logs/E10_E11_prereg.txt` before the first run starts.)

Background: in E9 (Adam, lr 1e-3, rho_init −5) the mean first-layer ρ follows
nearly the same path for n = 500, 2000 and 10000 (≈ −3.5 / −1.9 / −1.0 at
1k / 3k / 5k steps), and BBB's OOD gain follows optimizer steps rather than n
(E4).

- **Hypothesis (with mechanism):** early in training the gradient on ρ is
  dominated by the KL term, whose size per example scales as 1/N. Adam
  normalises each update to a step of about lr, so ρ moves by ~lr per step
  whatever N is. Then the time for σ to grow scales with 1/lr and not with N.
  Under plain SGD (with momentum) the step size is proportional to the
  gradient, so ρ should move ~N times more slowly at larger N.
- **Predictions** (mean first-layer σ per checkpoint; t₀.₃ = first checkpoint
  with mean σ_fc1 ≥ 0.3; 3 seeds each; E9 runs supply Adam lr 1e-3):
  P1 (lr clock): with Adam, t₀.₃ × lr is within ±30% of its mean across
  lr ∈ {5e-4, 1e-3, 2e-3}, at both n = 500 and n = 10000.
  P2 (no N-dependence under Adam): for each lr, t₀.₃ at n = 10000 is within
  ±25% of t₀.₃ at n = 500.
  P3 (N-dependence under SGD): with SGD on ρ (lr_ρ 0.05, momentum 0.9; μ on
  Adam 1e-3), mean σ_fc1 at 20k steps is at least 3× larger at n = 500 than at
  n = 10000 (or t₀.₃ is reached at n = 500 but not at n = 10000 within 40k steps).
  P4 (OOD follows the clock): with Adam, the first step at which the
  entropy-based Δ_KMNIST ≥ 0.04 scales with 1/lr: onset × lr within ±40% of
  its mean across the three lr (at n = 10000, where the window does not close).
- **Falsified if:** P1 spread > 30% at either n, or the P3 ratio < 2.
- **Config:** `bnnstudy.dynamics`, MNIST MLP, rho_init −5, N(0,1) prior,
  constant lr, batch 64, clip 1.0, 40k steps, eval every 1000 steps, 20 MC
  samples, `--proxies`. Arms: Adam lr 5e-4 (tag `lr5e-4`), Adam lr 2e-3
  (tag `lr2e-3`), SGD on ρ (tag `sgd`), each at n ∈ {500, 10000}, seeds
  0–2 (18 runs). Reference: E9 runs (Adam lr 1e-3, same seeds). DNN base:
  `results/E45/dnn_n<n>_s<seed>.npz`. Output `results/E10/`.
- **Cost:** 18 runs × ~3 min, ~30 min on 2 RTX 3090s. Go-ahead: user, 2026-10-22.
- **Decision it informs:** if P1–P3 hold, the clock is a finding, not a
  hypothesis, and the paper can state "BBB's OOD behaviour is set by lr × steps
  under Adam". If not, the steps-not-data result stays an unexplained observation.

## Result (filled after running, 2026-10-22)
Files: `results/E10/dyn_*.csv` (18 runs), `results/E10/per_run.csv`,
`results/E10/summary.txt`, `figures/E10_clock.pdf`; script
`scripts/analyze_E10_E11.py`; logs `logs/E10_E11_gpu*.log`. Reference arm
(Adam 1e-3) from E9, same seeds. t₀.₃ is measured on a 1000-step grid, so at
lr 2e-3 (3000 steps) the resolution is ±33%.

| n | arm | t₀.₃ (steps, all 3 seeds) | onset Δ_K ≥ 0.04 | σ_fc1 at 20k | acc at 40k |
|---|---|---|---|---|---|
| 500 | Adam 5e-4 | 10000, 10000, 10000 | 12k–14k | 0.83 | 0.883 |
| 500 | Adam 1e-3 | 5000 ×3 | 7k–8k | 0.96 | 0.882 |
| 500 | Adam 2e-3 | 3000 ×3 | 5k | 0.98 | 0.67 / 0.20 / 0.09 (collapse) |
| 500 | SGD on ρ | 9000 ×3 | never | 0.67 | 0.882 |
| 10000 | Adam 5e-4 | 12000 ×3 | 10k–12k | 0.55 | 0.960 |
| 10000 | Adam 1e-3 | 6000 ×3 | 5k–6k | 0.85 | 0.951 |
| 10000 | Adam 2e-3 | 3000 ×3 | 2k–4k | 0.96 | 0.948 |
| 10000 | SGD on ρ | never (σ 0.017) | never | 0.017 | 0.966 |

- **P1 CONFIRMED:** t₀.₃ × lr = 5 / 5 / 6 (n = 500) and 6 / 6 / 6 (n = 10000)
  for lr 5e-4 / 1e-3 / 2e-3; max deviation from the mean 12% and 0%.
- **P2 CONFIRMED:** t₀.₃ ratio n = 10000 / n = 500 is 1.20, 1.20, 1.00.
- **P3 CONFIRMED:** under SGD on ρ, σ_fc1 at 20k steps is 0.67 (n = 500)
  vs 0.017 (n = 10000), a 39× ratio. n = 10000 never reaches σ = 0.3.
- **P4 CONFIRMED:** onset × lr at n = 10000 = 5.5 / 5.3 / 6.0 (max deviation 7%).
- Not predicted: (a) with SGD at n = 10000, σ stays tiny and BBB has *no* OOD
  advantage (peak Δ_K −0.007) while reaching the best accuracy (0.966), which is
  direct evidence that σ growth is what produces the advantage. (b) At
  n = 500, SGD on ρ grows σ but gives only a small advantage (peak Δ_K +0.03,
  onset criterion never met). (c) Adam lr 2e-3 at n = 500 collapses (accuracy
  0.09–0.67 at 40k). (d) Plotted against lr × t, the Adam σ curves coincide
  during growth; n only shifts the plateau slightly (lower at n = 10000).
- CONFIRMATORY for P1–P4.
