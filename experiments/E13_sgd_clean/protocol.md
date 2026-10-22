# Protocol: E13 — the SGD arm without joint clipping, and with a learning rate scaled by N

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E12_E13_prereg.txt` before launch, self-attested.)

Background: in E10, SGD on ρ grew σ 39× faster at n = 500 than at
n = 10000. But gradient clipping acted on all parameters jointly
(`bnnstudy/dynamics.py`), so the ρ step also depended on the cross-entropy
gradient, and only one ρ learning rate (0.05) was tried.

- **Hypothesis (with mechanism):** under SGD the KL gradient on ρ is about
  1/N per weight while σ is small, so the ρ step is ∝ lr_ρ / N. Removing joint
  clipping should not change that. Scaling lr_ρ ∝ N should make the σ
  trajectories line up across n, the same as Adam's clock.
- **Predictions** (n ∈ {500, 10000}, seeds 0–2, SGD with momentum 0.9 on ρ, μ
  on Adam 1e-3, clipping on μ only; t₀.₃ = first checkpoint with mean
  σ_fc1 ≥ 0.3):
  P1 (fixed lr_ρ = 0.05): mean σ_fc1 at 20k steps is ≥ 3× larger at n = 500
  than at n = 10000.
  P2 (lr_ρ = 0.05 · N / 500, i.e. 0.05 at n = 500 and 1.0 at n = 10000):
  t₀.₃ at n = 10000 is within ±25% of t₀.₃ at n = 500.
  P3: with the scaled lr_ρ at n = 10000, the entropy-based Δ_KMNIST (vs the
  softness-matched DNN) reaches ≥ +0.04 at some checkpoint, unlike the fixed
  lr_ρ arm in E10 (peak −0.007).
- **Falsified if:** P1 ratio < 2 (then E10's N-dependence came from joint
  clipping), or P2 ratio outside 0.5–2 (then lr_ρ ∝ N does not align the
  clocks).
- **Config:** `bnnstudy.dynamics --rho-optimizer sgd --clip mu`, rho_init −5,
  N(0,1) prior, constant Adam lr 1e-3 for μ, 40k steps, eval every 1000 steps,
  20 MC samples, `--proxies`. Arms: tag `sgd_mu` (lr_ρ 0.05) and `sgd_scaled`
  (lr_ρ 0.05·N/500) at n ∈ {500, 10000}; at n = 500 the two arms coincide, so
  `sgd_scaled` is only run at n = 10000. Seeds 0–2 (9 runs). DNN base
  `results/E45/dnn_n<n>_s<seed>.npz`. Output `results/E13/`.
- **Cost:** 9 runs × ~3.5 min, ~20 min on 2 GPUs. Go-ahead: user, 2026-10-22.
- **Decision it informs:** whether the SGD contrast in the paper can be
  stated without the clipping caveat, and whether the clock is "Adam
  normalisation" specifically (P2 alignment would confirm the mechanism).

## Result (filled after running, 2026-10-22)
Files: `results/E13/dyn_*.csv` (9 runs), `results/E13/per_run.csv`,
`results/E13/summary.txt`; script `scripts/analyze_E13.py`; logs
`logs/E12_E13_gpu1.log` (n = 10000 `sgd_mu` seed 0), `logs/E13_resume_gpu*.log`.

| n | arm (lr_ρ) | t₀.₃ per seed | σ_fc1 at 20k | peak entropy Δ_K | diverged at |
|---|---|---|---|---|---|
| 500 | sgd_mu (0.05) | 5000, 5000, 5000 | 0.945 (seed 0; seeds 1–2 stopped earlier) | +0.057, +0.060, +0.058 | 34k, 17k, 18k |
| 10000 | sgd_mu (0.05) | never ×3 | 0.0177 ×3 | −0.015, +0.005, −0.005 | none |
| 10000 | sgd_scaled (1.0) | 5000, never (stopped at 5k), 5000 | — | +0.022, −0.002, +0.006 | 6k, 5k, 9k |

- **P1 HOLDS:** σ at 20k is 0.945 (n = 500, the one seed still running at 20k)
  vs 0.0177 (n = 10000, 3/3), a 53× ratio. In every n = 500 seed σ reached 0.3
  by 5k steps, and in no n = 10000 seed did it ever. The N-dependence under SGD
  is not caused by joint clipping.
- **P2 NOT CONFIRMED as stated:** in 2 of 3 scaled runs t₀.₃ = 5000, exactly
  the n = 500 value (ratio 1.00). The third diverged at 5000 before reaching
  σ = 0.3, so the pre-registered mean is undefined.
- **P3 FAILED:** all scaled runs diverged at 5k–9k steps, before an advantage
  developed (peak entropy Δ_K +0.022 at most; MI up to +0.063).
- Not predicted: SGD on ρ without clipping is **unstable**. It diverged in 6
  of 9 runs, namely every run in which σ grew large, and was stable only where σ
  stayed tiny. Without clipping, n = 500 reaches a +0.057 to +0.060 entropy
  advantage before diverging, versus about +0.03 with joint clipping in E10, so
  E10's "σ grew but the advantage was small" was at least partly clipping
  slowing σ.
- CONFIRMATORY for P1–P3 as stated (with the deviation noted above).

## Deviation during running (recorded before any E13 result was analysed)
2026-10-22: the first run (n = 500, seed 0, `sgd_mu`, ρ unclipped) produced
non-finite predictions at step ~34,000 and crashed, losing that run's output.
`bnnstudy/dynamics.py` was changed to stop a run at the first non-finite
checkpoint, keep all earlier checkpoints and record `diverged_step`. All
predictions are scored at or before 20k steps (P1) or at t₀.₃ (P2), so they
are unaffected unless a run diverges earlier, which will be reported. The
n = 500 runs were rerun with the fix; one n = 10000 run that had already
finished (`sgd_mu`, seed 0) is kept.
