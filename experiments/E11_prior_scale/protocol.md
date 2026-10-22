# Protocol: E11 — is the OOD advantage an artefact of the misspecified N(0,1) prior?

Written: 2026-10-22, before running. Status: planned. (Not committed before
running; SHA-256 and launch time recorded in `logs/E10_E11_prereg.txt`.)

Background: all BBB runs so far use an N(0,1) prior, while trained weights
have scale ~0.05. Along training the posterior drifts towards that prior
(σ → 0.98), the model underfits (accuracy 0.93 → 0.89, test entropy 0.30 →
0.94) and the entropy-based advantage at n = 2000 disappears (E6, E9).

- **Hypotheses:**
  H1: with a better-scaled prior (σ_p = 0.1 or 0.3) the drift towards the
  prior costs less fit, so accuracy holds up and the entropy-based advantage
  does not collapse.
  H2 (kill test from the expert review): if the MI-based advantage disappears
  under a better-scaled prior, the advantage is an artefact of the
  misspecified prior.
- **Predictions** (n = 2000, rho_init −5, Adam lr 1e-3 constant, 40k steps,
  3 seeds; reference σ_p = 1 from E9, same seeds):
  P1: final accuracy (40k) is higher for σ_p = 0.1 and 0.3 than for σ_p = 1
  (mean over seeds, difference ≥ 0.01).
  P2: entropy-based Δ_KMNIST at 40k is ≥ 0.02 higher than for σ_p = 1
  (σ_p = 1: −0.009) for at least one of σ_p ∈ {0.1, 0.3}.
  P3: peak MI-based Δ_KMNIST over training is ≥ +0.02 for both σ_p = 0.1 and
  0.3 (the advantage is not only a prior artefact).
- **Falsified if:** P3 fails for σ_p = 0.1 (MI-Δ < 0.02 at every checkpoint):
  then the advantage depends on the misspecified prior.
- **Config:** `bnnstudy.dynamics --prior-sigma {0.1, 0.3}` (tags `sp0.1`,
  `sp0.3`), MNIST MLP, n = 2000, seeds 0–2, 40k steps, eval every 1000 steps,
  20 MC samples, `--proxies`. σ init stays 0.007 (below both priors). DNN
  base: `results/E45/dnn_n2000_s<seed>.npz`. Output `results/E11/`.
- **Cost:** 6 runs × ~3 min, ~10 min on one RTX 3090. Go-ahead: user, 2026-10-22.
- **Decision it informs:** whether the paper's MI and steps results survive
  a sensible prior, which is the most predictable reviewer objection.

## Result (filled after running, 2026-10-22)
Files: `results/E11/dyn_*.csv` (6 runs), `results/E11/per_run.csv`,
`results/E11/summary.txt`; reference σ_p = 1 from E9 (n = 2000, rho −5,
same seeds). Seed 2 was run on the second GPU (`logs/E11_seed2_gpu1.log`).

| σ_p | acc at 40k | entropy Δ_K / Δ_F at 40k | MI Δ_K at 40k | peak MI Δ_K / Δ_F | σ_fc1 at 40k | test entropy at 40k |
|---|---|---|---|---|---|---|
| 1.0 | 0.896 | −0.009 / +0.015 | +0.065 | +0.080 / +0.093 | 0.98 | 0.94 |
| 0.3 | 0.897 | **+0.045 / +0.048** | +0.071 | +0.075 / +0.076 | 0.30 | 0.79 |
| 0.1 | 0.870 | −0.007 / −0.019 | +0.045 | +0.055 / +0.038 | 0.10 | 0.92 |

- **P1 FAILED for both:** a tighter prior does not improve accuracy. σ_p = 0.1
  is *worse* (0.870 vs 0.896) and underfits from early in training (0.86 at
  5k), because the same prior also pulls the weight means hard towards zero.
  σ_p = 0.3 matches σ_p = 1 (0.897).
- **P2 holds for σ_p = 0.3, fails for 0.1:** with σ_p = 0.3 the entropy-based
  advantage does not collapse (+0.045 at 40k vs −0.009 for σ_p = 1).
- **P3 CONFIRMED (kill test passed):** peak MI Δ_K is +0.055 (σ_p = 0.1) and
  +0.075 (σ_p = 0.3), both ≥ 0.02. The advantage is not only an artefact of
  the N(0,1) prior. With σ_p = 0.1, though, the Fashion-MNIST advantage mostly
  disappears (entropy Δ_F ≤ 0 throughout).
- Not predicted: with σ_p = 0.3, σ_fc1 reaches the prior scale by ~10k steps
  and plateaus, and both entropy- and MI-based advantages then stay stable
  (+0.045 to +0.072) to the end. The "window closing" seen with N(0,1) is
  largely σ drifting all the way to an oversized prior.
- CONFIRMATORY for P1–P3 as stated.
