# Protocol: E20 — does the exact posterior behave like BBB? (HMC that mixes, prior N(0, 0.3²))

Written: 2026-10-22, before the confirmatory runs. Status: planned. (SHA-256
recorded in `experiments/E20_prereg.txt` before launch, self-attested.)

Background: E19 (full-batch HMC, N(0,1) prior, width 100) failed its
convergence check (split-R̂ 90th percentile 2.5–2.8). Diagnosis after E19:
under N(0,1) the posterior's weights sit near prior scale (std ≈ 0.85 per
layer), the network's logits are ~700, and the likelihood behaves like a hard
wall around "all training points correct". The stable step size is ~3e-5 at
300 leapfrog steps and shrinks to ~1.5e-6 when trajectories are lengthened to
10⁴ steps, so a mixing run would need ~10⁶ gradients per iteration.

**Pilots (n = 500 only; only sampler diagnostics were looked at — step size,
acceptance, potential energy; no test, OOD or accuracy numbers were computed):**
- σ_p = 1, diagonal mass matrix with Stan-style windows (`bnnstudy.hmc --mass
  diag`, `logs/E20_pilot_c*.log`): whitened step size stayed ~1.6e-4 and fell
  to 8e-6 after the second window; abandoned.
- σ_p = 1, identity mass, 8 batched chains, 10⁴ leapfrog steps
  (`bnnstudy.hmc_batch`, `logs/E20_pilot2.log`): step size 1.5e-6.
- σ_p = 1, width 20 (`logs/E20_probe_w20.log`): step size 1e-5 at 3000 steps.
- σ_p = 0.3, width 100 (`logs/E20_probe_sp03.log`): step size 4–7e-4,
  acceptance ~0.78, potential flat after ~10 iterations. Tractable.

The prior is therefore changed to N(0, 0.3²). This is not chosen for any
outcome: E11 already showed that BBB keeps (and stabilises) its OOD advantage
at σ_p = 0.3 (entropy Δ_K +0.045, MI Δ_K +0.071 at 40k steps, width 256), so
this is the regime where BBB's effect is cleanest.

- **Design:** MLP 784-100-100-10 (89,610 parameters), N(0, 0.3²) prior on all
  weights and biases, MNIST subsets n ∈ {500, 2000}, data seed 0.
- **HMC settings (fixed now):** `bnnstudy.hmc_batch`; 8 chains per n, run as a
  batch in one process; each chain initialised at its own MAP (Adam, 3000
  full-batch steps, init seed 1000 + chain); identity mass matrix; per-chain
  dual-averaging step size towards 0.8 acceptance during 500 warm-up
  iterations, then frozen (±10% jitter); trajectory length π·σ_p/2 = 0.471
  (Izmailov et al., 2021), L = ceil(0.471 / median step size), at most 6000
  leapfrog steps; 1000 sampling iterations, every 10th kept (100 samples per
  chain, 800 in total). Predictions: Bayesian model average over all kept
  samples.
- **Comparators (width 100, same subset):** BBB under the same prior,
  `bbb_steps23550_sp0.3` (23,550 Adam steps, otherwise as `bbb_steps23550`),
  seed 0 (shares the HMC subset) for the predictions; seeds 1–2 reported as
  context only. Softness-matched DNN base: the 40-epoch width-100 DNN from
  E19 (`results/E19/dnn_n<n>_s0.npz`; it has no prior).
- **Convergence (decided now):** split-R̂ over the 8 chains, on the predicted
  probability of the true class for 1000 fixed test images (as in E19).
  Confirmatory only if the 90th percentile is < 1.1 at that n; otherwise
  exploratory. Also reported: median R̂, and the change in mean potential
  energy between the first and second half of sampling.
- **Precondition (not a prediction):** BBB at σ_p = 0.3 (seed 0) has entropy
  Δ_K ≥ +0.03 at n = 2000. If not, the comparison at that n cannot speak to
  "BBB's advantage", and is reported as such.
- **Predictions** (Δ = entropy AUROC minus that of the DNN softened to the
  method's validation entropy; OOD = Fashion-MNIST, KMNIST; applied separately
  at each n):
  P1 (decision rule, not a direction): if HMC's Δ_K and Δ_F are both
  ≥ +0.03, report "the exact posterior also has an OOD advantage beyond
  softness"; if both ≤ +0.01, report "BBB's advantage is not a property of the
  exact posterior"; otherwise inconclusive.
  P2: HMC is less overpruned than BBB: the fraction of hidden-layer weights
  whose within-chain |mean| > within-chain std (averaged over chains) is ≥ 2×
  BBB's fraction of |μ| > σ.
  P3: HMC's test accuracy is ≥ BBB's (seed 0).
  P4: HMC's MI-based Δ_K is within 0.03 of BBB's MI-based Δ_K (BBB's
  "knows what it doesn't know" signal matches the exact posterior's).
- **Falsified if:** P2 — HMC's fraction is < BBB's (the exact posterior is at
  least as "pruned"; overpruning is then not a mean-field artefact).
- **Config:** `bnnstudy/hmc_batch.py`, `scripts/run_E20.sh`,
  `scripts/analyze_E20.py`; `bnnstudy.train --hidden 100 --method
  bbb_steps23550_sp0.3`. Output `results/E20/`.
- **Cost:** about 0.5 h (n = 500) and 1.5–2.5 h (n = 2000) on two RTX 3090s
  in parallel, plus ~10 min of BBB training. Go-ahead: user ("do 2"),
  2026-10-22.
- **Decision it informs:** whether the paper can say that BBB's OOD advantage
  is (or is not) shared by the exact Bayesian posterior under the same prior.

## Result (filled after running, 2026-10-22)
Files: `results/E20/hmcb_n{500,2000}.pt` (8 chains × 100 samples each),
`results/E20/bbb_steps23550_sp0.3_n{500,2000}_s{0,1,2}.*`,
`results/E20/summary.txt`, `results/E20/compare.csv`; logs
`logs/E20_n500.log`, `logs/E20_n2000.log`. Run time: 25 min (n = 500) and
55 min (n = 2000) for HMC, ~1.5 min per BBB model.

- **Convergence: MET at both n.** Split-R̂ over 8 chains (p(true class), 1000
  test images): n = 500 median 1.000, 90th pct 1.005, max 1.015; n = 2000
  median 1.000, 90th pct 1.005, max 1.019. Frozen step sizes 1.1–1.4e-3
  (n = 500, 376 leapfrog steps) and 5.4–6.9e-4 (n = 2000, 760 steps);
  acceptance 0.77–0.84. Mean potential changed by −1.9 and +11.6 between the
  two halves of sampling (batch-means s.e. 5.5 and 5.3, i.e. −0.3 and +2.2
  s.e.; 6 of 8 chains drifted upwards at n = 2000, a hint of mild
  non-stationarity there). The R̂ used is classic split-R̂, not rank-normalised.

| n | method | test acc | NLL | ECE | Δ_F | Δ_K | MI Δ_F | MI Δ_K | FPR95 K (DNN_T) |
|---|---|---|---|---|---|---|---|---|---|
| 500 | HMC | 0.890 | 0.470 | 0.158 | +0.079 | +0.070 | +0.087 | +0.075 | 0.19 (0.49) |
| 500 | BBB σ_p 0.3, seed 0 | 0.798 | 0.818 | 0.253 | −0.002 | −0.021 | +0.100 | +0.081 | 0.70 (0.48) |
| 2000 | HMC | 0.929 | 0.273 | 0.072 | +0.115 | +0.074 | +0.119 | +0.077 | 0.14 (0.48) |
| 2000 | BBB σ_p 0.3, seed 0 | 0.909 | 0.392 | 0.131 | +0.077 | +0.033 | +0.080 | +0.072 | 0.34 (0.46) |

  DNN (width 100, seed 0): 0.873 (n = 500), 0.910 (n = 2000). BBB seeds 1–2
  (context, other subsets): n = 500 acc 0.808 / 0.747, Δ_K −0.020 / −0.054,
  MI Δ_K +0.058 / +0.055; n = 2000 acc 0.911 / 0.899, Δ_K +0.027 / +0.029,
  MI Δ_K +0.074 / +0.069.
- **Precondition:** met at n = 2000 (BBB entropy Δ_K +0.033); NOT met at
  n = 500 (−0.021), where BBB's advantage exists only in MI. At n = 500 the
  comparison therefore speaks to BBB's MI signal only.
- **P1 (decision rule): "the exact posterior also has an OOD advantage beyond
  softness"** at both n (Δ_F +0.079 / +0.115, Δ_K +0.070 / +0.074).
- **P2 FALSIFIED** at both n: within-chain |mean| > std for 0.0000 of hidden
  weights, vs 0.0043 / 0.0096 for BBB. See the follow-up below: the exact
  posterior's per-weight marginals are the prior, so this measure cannot
  distinguish BBB from it.
- **P3 CONFIRMED** at both n (0.890 vs 0.798; 0.929 vs 0.909).
- **P4 CONFIRMED** at both n: |MI Δ_K(HMC) − MI Δ_K(BBB)| = 0.006 / 0.006 for
  the BBB seed on the HMC subset. BBB seeds on other subsets differ from HMC
  by up to 0.02 (n = 500: MI Δ_K +0.058 / +0.055), still within 0.03.
- CONFIRMATORY (convergence criterion met at both n).

### Follow-up (exploratory, inference only; `scripts/analyze_E20_extra.py`, `results/E20/extra.txt`)
- Per-weight marginals of the exact posterior equal the prior: median
  within-chain std 0.299–0.300 in every layer (prior 0.3), median |mean|
  0.020–0.022 (≈ the sampling error of a mean of 100 draws). BBB at the same
  prior is close to the prior in both hidden layers (σ 0.295–0.299, |μ|
  0.011–0.025) and narrower in the last layer (σ 0.26 / 0.15).
- Drawing each weight independently from its HMC marginal (same means and
  stds, correlations removed) gives chance accuracy (0.087 / 0.089): all of
  the exact posterior's information is in correlations between weights.
- Every single chain shows the advantage (n = 500: Δ_K +0.064 to +0.069;
  n = 2000: +0.072 to +0.073), so it is not diversity between chains. One
  posterior draw alone is poor (accuracy 0.72 / 0.83, Δ_K −0.23 / −0.20).
- R₁ (E17/E18) from HMC's per-weight moments separates OOD weakly (AUROC
  0.69–0.88), against 0.85–0.97 from BBB's μ and σ: R₁ describes BBB's
  mean-field solution, not the exact posterior.
- What it suggests: under this prior, an OOD advantage beyond softness is a
  property of Bayesian model averaging itself, and BBB's MI recovers its size
  (within 0.006) while BBB's entropy recovers only part of it (or none at
  n = 500). BBB gets there by a different route: mean-field cannot hold the
  correlations that carry the exact posterior's information, and BBB instead
  fits a few means and leaves the rest at prior-scale noise.

### Independent code review and added checks (exploratory, `results/E20/extra.txt`)
A separate reviewer audited the sampler and analysis and found no error that
affects the conclusions (kinetic energy, acceptance, accept/reject masking,
state-independent step size and L, warm-up exclusion, R̂ axes, validation-only
temperature fit, same data subset for HMC, BBB and DNN all verified; Δ
recomputed independently). Checks added as a result:
- Sampler self-test: E[θ·∇U]/dim = 1.0001 (n = 500) and 0.9993 (n = 2000),
  every chain within 0.9982–1.0022 (must be 1 at stationarity).
- Accuracy confound: on test images that both HMC and the DNN classify
  correctly (86% / 90% of test), Δ_F +0.063 / +0.099 and Δ_K +0.054 / +0.060
  (all images: +0.079 / +0.115, +0.070 / +0.074). Part of the advantage comes
  from HMC's higher accuracy; most of it remains.
- Leapfrog steps actually used while sampling (jittered step size): 360–430
  (n = 500), 702–856 (n = 2000).
