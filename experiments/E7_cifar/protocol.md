# Protocol: E7 — does the transient OOD window replicate on CIFAR-10 with a CNN?

Written: 2026-10-21, before running. Status: planned

- **Hypothesis:** the MNIST mechanism (E4, E6, E9) is a property of
  mean-field VI training dynamics, not of MLPs on MNIST. On CIFAR-10 with a
  small CNN, BBB starts no better than a softness-matched DNN, opens a window
  where it is better, and loses it as σ approaches the prior. Synthetic-proxy
  selection finds the window better than NLL-based stopping.
- **Predictions** (n = 10000 CIFAR-10 images, seeds 0–2; OOD = SVHN (10k
  subset), CIFAR-100 test; Δ = entropy-AUROC difference vs the DNN softened to
  BBB's validation entropy at that checkpoint):
  P1: peak Δ_SVHN ≥ +0.03 in ≥ 2 of 3 seeds.
  P2: the window closes: Δ at the last checkpoint (60k steps) is < peak Δ − 0.02
  on SVHN in ≥ 2 of 3 seeds.
  P3: R_proxy (max mean proxy-MI AUROC) gets ≥ 50% of the oracle's mean Δ over
  both OOD sets, and R_nll (min validation NLL) gets less than R_proxy.
  P4 (reference): the 5-member ensemble beats its softness-matched DNN on SVHN.
- **Falsified if:** peak Δ_SVHN < +0.01 in all seeds (no window on CIFAR), or
  Δ never declines (no transient).
- **Config:** `bnnstudy/cifar.py`. CNN conv32-conv64-conv128 (3×3, max-pool),
  fc256, fc10; no augmentation. DNN: Adam 1e-3, cosine, 30 epochs, batch 128.
  Ensemble: 5 such DNNs. BBB: mean-field Gaussian on all layers, N(0,1) prior,
  KL/N with 2000-step warm-up, Adam 1e-3 constant, grad clip 1.0, rho_init −5,
  60k steps, eval every 2000 steps with 20 MC samples. Proxies (noise,
  upside-down, pixel-permuted) come from the 5000 validation images only.
  Output `results/E7/`.
- **Metrics:** per checkpoint: acc, ECE, entropy and MI AUROC, Δ vs matched
  DNN, proxy AUROCs, σ per layer, −ELBO. Selection rules as in E9.
- **Cost:** ~25 min per seed (BBB ~20 min, DNN + ensemble ~6 min); 3 seeds on
  2 RTX 3090s, ~1 h. Go-ahead given by the user 2026-10-21.
- **Decision it informs:** if P1–P2 hold, the paper's main claim generalises
  beyond MNIST MLPs, and E8 (theory/scaling) and E10 (HMC) are next. If not,
  the claim is scoped to small MLPs and the paper says so.

## Result (filled after running, 2026-10-22)
Files: `results/E7/dyn_n10000_rho-5_s{0,1,2}.csv`, `results/E7/summary.txt`,
`{dnn,ensemble}_n10000_s*.npz` (not tracked); logs `logs/E7_gpu*.log`.
DNN test accuracy 0.646–0.669, ensemble 0.687–0.698, BBB 0.61–0.68 along training.

| seed | peak Δ_SVHN (step) | last Δ_SVHN | peak Δ_C100 | BBB acc first / max / last | σ f1 / c1 at end |
|---|---|---|---|---|---|
| 0 | −0.044 (2k) | −0.112 | +0.031 | 0.631 / 0.672 / 0.614 | 0.996 / 0.039 |
| 1 | +0.006 (8k) | −0.010 | +0.027 | 0.653 / 0.675 / 0.613 | 0.996 / 0.038 |
| 2 | −0.056 (4k) | −0.128 | +0.024 | 0.664 / 0.683 / 0.626 | 0.996 / 0.039 |

Selection (mean over seeds, Δ̄ = mean of SVHN and CIFAR-100 Δ): Oracle −0.005
(step 4.7k); R_proxy −0.053 (53k); R_nll −0.025 (6k); R_elbo −0.040 (53k).

Ensemble vs DNN softened to the ensemble's validation entropy (T 1.73–1.83):
SVHN 0.740 / 0.731 / 0.727 vs 0.787 / 0.759 / 0.807 (ensemble worse in 3/3);
CIFAR-100 0.690 / 0.689 / 0.692 vs 0.661 / 0.662 / 0.669 (ensemble better in 3/3).

- **P1 FALSIFIED, and the protocol's falsification condition is met:** peak
  Δ_SVHN is below +0.01 in all seeds (−0.044, +0.006, −0.056). There is no
  OOD window on CIFAR-10 with this CNN. On CIFAR-100, BBB is at most +0.02 to
  +0.03 better, early in training.
- **P2 moot:** the advantage never appears. Δ_SVHN does fall further late in
  training in 2/3 seeds.
- **P3 FALSIFIED:** with no window, proxy selection picks late, prior-dominated
  checkpoints (Δ̄ −0.053), worse than NLL stopping (−0.025).
- **P4 FALSIFIED on SVHN:** the 5-member ensemble also loses to a single
  softened DNN on SVHN, and wins on CIFAR-100 (+0.02 to +0.03).
- The drift towards the prior does replicate: f1 σ → 0.996 by ~14k steps, the
  −ELBO keeps falling (206 → 2.6), and accuracy declines (max 0.67–0.68 →
  0.61–0.63). Conv-1 σ stays small (0.04), consistent with E6b (the advantage
  needs noise in the early layers together).
- Scope: one CNN, n = 10000, no augmentation, σ init −5, evaluation every 2k
  steps. A window shorter than 2k steps, or one that needs a different init,
  is not ruled out.
- CONFIRMATORY for P1–P4.
