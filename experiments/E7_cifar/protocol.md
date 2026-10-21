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

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
