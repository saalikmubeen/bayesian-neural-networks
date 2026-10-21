# Protocol: E7b — is the missing CIFAR window about architecture or data?

Written: 2026-10-22, before running. Status: planned

Background: the MNIST MLP shows a transient window in which BBB beats a
softness-matched DNN (E4, E6, E9). A CNN on CIFAR-10 shows none (E7).

- **Competing hypotheses:**
  H_arch: the window needs noise in fully connected early layers acting
  together (E6b). The CNN's first conv layer has few, heavily constrained
  weights (σ stays ≈ 0.04). An MLP on CIFAR-10 therefore *will* show a window.
  H_data: the window depends on MNIST-family images (sparse strokes on a
  constant background), so an MLP on CIFAR-10 *will not* show one.
- **Decision rule (pre-registered):** with Δ = entropy-AUROC difference vs the
  DNN softened to BBB's validation entropy at that checkpoint, H_arch is
  supported if peak Δ_SVHN or peak Δ_CIFAR100 ≥ +0.03 in ≥ 2 of 3 seeds at
  either n, *and* Δ is ≤ +0.01 at the first checkpoint (so the gain appears
  during training). H_data is supported if peak Δ < +0.01 on both OOD sets in
  all seeds at both n. Anything in between is reported as inconclusive.
- **Config:** `bnnstudy/cifar.py --arch mlp`: MLP 3072-256-256-10 (the MNIST
  study's architecture on flattened images), no augmentation. DNN: Adam 1e-3,
  cosine, 30 epochs at both n (as in E7), batch 128. BBB: mean-field on all
  layers, N(0,1) prior, KL/N, 2000-step warm-up, Adam 1e-3 constant, clip 1.0,
  rho_init −5, 60k steps, eval every 1000 steps (20 MC samples). n ∈ {2000,
  10000}, seeds 0–2. OOD: SVHN (10k subset), CIFAR-100 test. Output
  `results/E7b/`.
- **Metrics:** per checkpoint Δ_SVHN, Δ_C100 (entropy and MI), accuracy, ECE,
  σ per layer, −ELBO, proxy AUROCs.
- **Cost:** 6 BBB runs × ~6 min plus 6 DNN runs, ~25 min on 2 RTX 3090s.
  Go-ahead given by the user 2026-10-22.
- **Decision it informs:** the scope sentence of the paper. H_arch → "the
  window needs fully connected early layers with posterior noise". H_data →
  "the window is specific to MNIST-like data".

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
