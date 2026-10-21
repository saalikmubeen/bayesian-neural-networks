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

## Result (filled after running, 2026-10-22)
Files: `results/E7b/dyn_*.csv`, `results/E7b/summary.txt`; logs `logs/E7b_gpu*.log`.
DNN (MLP) test accuracy: see the `test acc` lines in the logs. BBB MLP accuracy
peaks at 0.47 (n = 10000) and 0.37–0.38 (n = 2000), then collapses towards
the prior (0.24–0.29 and 0.10–0.18 at 60k; σ_f1 → 0.997–0.999).

| n | seed | first Δ_S / Δ_C | peak Δ_SVHN (step, acc) | peak Δ_C100 (step, acc) |
|---|---|---|---|---|
| 10000 | 0 | −0.031 / +0.004 | +0.005 (4k, 0.47) | +0.032 (3k, 0.47) |
| 10000 | 1 | −0.049 / −0.003 | −0.014 (3k, 0.47) | +0.037 (4k, 0.47) |
| 10000 | 2 | −0.031 / −0.002 | +0.019 (3k, 0.46) | +0.030 (6k, 0.45) |
| 2000 | 0 | −0.042 / −0.004 | −0.001 (3k, 0.38) | +0.091 (58k, 0.17) |
| 2000 | 1 | −0.081 / +0.001 | −0.031 (3k, 0.38) | +0.103 (54k, 0.17) |
| 2000 | 2 | −0.012 / +0.015 | +0.030 (4k, 0.37) | +0.205 (39k, 0.10) |

- **Pre-registered rule: H_arch supported** (criterion met in 3/3 seeds at
  n = 10000 and 2/3 at n = 2000; H_data's criterion in 0/6).
- **But the evidence is weak.** The qualifying gains are on CIFAR-100 only and
  barely above the +0.03 threshold while the model still learns (+0.030 to
  +0.046 at checkpoints with ≥ 90% of max accuracy, steps 1k–7k). That is
  similar to the CNN's CIFAR-100 gain in E7 (+0.024 to +0.031). The largest
  n = 2000 gains (+0.09 to +0.21) come after collapse to near-chance accuracy
  and are not meaningful. SVHN shows no window for either architecture. The
  rule did not include an accuracy condition; that was an omission.
- Reading: on CIFAR-10 neither architecture shows the MNIST-sized window
  (+0.06 to +0.08 on both OOD sets). The MLP shows a brief, small CIFAR-100
  gain before collapsing. This does not cleanly separate architecture from
  data; it points more to the data (or to task difficulty, at 37–47%
  accuracy) than to the CNN's structure.
- CONFIRMATORY for the rule as written; the substantive reading is EXPLORATORY.
