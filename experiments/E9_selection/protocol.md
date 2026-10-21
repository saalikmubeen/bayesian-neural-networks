# Protocol: E9 — can the OOD window be found without OOD data? (+ how n moves it)

Written: 2026-10-21, before running. Status: planned

Background (E6): BBB's advantage over a softness-matched DNN exists only in a
transient window of training. Minimum held-out NLL stops too early and the
ELBO keeps improving past the window.

- **Hypothesis (with mechanism):** the window is where posterior samples
  disagree on inputs unlike the training data while still agreeing on
  in-distribution inputs. Synthetic outliers built from validation digits
  (noise, flipped digits, pixel-permuted digits) are also unlike the training
  data. So the checkpoint that best separates validation digits from these
  proxies (by mutual information) should lie inside the window for real OOD
  sets too.
- **Selection rules** (one checkpoint per run, every 1000 steps):
  R_nll = min validation NLL; R_elbo = min running −ELBO; R_noise, R_flip,
  R_permute = max validation-vs-proxy MI AUROC; R_proxy = max of the mean of
  the three proxy MI AUROCs; Oracle = max of (Δ_F + Δ_K)/2 (uses real OOD,
  reference only). Δ = entropy-AUROC difference vs the softness-matched DNN.
- **Predictions** (mean over the 12 runs of Δ̄ at the selected checkpoint,
  as a fraction of the oracle Δ̄):
  P1: R_proxy reaches ≥ 70% of oracle; R_nll ≤ 20%.
  P2: at least one single-proxy rule reaches ≥ 70%.
  P3 (what moves the window, rho −5): the step of peak Δ̄ increases with n
  (500 < 2000 < 10000, mean over seeds), because the KL weight 1/N is smaller
  at larger n, so σ grows more slowly per step.
- **Falsified if:** every proxy rule reaches < 50% of oracle (P1/P2), or peak
  steps do not increase with n (P3).
- **Config:** `bnnstudy.dynamics --proxies`, constant lr 1e-3, 40k steps, eval
  every 1000 steps (20 MC samples). Runs: n ∈ {500, 2000, 10000} with rho −5,
  seeds 0–2; plus n = 2000 with rho −3, seeds 0–2 (12 runs). Proxies come from
  the 5000 validation images only. DNN base for Δ: `results/E45/dnn_n<n>_s<seed>.npz`.
  Output `results/E9/`.
- **Metrics:** Δ_F, Δ_K, accuracy and ECE at each rule's checkpoint; fraction
  of oracle Δ̄; peak step per n.
- **Cost:** 12 runs × ~3 min, ~20 min on 2 RTX 3090s. Go-ahead given by the
  user 2026-10-21.
- **Decision it informs:** if P1 holds, the paper has a constructive half:
  "select BBB checkpoints with synthetic outliers". If not, the paper's
  message is that BBB's OOD advantage cannot be reliably obtained in practice.

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
