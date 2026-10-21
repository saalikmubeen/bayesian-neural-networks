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

## Result (filled after running, 2026-10-21)
Files: `results/E9/dyn_*.csv` (12 runs), `results/E9/selection.csv`; logs `logs/E9_gpu*.log`.
Δ̄ = (Δ_Fashion + Δ_KMNIST) / 2 at the selected checkpoint; mean over 12 runs.

| rule | mean step | Δ̄ | fraction of oracle | acc | ECE |
|---|---|---|---|---|---|
| Oracle (uses real OOD) | 10917 | +0.067 | 1.00 | 0.922 | 0.043 |
| **R_proxy (mean of 3 proxies)** | 10417 | **+0.049** | **0.73** | 0.924 | 0.043 |
| R_flip | 13750 | +0.045 | 0.67 | 0.923 | 0.048 |
| R_permute | 20583 | +0.027 | 0.42 | 0.909 | 0.105 |
| R_noise | 16167 | +0.020 | 0.32 | 0.912 | 0.086 |
| R_elbo | 39417 | +0.034 | 0.52 | 0.906 | 0.117 |
| R_nll | 2250 | −0.032 | −0.49 | 0.922 | 0.027 |

R_proxy by setting: n 500 → 0.69, n 2000 (rho −5) → 0.77, n 2000 (rho −3) → 0.75,
n 10000 → 0.72 of oracle.
Peak step (oracle, rho −5): n 500 → 9.7k, n 2000 → 11.0k, n 10000 → 18.7k.

- **P1 CONFIRMED:** the combined proxy rule reaches 73% of the oracle advantage
  (0.69–0.77 in every setting), at the same accuracy and ECE as the oracle
  checkpoint. NLL-based stopping lands before the window (negative Δ).
- **P2 FALSIFIED (narrowly):** the best single proxy (flip) reaches 67%, below
  70%. Combining proxies matters.
- **P3 confirmed as stated (mean peak step rises with n), but not robust.**
  The n = 10000 mean (18.7k) comes from one seed peaking at 35k; the other two
  peak at 10–11k. Median peak steps: 10k / 11k / 11k for n = 500 / 2000 /
  10000. Log-log slope of peak step vs n: 0.13 (12 runs, E6 + E9). Treat "the
  peak moves with n" as unsupported.
- Follow-up (EXPLORATORY, `scripts/analyze_window_scaling.py`,
  `results/E9/window_scaling.csv`, `figures/E9_window_vs_n.pdf`; window =
  contiguous span where the smoothed Δ̄ ≥ half its peak). The window *opens* at
  a similar step for every n (5k–8k). What changes is when it *closes*: by
  14k–19k at n = 2000 (6/6 runs), but still open at 40k at n = 10000 (3/3 runs),
  consistent with a weaker KL pull (1/N) slowing the drift towards the prior.
  n = 500 does not fit: 2 of 3 windows also stay open to 40k. Unexplained.
- CONFIRMATORY for P1–P2; P3 as above.
