# Protocol: E4 — is BBB's OOD gain set by optimizer steps or by data size?

Written: 2026-10-21, before running. Status: planned

Background (interim E2 analysis, exploratory): BBB beats its
softness-matched DNN only at n = 10000, where fixed epochs give 23550 steps,
versus 4800 at n = 500 and 2000. The gain correlates with ink on
"dead" pixels (constant in every training image).

- **Hypothesis (with mechanism):** first-layer weights from dead pixels get no
  likelihood gradient. Their mean-field σ is pushed only by the KL term, from
  its init (softplus(−5) ≈ 0.007) towards the prior (σ = 1), and that takes
  many Adam steps at lr 1e-3. Once these σ are large, any OOD ink on dead
  pixels makes the posterior samples disagree. So the OOD gain depends on the
  number of optimizer steps, not on n.
- **Predictions** (paired over seeds 0–4):
  P1 (steps, not n): at n = 2000, `bbb_steps23550` beats its softness-matched
  DNN on KMNIST (entropy AUROC) by ≥ 0.03 in ≥ 4 of 5 seeds, while `bbb`
  (4800 steps) does not. At n = 10000, `bbb_steps4800` loses most of the gain
  of `bbb` (23550 steps): its Δ vs matched DNN is less than half of `bbb`'s on
  both OOD sets.
  P2 (where the σ grow): in saved `bbb` models with 23550 steps, the mean σ of
  first-layer weights from dead pixels is > 5× that from live pixels. With
  4800 steps the ratio is smaller.
  P3: n = 500 with 23550 steps behaves like n = 2000 with 23550 steps (gain
  present), despite 4× less data.
- **Falsified if:** at equal steps the gain still tracks n (n = 2000 with
  23550 steps shows no gain, or n = 10000 with 4800 steps keeps most of it),
  or the σ ratio for dead vs live pixels is < 2 in the 23550-step models.
- **Config:** `bnnstudy`, MNIST subsets as E2, seeds 0–4. Methods: `bbb`
  (default epochs), `bbb_steps23550` at n ∈ {500, 2000}, `bbb_steps4800` at
  n = 10000. KL warm-up keeps the 20/150 fraction of training. Models saved
  (`--save-model`), on-support OOD variants added (`--on-support`). The DNN
  base for the softness match is the E5 `dnn` run from the same directory.
  Output: `results/E45/`.
- **Metrics:** Δ AUROC (entropy and MI) vs `dnn_Tmatch[<method>]`, paired by
  seed; per-pixel posterior σ of fc1 (mean over hidden units) for dead vs live
  pixels.
- **Cost:** shared with E5, ~25 min on 2 RTX 3090s. Go-ahead given by the user
  2026-10-21.
- **Decision it informs:** if confirmed, the paper's central claim becomes
  "BBB's OOD advantage on MNIST-type data is a pixel-support detector switched
  on by optimisation length". If falsified, the n = 10000 gain needs another
  explanation before any claim is made.

## Result (filled after running, 2026-10-21)
Files: `results/E45/` (85 runs, models `.pt`), `metrics.csv`, `paired_dnn.csv`,
`sigma_by_pixel.csv`, `figures/E4_sigma_maps.pdf`; script `scripts/analyze_E4.py`.

Δ = AUROC_entropy(M) − AUROC_entropy(dnn_Tmatch[M]), 5 seeds [95% CI] (wins/5):

| n | steps | Δ Fashion | Δ KMNIST | fc1 σ dead / live (ratio) | acc | ECE |
|---|---|---|---|---|---|---|
| 500 | 4800 (`bbb`) | −0.037 [−0.052, −0.021] (0) | −0.033 [−0.041, −0.027] (0) | 0.103 / 0.103 (1.00) | 0.885 | 0.030 |
| 500 | 23550 | **+0.070 [0.044, 0.096] (5)** | **+0.064 [0.054, 0.075] (5)** | 0.970 / 0.886 (1.10) | 0.886 | 0.039 |
| 2000 | 4800 (`bbb`) | −0.028 [−0.070, 0.013] (2) | −0.009 [−0.012, −0.005] (0) | 0.107 / 0.105 (1.02) | 0.937 | 0.022 |
| 2000 | 23550 | **+0.076 [0.056, 0.095] (5)** | **+0.069 [0.061, 0.076] (5)** | 0.921 / 0.787 (1.17) | 0.933 | 0.055 |
| 10000 | 4800 | −0.006 [−0.012, 0.001] (2) | +0.008 [0.005, 0.011] (5) | 0.106 / 0.096 (1.11) | 0.970 | 0.011 |
| 10000 | 23550 (`bbb`) | **+0.057 [0.043, 0.071] (5)** | **+0.070 [0.062, 0.078] (5)** | 0.829 / 0.666 (1.24) | 0.964 | 0.044 |

- **P1 CONFIRMED:** the OOD gain follows optimizer steps, not n. At n = 2000,
  23550 steps gives +0.07 in 5/5 seeds and 4800 steps gives none. At n = 10000,
  cutting to 4800 steps removes nearly all of it.
- **P3 CONFIRMED:** n = 500 with 23550 steps shows the gain (+0.064 to +0.070).
- **P2 FALSIFIED:** the σ of dead-pixel weights is only 1.0–1.24× that of
  live-pixel weights. σ grows *globally*, from ~0.10 at 4800 steps to 0.67–0.97
  (prior = 1) at 23550 steps. The mechanism is not dead-pixel specific.
- Not predicted: the longer-trained, wider posteriors cost calibration (ECE
  0.022 → 0.055 at n = 2000; 0.011 → 0.044 at n = 10000) and a little accuracy
  at n ≥ 2000.
- CONFIRMATORY for P1–P3 as stated.
