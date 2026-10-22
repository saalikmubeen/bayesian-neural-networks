# Protocol: E17 — do OOD inputs drive BBB's noisy weights?

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E17_prereg.txt` before running, self-attested.)

Background (E16): long-trained BBB is effectively overpruned. Only 1–5% of
hidden weights have |μ| > σ, |μ| and σ are anti-correlated in fc2, and the
advantage needs BBB's own means *and* its per-weight allocation of noise.

- **Hypothesis (mechanism):** in-distribution digits are computed mainly
  through the low-noise subnetwork, so the posterior noise reaching the
  hidden units is small relative to their mean input. OOD images also drive
  high-σ weights, so noise is large relative to signal and posterior samples
  disagree. Noise compounds across layers (E6b), so the second layer should
  separate ID from OOD better than the first.
- **Quantities** (analytic, per image x; posterior-mean forward pass for the
  previous layer's activations a = ReLU of the mean pre-activation; a⁰ = x):
  for layer ℓ ∈ {1, 2}, mean pre-activation m_j = Σ_i μ_ij a_i + μ_bj and
  noise variance v_j = Σ_i σ_ij² a_i² + σ_bj². Noise-to-signal ratio
  R_ℓ(x) = Σ_j v_j / Σ_j m_j².
- **Predictions** (BBB = `bbb_steps23550` at n = 2000 and `bbb` at n = 10000
  from `results/E45`, seeds 0–4, test set vs Fashion-MNIST and KMNIST):
  P1: R₂ used as an OOD score has AUROC ≥ 0.90 on both OOD sets at both n
  (mean over seeds).
  P2: per-image Spearman correlation between R₂ and BBB's mutual information
  (50 samples), over test + OOD images pooled, is ≥ 0.6 (mean over seeds and
  OOD sets) at both n.
  P3: AUROC(R₂) > AUROC(R₁) on both OOD sets at both n (mean over seeds).
- **Falsified if:** AUROC(R₂) < 0.75 on either OOD set at either n; then the
  noise-to-signal account does not explain BBB's OOD detection.
- **Config:** `scripts/analyze_E17.py`; data and splits as E12/E16; MI from
  `bnnstudy.models.BBB.predict` with 50 samples. No training.
- **Cost:** inference only, a few minutes. Go-ahead: user, 2026-10-22.
- **Decision it informs:** whether the paper can give a mechanistic account
  (optimizer clock → overpruned posterior → noise-to-signal on OOD inputs →
  disagreement), or only describe the effect.

## Result (filled after running, 2026-10-22)
Files: `results/E17/per_seed.csv`, `results/E17/summary.txt`; script
`scripts/analyze_E17.py`. Exploratory follow-up:
`results/E17/exploratory_R1_vs_MI.csv` (inline script run in session; same
quantities). 5 seeds; means shown.

| n | OOD | AUROC R₁ | AUROC R₂ | AUROC BBB MI | Spearman(R₂, MI) | Spearman(R₁, MI) pooled / within OOD (exploratory) |
|---|---|---|---|---|---|---|
| 2000 | Fashion | 0.960 | 0.507 | 0.959 | −0.15 | 0.71 / 0.50 |
| 2000 | KMNIST | 0.981 | 0.590 | 0.977 | +0.02 | 0.76 / 0.46 |
| 10000 | Fashion | 0.982 | 0.455 | 0.984 | −0.16 | 0.81 / 0.73 |
| 10000 | KMNIST | 0.990 | 0.692 | 0.995 | +0.30 | 0.83 / 0.72 |

- **P1 FAILED, falsification condition MET:** AUROC(R₂) is 0.45–0.69 (< 0.75)
  everywhere.
- **P2 FAILED:** Spearman(R₂, MI) is −0.07 and +0.07 on average.
- **P3 FAILED:** R₁ separates far better than R₂.
- Not predicted: the *first-layer* noise-to-signal ratio R₁ alone matches
  BBB's own MI-based detection (AUROC 0.96–0.99 vs 0.96–0.995), and it tracks
  BBB's per-image MI (Spearman 0.71–0.83 pooled, 0.46–0.73 within OOD images;
  about 0 within the in-distribution test set, where MI is uniformly low).
- Design flaw in R₂: it feeds layer 2 the *mean* layer-1 activations and so
  ignores noise propagated from layer 1, exactly the compounding that E6b
  showed matters. R₂ as defined does not test the compounding hypothesis.
- Reading (exploratory): OOD images send much more of their input through
  high-σ first-layer weights relative to the signal they produce. That
  first-layer noise-to-signal ratio is where ID and OOD separate. E6b shows
  that second-layer noise is needed to carry it into disagreement at the
  output (first-layer noise alone keeps ≤ 16% of Δ).
- CONFIRMATORY for P1–P3 as stated; the R₁ results are EXPLORATORY and need
  their own pre-registered test.
