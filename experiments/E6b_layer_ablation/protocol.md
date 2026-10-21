# Protocol: E6b — which layer's posterior noise drives the OOD advantage?

Written: 2026-10-21, before running. Status: planned

- **Hypothesis (with mechanism):** first-layer noise acts on the raw pixels.
  Its effect on the output depends on how the input projects onto weakly
  constrained directions, which differ between ID and OOD images. Most of the
  converged BBB's OOD signal therefore comes from fc1's σ.
- **Predictions** (converged models `results/E45/bbb_steps23550_n2000_s{0..4}.pt`,
  inference only; "keep L" = σ kept in layer L only, the other layers set to
  their posterior mean):
  P1: keep fc1 retains ≥ 70% of the full model's Δ_KMNIST and Δ_Fashion
  (entropy AUROC vs dnn_Tmatch, temperature refitted to each variant's
  validation entropy).
  P2: keep fc3 retains < 30%.
- **Falsified if:** keep fc1 retains < 50% on both OOD sets.
- **Config:** `scripts/analyze_E6b.py`; variants: all (full model), keep
  fc1, keep fc2, keep fc3, none (posterior mean only); 50 MC samples; DNN base
  `results/E45/dnn_n2000_s<seed>.npz`; seeds 0–4.
- **Metrics:** entropy-AUROC and MI-AUROC (Fashion, KMNIST), Δ vs matched DNN,
  test acc, ECE.
- **Cost:** inference only, ~2 min. Go-ahead given by the user 2026-10-21.
- **Decision it informs:** locates the mechanism for the paper's analysis
  section, and suggests whether a cheaper "Bayesian first layer only" model
  would keep the gain.

## Result (filled after running, 2026-10-21)
Files: `results/E6b/layers.csv`, `summary.txt`; script `scripts/analyze_E6b.py`.
5 seeds, mean Δ (entropy AUROC vs softness-matched DNN); "retained" = Δ / Δ(all).

| variant (noise kept in) | acc | ECE | Δ Fashion | Δ KMNIST | retained F / K |
|---|---|---|---|---|---|
| all (fc1, fc2, fc3) | 0.933 | 0.056 | +0.073 | +0.069 | 1.00 / 1.00 |
| keep_fc1 | 0.928 | 0.018 | +0.012 | −0.018 | 0.16 / −0.26 |
| keep_fc2 | 0.932 | 0.032 | −0.073 | −0.070 | −1.00 / −1.01 |
| keep_fc3 | 0.924 | 0.043 | −0.098 | −0.130 | −1.34 / −1.88 |
| none (posterior mean) | 0.924 | 0.053 | −0.109 | −0.160 | −1.49 / −2.32 |
| *drop_fc3 (fc1 + fc2)* | 0.934 | 0.035 | +0.060 | +0.050 | 0.82 / 0.73 |
| *drop_fc2 (fc1 + fc3)* | 0.929 | 0.026 | +0.009 | −0.020 | 0.12 / −0.29 |
| *drop_fc1 (fc2 + fc3)* | 0.931 | 0.022 | −0.069 | −0.059 | −0.95 / −0.85 |

(*italic rows: exploratory follow-up, added after seeing the pre-registered variants.*)

- Parity: "all" reproduces E4 (+0.073 / +0.069 vs +0.076 / +0.069).
- **P1 FALSIFIED:** first-layer noise alone keeps ≤ 16% of the advantage.
- **P2 CONFIRMED:** last-layer noise alone keeps none (it is worse than the control).
- Exploratory: noise in fc1 *and* fc2 together keeps 73–82%, and each alone
  keeps ~0. The advantage is an interaction across the two hidden layers, not
  a property of any single layer. This fits last-layer Laplace's poor results
  (E2). The posterior-mean network alone is a much worse OOD detector than the
  plain DNN (−0.11 / −0.16).
- CONFIRMATORY for P1–P2; drop-one-layer rows EXPLORATORY.
