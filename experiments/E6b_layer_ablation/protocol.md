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

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
