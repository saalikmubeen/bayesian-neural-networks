# Protocol: E5 — OOD detection with the pixel-support shortcut removed

Written: 2026-10-21, before running. Status: planned

Background (interim E2 analysis, exploratory): a model-free rule ("any
ink on a pixel that is constant across the training images") detects
Fashion-MNIST and KMNIST with AUROC 0.987–1.000, as good as or better than
every trained model.

- **Hypothesis (with mechanism):** part of every method's OOD score, and most
  of the Bayesian methods' advantage, comes from ink on dead pixels. Ensembles
  get it from random init of dead-pixel weights, and converged mean-field VI
  gets it from prior-sized σ on those weights. Removing that ink tests what
  each method detects from the image content the training data actually covers.
- **Predictions** (paired over seeds 0–4, OOD sets `<ood>_onsupp` = OOD images
  with dead pixels reset to background):
  P1: the model-free rule drops to AUROC 0.5 on on-support sets (sanity check,
  true by construction).
  P2: at n = 10000, BBB's Δ vs its softness-matched DNN (entropy AUROC) shrinks
  by ≥ 50% on both on-support sets compared with the original sets.
  P3: the deep ensemble's Δ also shrinks on the on-support sets (any amount, in
  ≥ 4 of 5 seeds), because its diversity on dead-pixel weights is lost.
  P4: the plain DNN's AUROC drops by less (in absolute terms) than BBB's at
  n = 10000.
- **Falsified if:** BBB at n = 10000 keeps ≥ 50% of its Δ on both on-support
  sets (then its advantage is not mainly the shortcut).
- **Config:** `bnnstudy`, seeds 0–4, n ∈ {2000, 10000}. Methods: dnn, map, bbb,
  ensemble, mc_dropout, laplace, all with `--on-support --save-model`. Dead
  pixels are defined per training subset (constant across all its images).
  The ID test set is unchanged. Output: `results/E45/` (shared with E4).
- **Metrics:** entropy and MI AUROC on original and on-support OOD sets; Δ vs
  the softness-matched DNN (`dnn_Tmatch`) and vs the matched MAP net
  (`map_Tmatch`); the model-free rule's AUROC on both.
- **Cost:** shared with E4, ~25 min on 2 RTX 3090s. Go-ahead given by the user
  2026-10-21.
- **Decision it informs:** if P2 holds, MNIST-family OOD results for BNNs are
  confounded by a support shortcut, and on-support evaluation becomes the
  paper's recommended protocol. Next is the CIFAR padding experiment (E7):
  natural images have no dead pixels, so does a constant border recreate the
  effect?

## Result (filled after running, 2026-10-21)
Files: `results/E45/metrics.csv`, `paired_dnn.csv`, `paired_map.csv`.

Δ vs `dnn_Tmatch` (entropy AUROC), original → on-support, 5 seeds:

| n | method | Fashion | KMNIST |
|---|---|---|---|
| 10000 | bbb | +0.057 → +0.055 | +0.070 → +0.070 |
| 10000 | ensemble | +0.024 → +0.023 | +0.022 → +0.022 |
| 2000 | bbb_steps23550 | +0.076 → +0.078 | +0.069 → +0.076 |
| 2000 | ensemble | +0.016 → +0.017 | +0.018 → +0.018 |
| 2000 | mc_dropout | +0.011 → +0.014 | −0.009 → −0.006 |
| 2000 | laplace | −0.038 → −0.030 | −0.040 → −0.037 |

Raw AUROC barely moves (e.g. bbb n = 10000: Fashion 0.974 → 0.971, KMNIST
0.987 → 0.986; dnn n = 10000: 0.919 → 0.918).

- **P1** holds by construction (no ink on dead pixels remains).
- **P2 FALSIFIED:** BBB keeps 96–100% of its Δ on on-support sets.
- **P3 FALSIFIED:** the ensemble's Δ is unchanged.
- **P4 not supported:** neither the DNN nor BBB depends on dead-pixel ink.
- Conclusion: the model-free "ink on never-inked pixels" rule (AUROC 0.99+) is
  a real weakness of MNIST → Fashion/KMNIST as a benchmark, but none of the
  trained models exploit it, so it does not explain BBB's gain. The interim
  interim hypothesis is withdrawn. CONFIRMATORY.
