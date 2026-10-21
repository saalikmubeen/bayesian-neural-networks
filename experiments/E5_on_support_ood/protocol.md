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

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
