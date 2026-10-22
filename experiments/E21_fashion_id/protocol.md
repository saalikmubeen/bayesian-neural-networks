# Protocol: E21 — do the core results hold with Fashion-MNIST as the in-distribution data?

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E21_prereg.txt` before launch, self-attested.)

Background: every positive result so far uses MNIST as the in-distribution
(ID) data, with Fashion-MNIST and KMNIST as OOD sets. The obvious objection is
that the effect is specific to MNIST (e.g. its many always-blank pixels).
This experiment swaps the roles: Fashion-MNIST is ID, MNIST and KMNIST are OOD.
Nothing else changes (same MLP 784-256-256-10, N(0,1) prior, KL/N, Adam,
batch 64, stratified subsets, fixed 5k validation pool from the ID training
set, normalisation with the ID statistics).

- **Runs** (`bnnstudy.train --id fashion`, seeds 0–4, output `results/E21/`):
  n = 2000: `dnn` (40 epochs), `ensemble` (5 × 40 epochs), `bbb` (standard
  budget, 4,800 steps), `bbb_steps23550` (long), `dnn_steps23550` and
  `ensemble_steps23550` (step-matched deterministic baselines);
  n = 10000: `dnn`, `bbb` (standard budget there = 23,550 steps).
- **Metric:** Δ = entropy AUROC of method M minus that of the 40-epoch DNN
  softened on validation data to M's mean validation entropy
  (`dnn_Tmatch[M]`, `bnnstudy.evaluate`); MI-Δ uses M's MI-AUROC. Means over
  5 seeds; seed counts.
- **Predictions** (from the MNIST results: standard budget no advantage,
  long budget advantage, not reproduced by step-matched deterministic models):
  P1: at n = 2000 with the standard budget, `bbb` has mean entropy Δ ≤ +0.02
  on both OOD sets (no clear advantage).
  P2: at n = 2000, `bbb_steps23550` has mean entropy Δ ≥ +0.03 on both OOD
  sets, positive in ≥ 4/5 seeds on each.
  P3: at n = 2000, `bbb_steps23550`'s Δ exceeds the Δ of `dnn_steps23550` and
  of `ensemble_steps23550` (each against its own softness-matched DNN) in
  ≥ 4/5 seeds on both OOD sets.
  P4: at n = 10000, `bbb` (23,550 steps) has mean entropy Δ ≥ +0.03 on both
  OOD sets.
  P5: at n = 2000, `bbb_steps23550` has mean MI-Δ ≥ +0.03 on both OOD sets.
- **Falsified if:** P2 fails on both OOD sets (no long-training advantage with
  Fashion-MNIST as ID; the core result would then be MNIST-specific, and the
  paper must say so).
- **Config:** `scripts/run_E21.sh`, `scripts/analyze_E21.py`;
  `bnnstudy.evaluate` and `bnnstudy.compare` on `results/E21/`.
- **Cost:** ~30 min on two RTX 3090s. Go-ahead: user ("do 1 and 2"),
  2026-10-22.
- **Decision it informs:** whether the paper can claim its core results beyond
  MNIST as in-distribution data (TMLR reviewers' main scope question).

## Result (filled after running, 2026-10-22)
Files: `results/E21/*.{npz,json}` (40 runs), `results/E21/metrics.csv`,
`results/E21/paired_dnn.csv`, `results/E21/E21_deltas.csv`,
`results/E21/E21_summary.txt`; logs `logs/E21_gpu*.log`. Run time ~20 min.

Means over 5 seeds; Δ against the softness-matched 40-epoch DNN; in brackets
seeds with Δ > 0.

| n | method | acc | Δ MNIST | Δ KMNIST | MI Δ MNIST | MI Δ KMNIST |
|---|---|---|---|---|---|---|
| 2000 | DNN (reference) | 0.827 | | | | |
| 2000 | BBB, standard (4,800 steps) | 0.831 | +0.080 (5/5) | +0.066 (5/5) | +0.105 | +0.109 |
| 2000 | BBB, 23,550 steps | 0.762 | +0.136 (5/5) | +0.082 (5/5) | +0.015 | +0.205 |
| 2000 | DNN, 23,550 steps | 0.823 | +0.040 (5/5) | +0.018 (5/5) | | |
| 2000 | ensemble, 40 epochs | 0.829 | +0.038 (5/5) | +0.043 (5/5) | +0.099 | +0.112 |
| 2000 | ensemble, 23,550 steps | 0.827 | +0.085 (5/5) | +0.084 (5/5) | +0.092 | +0.094 |
| 10000 | DNN (reference) | 0.863 | | | | |
| 10000 | BBB, standard (23,550 steps) | 0.849 | +0.160 (5/5) | +0.206 (5/5) | +0.180 | +0.295 |

- **P1 FALSIFIED:** with the standard budget BBB already beats the control
  (+0.080, +0.066). "No advantage at standard budgets" is specific to MNIST
  as ID.
- **P2 CONFIRMED:** long-trained BBB +0.136 (MNIST) and +0.082 (KMNIST), 5/5
  seeds each.
- **P3 MIXED:** BBB's Δ exceeds the step-matched DNN's in 5/5 seeds on both
  OOD sets, and the step-matched ensemble's in 5/5 (MNIST) but only 2/5
  (KMNIST: +0.082 vs +0.084, a tie). In absolute entropy AUROC BBB is higher
  than the step-matched ensemble on both (0.980 vs 0.849; 0.827 vs 0.792), but
  the pre-registered comparison is on Δ.
- **P4 CONFIRMED:** n = 10000, +0.160 and +0.206.
- **P5 MIXED:** long BBB's MI-based Δ is +0.205 on KMNIST but only +0.015 on
  MNIST (its MI AUROC on MNIST, 0.859, is below its entropy AUROC, 0.980).
- Falsification condition not met. CONFIRMATORY.
- Not predicted: long training costs BBB 6.5 points of accuracy here (0.762
  vs 0.827 for the DNN). Plain deterministic baselines are much weaker OOD
  detectors with Fashion-MNIST as ID (entropy AUROC 0.68–0.84), so all Δs are
  larger, and even step-matched deterministic training helps a little
  (+0.018 to +0.040).
- What it rules out: the long-training advantage over softness- and
  step-matched plain networks being specific to MNIST as ID. What it
  qualifies: at which budget the advantage appears depends on the data, and
  the step-matched ensemble can tie BBB.
