# Protocol: E22 — does the advantage survive a near-OOD split that a pixel rule cannot solve?

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E22_prereg.txt` before launch, self-attested.)

Background: an independent TMLR-style review of the draft pointed out that a
model-free "ink on an always-background pixel" rule separates MNIST from
Fashion-MNIST and KMNIST with AUROC 0.967–1.000 (E5), so the far-OOD
benchmarks are nearly trivial, and asked for a near-OOD or held-out-class
split. Here the network is trained on MNIST digits 0–4 only; the test digits
5–9 are the near-OOD set (same pen strokes, same canvas), with Fashion-MNIST
and KMNIST as far-OOD sets for reference.

- **Data** (`bnnstudy.data`, `ID = "mnist04"`): stratified training subsets
  of digits 0–4 (n = 2000, 400 per class, one per seed), a fixed 5k
  validation pool of digits 0–4, test digits 0–4 (5,139) as ID test set,
  test digits 5–9 (4,861) as OOD set `mnist59`. Same MLP, 10 outputs (the
  five unused classes are never targets), same training as elsewhere.
- **Runs** (`bnnstudy.train --id mnist04`, n = 2000, seeds 0–4, output
  `results/E22/`): `dnn`, `ensemble`, `bbb` (standard, 4,800 steps),
  `bbb_steps23550`, `dnn_steps23550`, `ensemble_steps23550`.
- **Metric:** Δ as in E21 (entropy AUROC of M minus that of the 40-epoch DNN
  softened to M's validation entropy); MI-Δ with M's MI.
- **Predictions:**
  P1: long-trained BBB (`bbb_steps23550`) has mean Δ ≥ +0.02 on held-out
  digits, positive in ≥ 4/5 seeds.
  P2: that Δ is smaller than its mean Δ over the two far-OOD sets.
  P3: long-trained BBB's Δ on held-out digits exceeds that of
  `dnn_steps23550` and of `ensemble_steps23550` in ≥ 4/5 seeds each.
  P4: the pixel rule (flag ink on any pixel constant across the training
  subset) has AUROC ≤ 0.80 on held-out digits (the split is not trivial).
  P5: standard-budget BBB (`bbb`) has mean Δ ≤ +0.02 on held-out digits.
- **Falsified if:** P1 fails (then the advantage does not extend to near-OOD
  inputs, and the paper's scope must be stated as far-OOD only).
- **Config:** `scripts/run_E22.sh`, `scripts/analyze_E22.py`;
  `bnnstudy.evaluate` and `bnnstudy.compare` on `results/E22/`.
- **Cost:** ~15 min on two RTX 3090s. Go-ahead: user put the assistant in
  charge of the TMLR revision ("do 1 and 2 ... you're in charge"), 2026-10-22.
- **Decision it informs:** whether the paper can claim an advantage on a
  near-OOD split, or must restrict its scope to far-OOD benchmarks.

## Result (filled after running, 2026-10-22)
Files: `results/E22/*.{npz,json}` (30 runs), `results/E22/metrics.csv`,
`results/E22/paired_dnn.csv`, `results/E22/E22_deltas.csv`,
`results/E22/E22_summary.txt`; logs `logs/E22_gpu*.log`. Run time ~10 min.

Means over 5 seeds (seeds with Δ > 0). DNN: accuracy 0.979, entropy AUROC
0.890 (held-out digits), 0.897 (Fashion), 0.922 (KMNIST).

| method | acc | held-out digits: AUROC, Δ | Fashion: AUROC, Δ | KMNIST: AUROC, Δ |
|---|---|---|---|---|
| BBB, standard (4,800 steps) | 0.982 | 0.915, +0.023 (5/5) | 0.926, +0.020 (4/5) | 0.930, −0.004 (1/5) |
| BBB, 23,550 steps | 0.981 | 0.904, +0.011 (4/5) | 0.958, +0.047 (5/5) | 0.972, +0.034 (5/5) |
| DNN, 23,550 steps | 0.979 | 0.898, +0.010 (5/5) | 0.898, +0.004 (3/5) | 0.927, +0.009 (5/5) |
| ensemble, 40 epochs | 0.979 | 0.895, +0.006 (5/5) | 0.911, +0.014 (5/5) | 0.930, +0.007 (5/5) |
| ensemble, 23,550 steps | 0.980 | 0.906, +0.018 (5/5) | 0.922, +0.027 (5/5) | 0.940, +0.021 (5/5) |

- **P1 FALSIFIED:** long-trained BBB's Δ on held-out digits is +0.011
  (4/5 seeds), below the +0.02 threshold.
- **P2 CONFIRMED:** +0.011 on held-out digits vs +0.041 on far-OOD sets.
- **P3 FALSIFIED:** BBB's Δ exceeds the step-matched DNN's in 3/5 seeds and
  the step-matched ensemble's in 1/5 (their means +0.010 and +0.018).
- **P4 CONFIRMED:** the pixel rule reaches AUROC 0.569 (0.540–0.605) on
  held-out digits, against 1.000 and 0.999 on Fashion-MNIST and KMNIST: this
  split is not solvable by the shortcut.
- **P5 FALSIFIED:** standard-budget BBB has Δ +0.023 (5/5) on held-out digits,
  larger than long-trained BBB's.
- **Falsification condition MET.** CONFIRMATORY.
- What it rules out: an OOD advantage of long-trained BBB beyond softness on
  near-OOD inputs (held-out classes of the same dataset). The paper's claims
  must be scoped to far-OOD detection. It is consistent with E18, where the
  first-layer account failed on rotated digits.
