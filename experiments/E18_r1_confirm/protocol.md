# Protocol: E18 — confirming the first-layer noise-to-signal account on fresh models and new OOD sets

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E18_prereg.txt` before any E18 training run, self-attested.)

Background (E17, exploratory): for long-trained BBB, the first-layer
noise-to-signal ratio R₁(x) = Σ_j v_j / Σ_j m_j², with m_j = Σ_i μ_ij x_i + μ_bj
and v_j = Σ_i σ_ij² x_i² + σ_bj², separated MNIST from Fashion-MNIST and
KMNIST almost exactly as well as BBB's sampled mutual information (AUROC
0.96–0.99 vs 0.96–0.995), and tracked BBB's per-image MI (Spearman 0.71–0.83).
That was found on seeds 0–4 and on the two OOD sets used throughout, so it
needs confirmation on fresh models and unseen OOD data.

- **Hypothesis:** R₁, computed analytically in one pass from BBB's learned μ
  and σ, captures where BBB's OOD signal comes from, on new models and new OOD
  sets as well.
- **Fresh models:** BBB with the same configuration as before, trained anew on
  seeds 5–9: `bbb_steps23550` at n = 2000 and `bbb` (23,550 steps) at
  n = 10000 (`bnnstudy.train --save-model`), output `results/E18/`.
- **OOD sets:** Fashion-MNIST and KMNIST (as before) plus two sets never used
  with R₁: grayscale CIFAR-10 test (10k) and grayscale SVHN test (fixed
  10k subset), both converted to luminance, resized to 28×28 (bilinear) and
  normalised with MNIST statistics. Near-OOD: the MNIST test images rotated
  by 90°.
- **Predictions** (seeds 5–9, means over seeds; "far OOD" = Fashion, KMNIST,
  CIFAR-10-gray, SVHN-gray):
  P1: AUROC(R₁) ≥ 0.90 on every far-OOD set at both n.
  P2: |AUROC(R₁) − AUROC(BBB MI)| ≤ 0.03 on every far-OOD set at both n.
  P3: Spearman(R₁, MI) over test + OOD images pooled is ≥ 0.6 on every
  far-OOD set at both n.
  P4 (near OOD, rotated digits): |AUROC(R₁) − AUROC(BBB MI)| ≤ 0.05 at both n
  (R₁ tracks BBB even where both may be weaker).
- **Falsified if:** P1 fails on ≥ 2 far-OOD sets at either n, or P3 is below
  0.4 on ≥ 2 far-OOD sets at either n.
- **Config:** `scripts/analyze_E18.py`; MI from `BBB.predict` with 50
  samples; R₁ as in `scripts/analyze_E17.py`. Training: 10 BBB runs
  (~1.5 min each) on 2 RTX 3090s.
- **Cost:** ~15 min. Go-ahead: user, 2026-10-22.
- **Decision it informs:** whether the paper can state the first-layer
  noise-to-signal account as a confirmed finding, and whether R₁ is a usable
  sampling-free OOD score for BBB.

## Result (filled after running, 2026-10-22)
Files: fresh models `results/E18/{bbb_steps23550_n2000,bbb_n10000}_s{5..9}.{pt,npz}`
(test accuracy 0.927–0.938 at n = 2000, mean 0.963 at n = 10000);
`results/E18/per_seed.csv`, `results/E18/summary.txt`; script
`scripts/analyze_E18.py`; logs `logs/E18_gpu*.log`. Means over seeds 5–9.

| n | OOD set | AUROC R₁ | AUROC BBB MI | |difference| | Spearman(R₁, MI) |
|---|---|---|---|---|---|
| 2000 | Fashion-MNIST | 0.960 | 0.955 | 0.005 | 0.71 |
| 2000 | KMNIST | 0.982 | 0.979 | 0.003 | 0.77 |
| 2000 | CIFAR-10 gray (new) | 0.987 | 0.999 | 0.012 | 0.73 |
| 2000 | SVHN gray (new) | 0.985 | 0.995 | 0.010 | 0.72 |
| 2000 | rotated digits (near) | 0.875 | 0.894 | 0.020 | 0.50 |
| 10000 | Fashion-MNIST | 0.983 | 0.986 | 0.003 | 0.81 |
| 10000 | KMNIST | 0.991 | 0.994 | 0.004 | 0.83 |
| 10000 | CIFAR-10 gray (new) | 0.981 | 1.000 | 0.019 | 0.72 |
| 10000 | SVHN gray (new) | 0.984 | 0.997 | 0.013 | 0.72 |
| 10000 | rotated digits (near) | 0.863 | 0.950 | 0.086 | 0.58 |

- **P1 CONFIRMED:** AUROC(R₁) 0.960–0.991 on all four far-OOD sets at both n.
- **P2 CONFIRMED:** R₁ is within 0.003–0.019 of BBB's sampled MI.
- **P3 CONFIRMED:** Spearman(R₁, MI) 0.71–0.83.
- **P4 holds at n = 2000 (0.020), FAILS at n = 10000 (0.086):** on near-OOD
  rotated digits, BBB's full MI (0.950) beats the first-layer score (0.863).
- Falsification condition not met. The first-layer noise-to-signal account
  is confirmed on fresh models and on two OOD sets never used before
  (CIFAR-10 and SVHN in grayscale): a single analytic pass over BBB's
  first-layer μ and σ reproduces its 50-sample OOD detection for far-OOD
  inputs. For near-OOD inputs that look like digits, deeper layers add signal
  that R₁ misses.
- CONFIRMATORY for P1–P4.
