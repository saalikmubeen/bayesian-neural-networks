# Protocol: E23 — controls for the HMC comparison, and trivial baselines for R₁

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E23_prereg.txt` before launch, self-attested.)

Background: an external review of the TMLR draft raised two gaps.
(A) In E20/E20b the softness-matched network is the plain width-100 DNN,
trained without a prior, while HMC and BBB use an N(0, 0.3²) prior, so the HMC
advantage could partly be prior regularisation (and ensembling) rather than
posterior averaging; and HMC averages 800 samples while BBB uses 50.
(B) R₁ reaches AUROC 0.96–0.99 on far-OOD sets where a trivial pixel rule is
near 1.0, so "R₁ is within 0.02 of BBB's MI" may say little; R₁ should be
compared with trivial input statistics as predictors of BBB's per-image MI.

**Part A (HMC controls).** Width 100, n ∈ {500, 2000}, data seeds 0–2 (the
three HMC subsets of E20/E20b). New runs (`bnnstudy.train --hidden 100`,
output `results/E23/`): `map_sp0.3` (MAP under N(0, 0.3²), BBB's 150-epoch
budget, clipping) and `map_ensemble_sp0.3` (5 such MAP networks, different
inits). BBB (`results/E20/bbb_steps23550_sp0.3_*`) re-evaluated with 800
posterior samples (as many as HMC). Δ as in E20 (against the width-100 E19 DNN
softened to the method's validation entropy).
- P1: HMC's entropy Δ_K and Δ_F exceed the MAP ensemble's on every subset at
  both n.
- P2: single MAP (same prior) has mean entropy Δ_K ≤ +0.02 at both n.
- P3: BBB with 800 samples changes the mean (over subsets) entropy Δ_K, Δ_F
  and MI Δ_K by ≤ 0.01 relative to 50 samples, at both n.
- Falsified if: the MAP ensemble's mean Δ is ≥ HMC's on both OOD sets at both
  n (then the HMC advantage is explained by prior plus ensembling).

**Part B (R₁ against trivial statistics).** The ten E18 BBB models (seeds 5–9,
n ∈ {2000, 10000}), the same OOD sets as E18 (Fashion-MNIST, KMNIST,
CIFAR-10 gray, SVHN gray; rotated digits as near-OOD). Baselines computed from
the input alone: L1 norm of the [0,1] image (total ink), ink count (pixels >
0.5), and the E5 pixel rule (ink on pixels constant across the training
subset).
- P4: Spearman(R₁, BBB MI) over test + OOD images exceeds Spearman(b, BBB MI)
  for every baseline b on each of the four far-OOD sets, at both n (means over
  seeds).
- P5: on rotated digits, R₁'s AUROC exceeds every baseline's AUROC at both n.
- Falsified if: some baseline's Spearman with MI is ≥ R₁'s on ≥ 2 of the 4
  far-OOD sets at either n (then R₁ adds little over trivial input statistics
  as an account of BBB's MI, and the mechanism claim must be softened).

- **Config:** `scripts/run_E23.sh`, `scripts/analyze_E23.py` (part A),
  `scripts/analyze_E23_r1.py` (part B).
- **Cost:** ~15 min of training, a few minutes of inference. Go-ahead: user
  asked to act on the external review (items 1–4), 2026-10-22.
- **Decision it informs:** whether the paper can attribute HMC's advantage to
  posterior averaging rather than the prior, and whether the R₁ mechanism
  claim survives trivial baselines.

## Result (filled after running, 2026-10-22)
Files: `results/E23/{map_sp0.3,map_ensemble_sp0.3}_n{500,2000}_s{0,1,2}.*`,
`results/E23/summary.txt`, `results/E23/controls.csv` (part A);
`results/E23/r1_baselines.txt`, `results/E23/r1_baselines.csv` (part B). Note:
`scripts/analyze_E23.py` had a syntax error when hashed; fixed before any
analysis ran, amendment and new hash in `experiments/E23_prereg.txt`.

**Part A** (means over data seeds 0–2; Δ vs the softened width-100 DNN):

| n | method | acc | Δ_F | Δ_K | MI Δ_F | MI Δ_K |
|---|---|---|---|---|---|---|
| 500 | DNN | 0.870 | | | | |
| 500 | HMC | 0.887 | +0.077 | +0.060 | +0.084 | +0.065 |
| 500 | BBB, 50 samples | 0.784 | −0.009 | −0.032 | +0.065 | +0.065 |
| 500 | BBB, 800 samples | 0.792 | +0.012 | +0.000 | +0.076 | +0.069 |
| 500 | MAP, same prior | 0.868 | −0.064 | −0.015 | | |
| 500 | MAP ensemble (5), same prior | 0.867 | −0.053 | −0.014 | −0.020 | −0.023 |
| 2000 | DNN | 0.917 | | | | |
| 2000 | HMC | 0.931 | +0.088 | +0.075 | +0.092 | +0.078 |
| 2000 | BBB, 50 samples | 0.906 | +0.035 | +0.030 | +0.047 | +0.071 |
| 2000 | BBB, 800 samples | 0.908 | +0.041 | +0.039 | +0.057 | +0.077 |
| 2000 | MAP, same prior | 0.927 | +0.017 | +0.017 | | |
| 2000 | MAP ensemble (5), same prior | 0.929 | +0.033 | +0.024 | +0.055 | +0.035 |

- **P1 CONFIRMED:** HMC's entropy Δ_K and Δ_F exceed the MAP ensemble's on
  every subset at both n (3/3 each).
- **P2 CONFIRMED:** single MAP mean Δ_K −0.015 (n = 500), +0.017 (n = 2000).
- **P3 MIXED:** 800 vs 50 samples changes BBB by ≤ 0.01 at n = 2000 (0.0095,
  0.0063, 0.0052) but not at n = 500 (entropy Δ_K +0.032, Δ_F +0.022; MI Δ_K
  +0.004): 50 samples understate BBB's entropy-based advantage at n = 500.
- Falsification not met: the HMC advantage is not explained by the prior plus
  ensembling.

**Part B** (E18 models; means over seeds 5–9; |Spearman| with BBB's per-image
MI; AUROC of baselines in their better direction):
- On all four far-OOD sets at both n, the pixel rule correlates with BBB's MI
  at least as well as R₁ (|ρ| 0.772–0.865 vs 0.711–0.825); the L1 norm also
  matches or exceeds R₁ on CIFAR-10 gray and SVHN gray (0.729–0.740 vs
  0.722–0.726).
- **P4 FALSIFIED** at both n; **falsification condition MET**.
- **P5 CONFIRMED:** on rotated digits R₁ reaches AUROC 0.875 / 0.863 while
  every baseline is at chance (0.500–0.548).
- What it rules out: R₁ as an explanation of BBB's far-OOD signal beyond
  where the ink is. R₁ remains a sampling-free score that matches BBB's MI
  AUROC and carries signal beyond input statistics on near-OOD rotations.
- CONFIRMATORY for both parts.
