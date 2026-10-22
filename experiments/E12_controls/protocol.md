# Protocol: E12 — is BBB's long-training OOD advantage Bayesian, and is it fair?

Written: 2026-10-22, before running. Status: planned. (Not committed before
running; SHA-256 and launch time recorded in `experiments/E12_E13_prereg.txt`
before the first run, self-attested.)

Background: BBB trained for 23,550 steps beats a DNN softened to its
validation entropy by +0.06 to +0.08 AUROC and scores above every
deterministic score tried (E4, strong-baseline analysis). Reviewers objected:
(i) any weight noise might do this; (ii) the DNNs and ensembles had smaller
step budgets; (iii) feature-space detectors were not tried.

- **Hypotheses:**
  H1 (Bayesian vs any noise): BBB's learned, per-weight σ matters. Isotropic
  weight noise added to a trained DNN, scaled per layer and matched to BBB's
  validation entropy, does *not* reach BBB's OOD AUROC.
  H2 (budget): training the DNN and the 5-member ensemble for the same 23,550
  steps does not close the gap to BBB.
  H3 (feature detectors): Mahalanobis and kNN detectors on DNN penultimate
  features are strong on these easy OOD sets; I expect them to be comparable
  to BBB (no strong prediction of a winner).
- **Predictions** (n ∈ {2000, 10000}, seeds 0–4, OOD = Fashion-MNIST and KMNIST;
  BBB = `bbb_steps23550` at n = 2000 and `bbb` at n = 10000, both 23,550
  steps, from `results/E45`; AUROC by predictive entropy unless stated):
  P1: BBB − (DNN + softness-matched weight noise) ≥ +0.02 in ≥ 4 of 5 seeds,
  on both OOD sets at both n.
  P2: BBB − (step-matched DNN softened to BBB's validation entropy) ≥ +0.04 on
  average on both OOD sets at both n, with the 95% bootstrap CI excluding 0.
  P3: BBB − (step-matched 5-member ensemble) > 0 on average on both OOD sets at
  both n, in ≥ 4 of 5 seeds.
  P4 (descriptive): report Mahalanobis and kNN AUROC; I expect both ≥ 0.95 and
  within 0.03 of BBB on average.
- **Falsified if:** P1 — the noise control comes within 0.02 of BBB (or beats
  it) on either OOD set at either n in ≥ 3 of 5 seeds; then the advantage is
  "weight noise", not specifically Bayesian. P2/P3 — the step-matched
  baselines match BBB.
- **Config:**
  - Weight noise: load `results/E45/dnn_n<n>_s<seed>.pt` (40-epoch DNN). For
    each layer l add N(0, (s·std(W_l))²) to weights and biases independently,
    50 samples at test time. Choose s from a log grid (0.01–3, 25 values) so
    that the mean validation entropy equals BBB's; also report the best s
    chosen with OOD knowledge (an upper bound). Scores: entropy and MI.
  - Step-matched: `dnn_steps23550` and `ensemble_steps23550` (Adam 1e-3,
    cosine over the run, batch 64; epochs = ceil(23550 / n_batches)), seeds
    0–4, `--save-model`, output `results/E12/`. The DNN is softened to BBB's
    validation entropy as in the main control.
  - Feature detectors on the 40-epoch DNN and on the step-matched DNN:
    penultimate (256-d, post-ReLU) features. Mahalanobis: class means and a
    shared covariance from the training subset (+1e-6·I); score = minimum
    class Mahalanobis distance. kNN: L2-normalised features; score = distance
    to the 10th nearest training feature (k = 10 for both n).
- **Metrics:** AUROC (test vs OOD); paired differences BBB − control per
  seed; mean, 95% bootstrap CI, wins out of 5.
- **Cost:** step-matched training ~30 min on 2 RTX 3090s; the rest is
  inference. Go-ahead: user, 2026-10-22.
- **Decision it informs:** whether the paper can say "Bayesian" (H1), and
  whether "beats every deterministic score / ensembles" survives fair budgets
  and standard detectors (H2–H4).

## Result (filled after running, 2026-10-22)
Files: `results/E12/{noise,matched,features}.csv`, `results/E12/summary.txt`;
script `scripts/analyze_E12.py`; training logs `logs/E12_E13_gpu*.log`.
5 seeds per cell; differences BBB − control, mean [95% bootstrap CI], wins/5.
Step-matched DNN accuracy 0.927 (n = 2000) and 0.969 (n = 10000); step-matched
ensemble 0.931 and 0.972.

| control | n | Fashion (entropy) | KMNIST (entropy) |
|---|---|---|---|
| DNN + weight noise (softness-matched) | 2000 | +0.144 [0.110, 0.171] 5/5 | +0.150 [0.138, 0.161] 5/5 |
| DNN + weight noise (softness-matched) | 10000 | +0.108 [0.085, 0.132] 5/5 | +0.102 [0.089, 0.116] 5/5 |
| DNN + weight noise (best scale, uses OOD) | 2000 / 10000 | +0.088 / +0.067, 5/5 | +0.097 / +0.084, 5/5 |
| step-matched DNN, softened to BBB | 2000 | +0.087 [0.052, 0.117] 5/5 | +0.075 [0.063, 0.087] 5/5 |
| step-matched DNN, softened to BBB | 10000 | +0.040 [0.033, 0.048] 5/5 | +0.078 [0.065, 0.097] 5/5 |
| step-matched ensemble | 2000 | +0.058 [0.051, 0.067] 5/5 | +0.075 [0.070, 0.080] 5/5 |
| step-matched ensemble | 10000 | +0.041 [0.036, 0.046] 5/5 | +0.058 [0.055, 0.062] 5/5 |
| Mahalanobis (40-ep / step-matched DNN) | 2000 | +0.022 / +0.021, 5/5 | +0.030 / +0.037, 5/5 |
| Mahalanobis (40-ep / step-matched DNN) | 10000 | +0.092 / +0.109, 5/5 | +0.059 / +0.059, 5/5 |
| kNN (40-ep / step-matched DNN) | 2000 | +0.022 / +0.023, 5/5 | +0.022 / +0.017, 5/5 |
| kNN (40-ep / step-matched DNN) | 10000 | +0.010 / +0.006 [−0.002, 0.016], 5/5 / 2/5 | +0.021 / +0.023, 5/5 |

Raw AUROC: weight noise (matched) 0.81–0.89; step-matched DNN softened
0.86–0.93; step-matched ensemble 0.89–0.93; Mahalanobis 0.86–0.94; kNN
0.93–0.97; BBB 0.95–0.99 (entropy), 0.96–0.99 (MI).

- **P1 CONFIRMED:** weight noise falls far short of BBB (≥ +0.10 matched,
  ≥ +0.067 even at the OOD-chosen best scale), 5/5 seeds everywhere. The
  advantage is not "any weight noise"; BBB's learned σ matter.
- **P2 CONFIRMED** (narrowly at n = 10000 Fashion: mean +0.040, threshold 0.04):
  a step-matched DNN softened to BBB's entropy does not close the gap.
- **P3 CONFIRMED:** BBB beats the step-matched 5-member ensemble in 5/5
  seeds everywhere (+0.041 to +0.075 entropy; +0.054 to +0.082 MI).
- **P4 mixed:** kNN is, as expected, within 0.03 of BBB (+0.006 to +0.023; a tie
  at n = 10000 Fashion with the step-matched DNN). Mahalanobis does *not*
  reach 0.95 (0.86–0.94) and is clearly below BBB.
- CONFIRMATORY for P1–P4 as stated.
