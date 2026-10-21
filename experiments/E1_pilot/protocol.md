# Protocol: E1 — pilot: KL-weight bug and softness-matched baseline

Written: 2026-10-21, before running. Status: planned

Background: the multi-seed course scripts (`step1_fixed_baselines.py`) train
BBB with `loss = CE_mean + KL / n_batches`. With CE averaged over a batch of
64, the correct per-example ELBO weight is `KL / N`, so the KL term is
over-weighted by the batch size (64×). The course notebook uses `1/N`
correctly; only the multi-seed scripts have the bug. Those scripts produced
the table where BBB is worst on every metric (acc 0.899, OOD AUROC 0.813,
ECE 0.157 at n=2000).

- **Hypothesis (with mechanism):**
  H1: the 64× KL over-weighting forces the posterior towards the prior,
  so BBB underfits and spreads probability on every input.
  H2: BBB's OOD separation is largely explained by global softness: a
  deterministic net softened by one temperature chosen to match BBB's
  in-distribution entropy (on validation data only) reaches similar OOD AUROC.
- **Predictions:**
  P0 (pipeline parity): `bbb_legacy` (old config, bug included) reproduces the
  old numbers within ±0.02 accuracy and ±0.03 AUROC (0.899 / 0.813).
  P1: `bbb` (KL/N, otherwise identical optimiser to `bbb_bug`) has higher
  accuracy and lower ECE than `bbb_bug`.
  P2: |AUROC(bbb) − AUROC(dnn_Tmatch)| < 0.02 on MNIST→Fashion-MNIST and
  MNIST→KMNIST (mean over 3 seeds).
- **Falsified if:** P1 — `bbb` accuracy not ≥ 0.01 above `bbb_bug`.
  P2 — `bbb` beats `dnn_Tmatch` by ≥ 0.02 AUROC on both OOD sets in all seeds
  (then the Bayesian advantage survives the control, which is itself the result).
- **Config:** MNIST train subset n=2000 (stratified, seeds 0/1/2); validation =
  5000 MNIST-train images disjoint from the subset; test = MNIST test (10k);
  OOD = Fashion-MNIST test, KMNIST test (10k each). MLP 784-256-256-10.
  Methods: `dnn` (Adam 1e-3, 40 ep, batch 64); `bbb` (KL/N, Adam 1e-3, 150 ep,
  KL anneal 20 ep, 50 MC samples); `bbb_bug` (same as `bbb` but KL/n_batches);
  `bbb_legacy` (exact old config: KL/n_batches, Adam 1e-4, 120 ep).
  Post-hoc on `dnn` logits: `dnn_Tnll` (temperature minimising val NLL) and
  `dnn_Tmatch[ref]` (temperature giving the same mean val entropy as `ref`).
- **Metrics:** primary — OOD AUROC with predictive entropy as score.
  Secondary — accuracy, NLL, ECE (15 bins), AURC, accuracy at 80% coverage,
  AUROC with max-prob and (sampled methods) mutual information.
- **Cost:** ~10 min on one RTX 3090. Go-ahead given by user 2026-10-21.
- **Decision it informs:** if P2 holds → the paper is "BBB's OOD advantage is
  softness; here is the control every BNN comparison needs", then scale to
  ensembles/Laplace/HMC and CIFAR. If P2 fails → the Bayesian advantage is
  real at small n; the paper characterises when and why.

## Result (filled after running, 2026-10-21)
Files: `results/E1_pilot/metrics.csv` (per seed), `summary.txt`, log `logs/E1_pilot.log`.
Run: `scripts/run_E1.sh` on one RTX 3090, ~4 min total.

| method | acc | ECE | AUROC Fashion (H) | AUROC KMNIST (H) | AUROC KMNIST (MI) |
|---|---|---|---|---|---|
| dnn | 0.925 ± 0.004 | 0.045 | 0.857 ± 0.045 | 0.872 ± 0.005 | — |
| dnn_Tnll (T = 2.05) | 0.925 | 0.008 | 0.867 ± 0.040 | 0.893 ± 0.008 | — |
| dnn_Tmatch[bbb] (T = 2.05) | 0.925 | 0.007 | 0.866 ± 0.040 | 0.893 ± 0.007 | — |
| bbb (KL/N) | **0.936 ± 0.003** | 0.022 | 0.832 ± 0.096 | 0.886 ± 0.005 | 0.895 |
| bbb_bug (KL/n_batches) | 0.904 ± 0.004 | 0.185 | 0.836 ± 0.018 | 0.895 ± 0.007 | **0.945** |
| bbb_legacy | 0.900 ± 0.002 | 0.154 | 0.848 ± 0.018 | 0.887 ± 0.013 | 0.927 |

(mean ± std over 3 seeds; H = predictive entropy, MI = mutual information)

- **P0 (parity):** accuracy 0.900 vs old 0.899 ✓, ECE 0.154 vs 0.157 ✓;
  Fashion AUROC 0.848 vs old 0.813 — just outside the ±0.03 band. Seed-to-seed
  spread of Fashion AUROC is ±0.02–0.10 for every method, so not alarming, but
  noted.
- **P1 confirmed:** fixing the KL weight raises BBB accuracy by 3.2 points
  (0.904 → 0.936) and cuts ECE 8× (0.185 → 0.022). Correct BBB is the most
  accurate and among the best-calibrated models.
- **P2:** no Bayesian advantage after the softness control. Paired per-seed
  BBB − dnn_Tmatch: Fashion +0.026 / −0.025 / −0.104, KMNIST −0.010 / −0.001 /
  −0.010. Mean diff −0.034 (Fashion) and −0.007 (KMNIST): within 0.02 on KMNIST,
  BBB *worse* on Fashion (driven by seed 2). Falsification condition (BBB ≥ +0.02
  in all seeds on both) not met.
- CONFIRMATORY for P1 and P2's falsification test; the rest EXPLORATORY.
- **Unexpected (exploratory):** (a) the entropy-matched temperature for correct
  BBB equals the NLL-optimal temperature (2.05 vs 2.05). (b) The over-regularised
  bbb_bug has the best KMNIST detection by MI (0.945 vs 0.895 for bbb) despite
  worse accuracy, so KL weight trades accuracy for MI-based detection. (c)
  Seed variance on Fashion (0.72–0.90 for bbb) dominates method differences:
  3 seeds cannot resolve them.
- Suggests: more seeds (≥10) with paired tests; sweep KL weight β as an
  explicit knob; add ensembles/MC dropout/Laplace and an HMC reference.
