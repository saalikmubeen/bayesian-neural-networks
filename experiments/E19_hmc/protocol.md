# Protocol: E19 — does the exact posterior behave like BBB? (HMC reference)

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E19_prereg.txt` before any E19 run, self-attested.)

Background: long-trained BBB beats a softness-matched DNN on OOD detection
(+0.06 to +0.08), and is effectively overpruned (only 1–5% of hidden weights
have |μ| > σ; E16). IVON, whose variances follow the data, shows no advantage
(E14). Open question from every review: is BBB's advantage a property of the
Bayesian posterior, or an artefact of mean-field VI under Adam?

- **Design:** full-batch Hamiltonian Monte Carlo on the same model as BBB:
  an MLP (784-100-100-10, ~90k parameters; width reduced from 256 so that HMC
  is feasible), N(0, 1) prior on all weights and biases, MNIST subsets
  n ∈ {500, 2000} (data seed 0). For a like-for-like comparison, BBB
  (`bbb_steps23550`, 23,550 steps) and the 40-epoch DNN are retrained at
  width 100 on the same subsets (seeds 0–2; seed 0 shares the HMC subset).
- **HMC settings (fixed now):** 4 chains per n, each initialised at a MAP
  estimate (full-batch Adam, 3000 steps, from a different random init per
  chain); identity mass matrix; 300 leapfrog steps per proposal; step size
  adapted during 1500 warm-up iterations towards 0.75 acceptance, then frozen;
  1000 sampling iterations, keeping every 10th (100 samples per chain, 400 in
  total). Predictions: Bayesian model average over all kept samples.
- **Convergence (decided now):** split-R̂ over the 4 chains, computed for the
  predicted probability of the true class on 1000 fixed test images. Results
  count as confirmatory only if the 90th percentile of these R̂ values is
  < 1.1 at that n; otherwise they are reported as exploratory.
- **Questions and predictions** (Δ = entropy-AUROC minus that of the width-100
  DNN softened to the method's validation entropy; OOD = Fashion-MNIST, KMNIST):
  P1 (decision rule, not a direction): if HMC's Δ_K and Δ_F are both
  ≥ +0.03, report "the exact posterior also has an OOD advantage beyond
  softness"; if both ≤ +0.01, report "BBB's advantage is not a property of the
  exact posterior"; otherwise inconclusive. Applied separately at each n.
  P2: HMC is less overpruned than BBB. Within each chain, take each weight's
  sample mean and standard deviation; the fraction of hidden-layer weights
  with |mean| > std is ≥ 2× BBB's fraction (width 100, same n, seed 0).
  P3: HMC's test accuracy is ≥ BBB's (width 100, seed 0) at both n.
- **Falsified if:** P2 — HMC's fraction is < BBB's (then the exact posterior
  is at least as "pruned"; overpruning would not be a mean-field artefact).
  P1 has no falsification condition (it is a decision rule).
- **Config:** `bnnstudy/hmc.py` (new; plain PyTorch, full-batch gradients);
  `bnnstudy.train --hidden 100` for BBB and the DNN; analysis
  `scripts/analyze_E19.py`; evaluation splits and metrics as elsewhere
  (`bnnstudy.softmatch.summarize`, incl. FPR95). Output `results/E19/`.
- **Cost:** estimated 1–3 GPU-hours (8 chains), plus ~10 min of BBB/DNN
  training. Go-ahead: user, 2026-10-22.
- **Decision it informs:** whether the paper can say "the exact posterior
  shows (or does not show) BBB's advantage", which decides whether the
  advantage is called Bayesian or an artefact of mean-field VI under Adam.

## Result (filled after running, 2026-10-22)
Files: `results/E19/hmc_n{2000,500}_c{0..3}.pt`, width-100 BBB/DNN models
`results/E19/{bbb_steps23550,dnn}_n{500,2000}_s{0,1,2}.*`, `results/E19/summary.txt`,
`results/E19/compare.csv`; script `scripts/analyze_E19.py`.

- **Convergence: FAILED at both n.** Split-R̂ (p(true class), 1000 test
  images): n = 2000 median 1.067, 90th pct 2.79; n = 500 median 1.44, 90th
  pct 2.54 (threshold < 1.1). Acceptance 0.70–0.75 as targeted, but the
  adapted step size was 2–5e-5, so one 300-step trajectory moves only
  ~0.01 in parameter space. The potential energy was still rising during
  sampling in all 8 chains. Per weight, the spread of the chain means
  across chains (median 0.64 at n = 2000, 0.75 at n = 500) is 6–8× the
  within-chain std (0.08, 0.13): each chain explored only a small region
  around its own MAP start.
- Numbers as measured (BBB and DNN width 100, seed 0, same subset):

| n | method | test acc | Δ_F | Δ_K | Δ_F (MI) | Δ_K (MI) | frac |μ|>σ |
|---|---|---|---|---|---|---|---|
| 2000 | "HMC" (4 chains) | 0.899 | +0.061 | +0.031 | +0.061 | +0.031 | 0.91 |
| 2000 | BBB | 0.920 | +0.072 | +0.039 | +0.109 | +0.082 | 0.022 |
| 500 | "HMC" (4 chains) | 0.833 | +0.003 | +0.000 | +0.004 | +0.001 | 0.87 |
| 500 | BBB | 0.881 | −0.036 | −0.065 | +0.098 | +0.065 | 0.016 |

  DNN (width 100, seed 0) test accuracy: 0.910 (n = 2000), 0.873 (n = 500).
- P1 (decision rule) nominally gives "advantage" at n = 2000 and
  "not a property of the exact posterior" at n = 500. Neither reading
  stands: the samples are not from the posterior. What was measured is a
  4-member ensemble of locally perturbed MAP solutions.
- P2 nominally holds (0.91 vs 0.022) but is **uninformative**: with
  trajectories this short, the within-chain std measures how far a chain
  moved, not posterior width. It cannot say whether the exact posterior
  is less overpruned than BBB.
- P3 fails at both n (HMC accuracy 0.899 < 0.920, 0.833 < 0.881); it is
  also below the DNN, which is another sign the chains had not mixed.
- EXPLORATORY (convergence criterion not met). E19 does not answer
  whether BBB's advantage is a property of the exact posterior. A real
  answer needs a better sampler (e.g. a preconditioned or adapted mass
  matrix, many more iterations, or a smaller model) and an R̂ check
  passed before any comparison. The paper keeps the question open.
