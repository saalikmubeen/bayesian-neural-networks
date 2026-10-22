# Protocol: E16 — do BBB's means have to co-adapt with its variances?

Written: 2026-10-22, before running. Status: planned. (SHA-256 recorded in
`experiments/E16_prereg.txt` before running, self-attested.)

Background: BBB trained for 23,550 steps beats a softness-matched DNN by
+0.06 to +0.08 AUROC. Adding noise to a trained DNN does not reproduce this
(E12), and inflating IVON's variances does not either (E15). Hypothesis left
standing: BBB's advantage needs large σ *together with* mean weights that were
trained jointly with that noise.

- **Hypotheses:**
  H1 (co-adaptation): BBB's learned σ applied to a deterministic network's
  means (trained for the same number of steps) does not give BBB's advantage;
  the noise mostly destroys the deterministic network.
  H2 (pattern vs scale): BBB's σ within a layer is nearly uniform (E4: dead/
  live ratio 1.0–1.24), so shuffling σ values within each layer of BBB keeps
  most of the advantage; what matters is BBB's means plus noise of the right
  scale, not which weight gets which σ.
- **Variants** (inference only, 50 posterior samples; for every variant, Δ =
  entropy-AUROC minus that of the DNN softened to the variant's own validation
  entropy, as in all earlier experiments):
  A. BBB as trained (μ_BBB, σ_BBB): reference.
  B. Swap: μ from the step-matched DNN (`results/E12/dnn_steps23550_n<n>_s<seed>.pt`),
     σ_BBB at the same positions.
  C. Shuffled σ: μ_BBB, σ_BBB randomly permuted within each weight matrix and
     each bias vector (fixed permutation seed).
- **Predictions** (n ∈ {2000, 10000}; seeds 0–4; BBB = `bbb_steps23550` at
  n = 2000 and `bbb` at n = 10000 from `results/E45`):
  P1: variant B's test accuracy is ≥ 0.05 below the step-matched DNN's, in
  ≥ 4 of 5 seeds at both n.
  P2: variant B's Δ_KMNIST and Δ_Fashion are each ≥ 0.05 below variant A's
  (mean over seeds) at both n.
  P3: variant C keeps ≥ 80% of variant A's Δ on both OOD sets at both n
  (mean over seeds), and its accuracy is within 0.01 of A's.
- **Falsified if:** P2 fails at both n (then the deterministic means work as
  well as BBB's, so co-adaptation is not needed), or P3 fails at both n (then
  the per-weight σ pattern matters, not only its scale).
- **Config:** `scripts/analyze_E16.py`; same data, splits and DNN base
  (`results/E45/dnn_n<n>_s<seed>.npz`) as E12; `bnnstudy.softmatch.summarize`
  for metrics (incl. FPR95). No training.
- **Cost:** inference only, a few minutes. Go-ahead: user, 2026-10-22.
- **Decision it informs:** if P1–P3 hold, the mechanism statement becomes
  "the advantage comes from BBB's means co-adapting with near-uniform,
  large posterior noise during joint training".

## Result (filled after running, 2026-10-22)
Files: `results/E16/variants.csv`, `results/E16/summary.txt`; script
`scripts/analyze_E16.py`. 5 seeds per cell; means shown.

| n | variant | accuracy | Δ Fashion | Δ KMNIST | FPR95 KMNIST |
|---|---|---|---|---|---|
| 2000 | A: BBB as trained | 0.933 | +0.073 | +0.069 | 0.18 |
| 2000 | B: DNN means + BBB σ | 0.126 | −0.350 | −0.401 | 0.95 |
| 2000 | C: BBB means + shuffled σ | 0.807 | −0.157 | −0.189 | 0.87 |
| 10000 | A: BBB as trained | 0.964 | +0.056 | +0.070 | 0.06 |
| 10000 | B: DNN means + BBB σ | 0.147 | −0.404 | −0.398 | 0.95 |
| 10000 | C: BBB means + shuffled σ | 0.588 | −0.324 | −0.306 | 0.94 |

(Step-matched DNN accuracy: 0.927 and 0.969.)

- **P1 CONFIRMED:** BBB's σ on the step-matched DNN's means collapses it to
  0.11–0.17 accuracy, 5/5 seeds at both n.
- **P2 CONFIRMED:** Δ(A) − Δ(B) = +0.42 to +0.47.
- **P3 FAILED (falsification condition met):** shuffling σ within each layer
  destroys both accuracy (0.81, 0.59) and the advantage (Δ −0.16 to −0.32). The
  per-weight σ pattern matters, not only its scale. My premise that σ is
  "nearly uniform" (from E4's dead/live *means*) was wrong.
- Exploratory follow-up (not pre-registered; seed 0 models): per-weight σ is
  widely spread, not uniform. fc1/fc2 have a σ median of 0.71–0.84 with a
  10th–90th percentile range of about 0.28–0.99. Only 1.2–4.5% of hidden weights
  have |μ| > σ, and |μ| and σ are strongly anti-correlated in fc2 (−0.64 to
  −0.72). The last layer is nearly deterministic (σ median 0.06–0.11). The
  long-trained BBB is effectively an overpruned network (cf. Trippe & Turner):
  a small, low-noise subnetwork computes the prediction and most other weights
  carry prior-scale noise.
- Reading: the advantage needs BBB's own means *and* its learned allocation
  of noise (small σ exactly on the weights that carry signal). Plain weight
  noise (E12), tempered IVON (E15), foreign means (B) and shuffled σ (C) all
  break that pairing. Hypothesis for next test: in-distribution inputs are
  computed mainly through the low-noise subnetwork, while OOD inputs also
  drive the high-noise weights, so posterior samples disagree on them.
- CONFIRMATORY for P1–P3 as stated; the σ-distribution analysis is EXPLORATORY.
