# Protocol: E3 — budget-matched MAP baseline

Written: 2026-10-21, before running. Status: planned

Motivation: an independent review. BBB trains for 150
epochs with gradient clipping and an N(0,1) prior. The DNN trains for 40
epochs with no weight decay or clipping. BBB's accuracy lead over the DNN
(+1.1 points in E1) could come from extra regularisation and compute rather
than from the posterior.

- **Hypothesis (with mechanism):** BBB's posterior mean already behaves like a
  MAP estimate under the same prior. A deterministic network trained to MAP
  under N(0,1) (L2 = ||w||² / (2N)), with BBB's epochs, learning rate,
  schedule and clipping, matches BBB's accuracy. Any accuracy lead of BBB
  over the plain DNN is therefore regularisation, not Bayesian averaging.
- **Predictions** (paired over seeds 0–9 at each n ∈ {500, 2000, 10000}):
  P1: |acc(bbb) − acc(map)| < 0.005 at n = 2000 and 10000.
  P2: OOD AUROC (entropy): Δ = bbb − map_Tmatch[bbb] has a 95% CI including 0
  or below 0 on both OOD sets at every n (no advantage beyond softness against
  the stronger base either).
- **Falsified if:** P1 — BBB more accurate than MAP by ≥ 0.005 with Wilcoxon
  p < 0.05 at n = 2000 or 10000 (then averaging helps accuracy). P2 — BBB's Δ > 0
  with CI excluding 0 on either OOD set at any n.
- **Config:** method `map` = MLP 784-256-256-10, Adam 1e-3, cosine schedule,
  150 epochs at n ≥ 2000 (×4 at n = 500, as for BBB), batch 64, gradient
  clipping 1.0, L2 ||w||²/(2·1²·N) on all weights and biases. Runs saved in
  `results/E2/` next to the E2 runs (same seeds, subsets and splits) so they
  pair. Temperature variants `map_Tnll`, `map_Tmatch[<ref>]` are fitted on
  validation only, as for the DNN.
- **Metrics:** accuracy (primary for P1); OOD AUROC with entropy, paired with
  bootstrap CI and Wilcoxon (primary for P2); NLL, ECE.
- **Cost:** 30 runs, ~10–15 min on one RTX 3090. Go-ahead given by user
  2026-10-21 ("queue the budget-matched DNN after E2").
- **Decision it informs:** if P1 holds, the report drops any accuracy claim for
  BBB and uses `map` as the main deterministic baseline. If P1 fails, BBB's
  accuracy gain is a (small) genuine benefit of posterior averaging to report.

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
