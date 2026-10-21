# Protocol: E4 — is BBB's OOD gain set by optimizer steps or by data size?

Written: 2026-10-21, before running. Status: planned

Background (interim E2 analysis, exploratory): BBB beats its
softness-matched DNN only at n = 10000, where fixed epochs give 23550 steps,
versus 4800 at n = 500 and 2000. The gain correlates with ink on
"dead" pixels (constant in every training image).

- **Hypothesis (with mechanism):** first-layer weights from dead pixels get no
  likelihood gradient. Their mean-field σ is pushed only by the KL term, from
  its init (softplus(−5) ≈ 0.007) towards the prior (σ = 1), and that takes
  many Adam steps at lr 1e-3. Once these σ are large, any OOD ink on dead
  pixels makes the posterior samples disagree. So the OOD gain depends on the
  number of optimizer steps, not on n.
- **Predictions** (paired over seeds 0–4):
  P1 (steps, not n): at n = 2000, `bbb_steps23550` beats its softness-matched
  DNN on KMNIST (entropy AUROC) by ≥ 0.03 in ≥ 4 of 5 seeds, while `bbb`
  (4800 steps) does not. At n = 10000, `bbb_steps4800` loses most of the gain
  of `bbb` (23550 steps): its Δ vs matched DNN is less than half of `bbb`'s on
  both OOD sets.
  P2 (where the σ grow): in saved `bbb` models with 23550 steps, the mean σ of
  first-layer weights from dead pixels is > 5× that from live pixels. With
  4800 steps the ratio is smaller.
  P3: n = 500 with 23550 steps behaves like n = 2000 with 23550 steps (gain
  present), despite 4× less data.
- **Falsified if:** at equal steps the gain still tracks n (n = 2000 with
  23550 steps shows no gain, or n = 10000 with 4800 steps keeps most of it),
  or the σ ratio for dead vs live pixels is < 2 in the 23550-step models.
- **Config:** `bnnstudy`, MNIST subsets as E2, seeds 0–4. Methods: `bbb`
  (default epochs), `bbb_steps23550` at n ∈ {500, 2000}, `bbb_steps4800` at
  n = 10000. KL warm-up keeps the 20/150 fraction of training. Models saved
  (`--save-model`), on-support OOD variants added (`--on-support`). The DNN
  base for the softness match is the E5 `dnn` run from the same directory.
  Output: `results/E45/`.
- **Metrics:** Δ AUROC (entropy and MI) vs `dnn_Tmatch[<method>]`, paired by
  seed; per-pixel posterior σ of fc1 (mean over hidden units) for dead vs live
  pixels.
- **Cost:** shared with E5, ~25 min on 2 RTX 3090s. Go-ahead given by the user
  2026-10-21.
- **Decision it informs:** if confirmed, the paper's central claim becomes
  "BBB's OOD advantage on MNIST-type data is a pixel-support detector switched
  on by optimisation length". If falsified, the n = 10000 gain needs another
  explanation before any claim is made.

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
