# Protocol: E6 — training dynamics of BBB's OOD advantage, and σ initialisation

Written: 2026-10-21, before running. Status: planned

Background: E4 showed that BBB's OOD advantage over a softness-matched DNN
appears with 23550 Adam steps but not with 4800, at every n. First-layer σ
grows from 0.007 (init) to ~0.1 (4800 steps) and ~0.8–0.97 (23550 steps).

- **Hypothesis (with mechanism):** the advantage appears when the
  variational σ, pushed by the KL term towards the prior, become large enough
  that posterior samples disagree on inputs far from the data. The amount of
  disagreement on ID inputs stays limited by the likelihood. This is a
  convergence process: AUROC Δ rises together with mean σ and with the
  negative ELBO's KL term, and plateaus once σ plateaus. Starting σ closer to
  its end value (rho_init −1 or 0 instead of −5) reaches the plateau in fewer
  steps.
- **Predictions** (n = 2000, seeds 0–2, constant lr 1e-3, 50k steps, eval
  every 1000 steps):
  P1: Δ_KMNIST (entropy AUROC vs dnn_Tmatch at that checkpoint) correlates
  with mean fc1 σ across checkpoints (Spearman ρ > 0.8 per run, default init).
  P2: with the default init (rho −5), Δ_KMNIST at 4000–5000 steps is < 0.01
  and reaches > 0.05 by 25k steps. Δ changes by < 0.01 between 30k and 50k
  steps (plateau).
  P3: steps needed to first reach Δ_KMNIST ≥ 0.04 fall monotonically as
  rho_init goes −5 → −3 → −1 → 0 (mean over seeds).
  P4: the negative ELBO is still decreasing at 4800 steps: its mean over steps
  4000–5000 exceeds its mean over 24000–25000 by more than its seed SD.
- **Falsified if:** Δ does not track σ (ρ < 0.5), Δ keeps rising after 30k
  steps by > 0.02 (no plateau), or init has no monotone effect on time-to-Δ.
- **Config:** `bnnstudy/dynamics.py`. BBB (KL/N, N(0,1) prior, warm-up over
  the first 640 steps), Adam lr 1e-3 constant (no cosine, so that every
  checkpoint is a valid "train for k steps" model up to the LR schedule),
  batch 64, grad clip 1.0. Eval with 20 MC samples on val (5k), MNIST test,
  Fashion, KMNIST. rho_init ∈ {−5, −3, −1, 0}; seeds 0–2. The DNN for the
  softness match is `results/E45/dnn_n2000_s<seed>.npz` (same subset).
- **Metrics per checkpoint:** test acc, NLL, ECE; entropy and MI AUROC
  (Fashion, KMNIST); Δ vs softness-matched DNN; mean σ per layer; running
  CE and KL/N (negative ELBO = CE + KL/N after warm-up).
- **Cost:** 12 runs × ~4 min, ~25 min on 2 RTX 3090s. Go-ahead given by the
  user 2026-10-21.
- **Decision it informs:** if P1–P3 hold, the paper's central figure is "Δ vs
  steps tracks σ", and the practical recommendation is to check σ/ELBO
  convergence (or initialise σ larger) before comparing BBB with other
  methods.

## Result (filled after running)
- Outcome:
- CONFIRMATORY / EXPLORATORY
- What it rules out / suggests:
