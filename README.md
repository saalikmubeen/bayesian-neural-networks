# When Does Bayes by Backprop Know What It Doesn't Know?

A study of out-of-distribution (OOD) detection with Bayesian neural networks,
built around a Bayes-by-Backprop (BBB) implementation written from scratch in
PyTorch. Every experiment's hypothesis, predictions and falsification
criteria were committed to git before it ran.

**Report:** [bnn.pdf](bnn.pdf) (9 pages; source `report/bnn.tex`)

## The question

A BNN averages predictions over a distribution of weights, so its outputs are
softer than a plain network's on *every* input. A plain network can be
softened for free with a temperature. So each method here is compared with a
plain network softened, on validation data only, to that method's own
in-distribution entropy. The question is whether the method separates OOD
inputs from in-distribution inputs better than a network of the same softness.

## Findings

On MLPs trained on MNIST (OOD sets: Fashion-MNIST, KMNIST):

1. **At standard training budgets, BBB is no better than the softness-matched
   network.** Deep ensembles are slightly better, MC dropout ties, and
   last-layer Laplace is worse (10 seeds per setting).
2. **BBB's advantage is set by the number of optimizer steps, not by the
   amount of data.** With the same data, about 23,500 steps give +0.06 to
   +0.08 AUROC over the control (5/5 seeds); 4,800 steps give none.
3. **It is a transient window.** As training continues, the KL term widens the
   posterior. For a while BBB beats the control and a deep ensemble. Then the
   ELBO keeps improving towards a near-prior posterior that underfits and loses
   the advantage. A large initial σ prevents learning altogether.
4. **It needs posterior noise in both hidden layers together**, and it is not
   explained by a pixel shortcut (removing ink on always-blank pixels changes
   nothing).
5. **Neither held-out NLL nor the ELBO finds the window. Synthetic outliers
   do:** picking the checkpoint that best separates validation digits from
   noise, flipped and pixel-permuted digits recovers 73% of the best achievable
   advantage, with no real OOD data.

On CIFAR-10 (OOD: SVHN, CIFAR-100), neither a small CNN nor the same MLP
shows a comparable window. There, even a deep ensemble loses to a single
softness-matched network on SVHN. The drift towards the prior does replicate.

![BBB training dynamics](figures/E6_dynamics.pdf)

## How the study got here

The project started from a simpler question ("is the BNN's advantage just
softness?"). Most of my pre-registered predictions failed: 16 of 29, listed
in the report's appendix. Those failures are what led to the training-window
account. Each step is in [experiments/](experiments/) (E1–E9, E7b), with the
prediction written first and the result added after the run.

## Methodological notes

- **KL scaling.** With the cross-entropy averaged over a mini-batch, the
  per-example ELBO needs `KL / N`. Dividing by the number of mini-batches
  instead over-weights the KL term by about the batch size (62.5× at N = 2,000)
  and costs about 3 points of accuracy. It is kept as an ablation (`bbb_bug`).
- **No fixed-threshold rejection rates.** Comparisons use threshold-free
  AUROC, paired by seed, with bootstrap CIs.
- **Every number in the report is traced** to a result file in
  [CLAIMS.md](CLAIMS.md).

## Repository layout

```
bnn.pdf                       report (source: report/bnn.tex, report/refs.bib)
bnnstudy/                     code
  data.py, models.py          MNIST data; DNN, MAP, MC dropout, ensemble, BBB, last-layer Laplace
  train.py, evaluate.py       one run -> per-sample predictions; metrics + softness-matched controls
  compare.py                  paired per-seed comparisons (bootstrap CI, Wilcoxon)
  dynamics.py                 BBB training dynamics with checkpoint-wise evaluation (E6, E9)
  cifar.py                    CIFAR-10 pipeline, CNN and MLP (E7, E7b)
  softmatch.py                softness-matched comparison for on-the-fly variants
experiments/<id>/protocol.md  pre-registered hypotheses and predictions, with results
scripts/                      run scripts, analysis and figure scripts
results/<id>/                 metrics and summaries (per-sample .npz and model .pt files are not tracked)
figures/                      report figures
COMMANDS.md, CLAIMS.md           every command run; every report number -> its source file
```

## Reproduce

Every command actually run, in order, is in [COMMANDS.md](COMMANDS.md).
Developed with torch 1.9 (CUDA 11.1) on RTX 3090s. The datasets download into
`data/` on first run.

```bash
pip install -r requirements.txt
scripts/run_E1.sh                          # pilot, ~4 min
scripts/run_E6.sh 0 -5 -3                  # training dynamics, ~15 min per GPU
python scripts/analyze_E6.py
cd report && pdflatex bnn && bibtex bnn && pdflatex bnn && pdflatex bnn
```

## License

MIT — see [LICENSE](LICENSE).
