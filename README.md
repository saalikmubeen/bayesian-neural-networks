# Bayes by Backprop's OOD Advantage Depends on Where Training Stops

A study of out-of-distribution (OOD) detection with Bayesian neural networks,
built around a Bayes-by-Backprop (BBB) implementation written from scratch in
PyTorch. Every experiment's hypothesis, predictions and falsification
criteria were committed to git before it ran.

**Report:** [bnn.pdf](bnn.pdf) (10 pages; source `report/bnn.tex`)

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
   +0.08 AUROC over the control (5/5 seeds); 4,800 steps give none. The
   posterior standard deviations follow almost the same path for 500 to
   10,000 training images, which suggests the optimizer (Adam) sets the pace.
   This is a hypothesis I have not yet tested directly.
3. **The uncertainty score matters.** Along training, the entropy-based
   advantage can rise and then disappear as the model starts to underfit and
   becomes uncertain on normal digits too. The mutual-information (MI)
   advantage persists. But MI is also high for runs that never learned the
   task, so MI alone does not show a useful model.
4. **The advantage needs posterior noise in both hidden layers together**, and
   it is not explained by a pixel shortcut (removing ink on always-blank pixels
   changes nothing).
5. **Choosing when to stop is unsolved.** Held-out NLL stops too early and the
   ELBO too late. A synthetic-outlier selection rule does no better than
   stopping at a fixed step count.

On CIFAR-10 with small, underfit models (37–68% accuracy), no comparable
advantage appears. That result says little about well-trained networks.

![BBB training dynamics](figures/E6_dynamics.png)

## How the study got here

The project started from a simpler question ("is the BNN's advantage just
softness?"). Most of my pre-registered predictions failed: 16 of 29, listed
in the report's appendix. Those failures are what led to the training-path
account. Each step is in [experiments/](experiments/) (E1–E9, E7b), with the
prediction written first and the result added after the run.

## Limitations and next steps

The N(0,1) prior is misspecified for these layers and has not been swept; the
dynamics runs use a constant learning rate and 3 seeds; there is no
exact-posterior (HMC) reference; and the CIFAR models are small and underfit.
Planned next experiments: learning-rate and SGD tests of the optimizer-clock hypothesis, a prior-scale sweep, ResNet-20 on CIFAR-10, HMC.

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
