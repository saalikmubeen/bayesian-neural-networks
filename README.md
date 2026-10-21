# Is the Bayesian OOD Advantage Just Softness?

A study of uncertainty in Bayesian neural networks, built around a
Bayes-by-Backprop (BBB) implementation written from scratch in PyTorch. One
question drives it: **does a Bayesian neural network still detect
out-of-distribution (OOD) inputs better once a plain network is given the
same overall softness?**

**Report:** [bnn.pdf](bnn.pdf)

## The question

A BNN averages predictions over a distribution of weights. Its outputs are
therefore softer (higher-entropy) than a plain network's on every input, and
many comparisons read "higher entropy on OOD data" as "the BNN knows what it
doesn't know". Temperature scaling also softens a plain network, at no cost.
The fair question is whether the BNN separates OOD inputs from
in-distribution inputs better than a plain network of the **same softness**.
Here the temperature is fitted on validation data only.

## Pilot result (MNIST, n = 2,000, 3 seeds)

| Model | Accuracy | ECE | OOD AUROC Fashion-MNIST | OOD AUROC KMNIST |
| ----- | -------- | --- | ----------------------- | ---------------- |
| DNN | 0.925 | 0.045 | 0.857 | 0.872 |
| DNN, softness-matched (T = 2.05) | 0.925 | **0.007** | **0.866** | **0.893** |
| BNN (BBB, KL weight 1/N) | **0.936** | 0.022 | 0.832 | 0.886 |
| BNN (BBB, KL weight 1/n_batches) | 0.904 | 0.185 | 0.836 | 0.895 |

- **No detectable OOD advantage for the BNN.** On Fashion-MNIST even the plain
  DNN does as well; on KMNIST the BNN's small lead (+0.014) disappears once the
  DNN is softened. My pre-registered prediction of equivalence held on KMNIST
  but failed on Fashion-MNIST, where the BNN was worse.
- **The matched temperature equals ordinary temperature scaling (T = 2.05)**,
  so here the control is the standard calibration baseline.
- **One hint of a real Bayesian signal:** scored by mutual information, an
  over-regularised BNN beats its softness-matched DNN on KMNIST in every seed
  (not on Fashion-MNIST). The KL-weight sweep in E2 tests this.
- **Three seeds are too few.** Fashion-MNIST AUROC ranges from 0.72 to 0.90 for
  the same method across seeds. A 10-seed study is under way, covering MC
  dropout, deep ensembles, last-layer Laplace, three training sizes and a
  KL-weight sweep ([protocol](experiments/E2_mnist_grid/protocol.md)).

Details, per-seed numbers and the pre-registered predictions are in
[experiments/E1_pilot/protocol.md](experiments/E1_pilot/protocol.md).

## Methodological notes

1. **KL scaling.** With the cross-entropy averaged over a mini-batch, the
   per-example ELBO needs `KL / N`. Dividing by the number of mini-batches
   instead is an easy slip, and it over-weights the KL term by about the
   batch size (62.5× at N = 2,000, batch 64). The over-weighted posterior
   underfits: accuracy falls from 0.936 to 0.904 and ECE rises from 0.022 to
   0.185. This variant is kept as an ablation (`bbb_bug`).
2. **No fixed-threshold rejection rates.** Thresholding entropy for one model
   and 1 − max-probability for another compares different scales.
   Comparisons use threshold-free AUROC, paired by seed.
3. **Process.** Each hypothesis and prediction is committed to git before the
   run. Metrics are recomputed from saved per-sample predictions, and every
   number in the report is traced to a file in [CLAIMS.md](CLAIMS.md).

## Repository layout

```
bnn.pdf                    revised report (source: report/bnn.tex)
bnnstudy/                  study code
  data.py                  MNIST subsets, disjoint validation pool, OOD sets
  models.py                DNN, MC dropout, ensembles, BBB, last-layer Laplace
  train.py                 one run -> per-sample predictions (.npz)
  evaluate.py              metrics + temperature-scaled / softness-matched DNNs
  compare.py               paired per-seed comparisons with bootstrap CIs
experiments/<id>/protocol.md   pre-registered hypotheses, predictions, results
scripts/                   run scripts and figure scripts
results/<id>/              metrics.csv, summary.txt (per-sample .npz not tracked)
figures/                   report figures
COMMANDS.md, CLAIMS.md           every command run; every report number -> its source file
```

## Reproduce

```bash
pip install -r requirements.txt
scripts/run_E1.sh                       # pilot, ~4 min on one GPU
python -m bnnstudy.evaluate results/E1_pilot
python scripts/make_figures_E1.py
cd report && pdflatex bnn && bibtex bnn && pdflatex bnn && pdflatex bnn
```

Every command actually run, in order: [COMMANDS.md](COMMANDS.md). Developed
with torch 1.9 (CUDA 11.1) on an RTX 3090. MNIST, Fashion-MNIST and KMNIST
download into `data/` on first run.

## License

MIT — see [LICENSE](LICENSE).
