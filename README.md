# Bayesian Neural Networks: Uncertainty Quantification

Comparing predictive uncertainty from a Bayes-by-Backprop (BBB) Bayesian neural network against deterministic baselines on MNIST (in-distribution) and Fashion-MNIST (out-of-distribution) data.

**Report:** [bnn.pdf](bnn.pdf)

## Overview

This repository implements a three-layer BNN from scratch in pure PyTorch using the reparameterization trick and ELBO training. The main notebook walks through:

- A deterministic MLP baseline
- A Bayesian MLP with weight posteriors trained via Bayes by Backprop
- Monte Carlo predictive uncertainty (entropy, confidence, epistemic std)
- OOD detection and rejection curves
- Posterior weight analysis and calibration plots

The extended scripts compare additional uncertainty baselines (MC Dropout, deep ensembles) and run adversarial / CIFAR-10 follow-up experiments.

## Key Results (MNIST, n = 2,000)

| Metric                   | DNN   | BNN   |
| ------------------------ | ----- | ----- |
| Test accuracy            | 0.919 | 0.923 |
| Mean confidence (ID)     | 0.968 | 0.890 |
| Mean confidence (OOD)    | 0.808 | 0.584 |
| Predictive entropy (ID)  | —     | 0.327 |
| Predictive entropy (OOD) | —     | 1.099 |

The BNN matches DNN accuracy while assigning lower confidence and higher entropy on OOD inputs.

## Repository Structure

```
bnn_project.ipynb          # Main course project notebook
bnn.pdf                    # Written report (linked above)
bnn_project.html           # HTML export of the notebook

step1_train_baselines.py   # Train DNN, BNN, MC Dropout, ensemble (multi-seed)
step2_adversarial_experiment.py   # FGSM / rotated MNIST OOD tests
step3_results_and_plots.py        # Aggregate plots from baseline runs

step1_fixed_baselines.py
step2_fixed_adversarial.py
step3_fixed_plots.py       # Corrected KL-weight / annealing variants

CIFAR-10 full analysis.py  # CIFAR-10 BNN vs DNN distance analysis
PGD attack.py              # PGD adversarial attack utilities

results/                   # Baseline experiment JSON outputs
results_fixed/             # Fixed-pipeline experiment outputs
figures_cifar/             # CIFAR-10 figures and results
figures_mechanism/         # PGD / mechanism analysis outputs
tables_fixed/              # Summary tables
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`.

## Running

**Main notebook (recommended starting point):**

```bash
jupyter notebook bnn_project.ipynb
```

Datasets (MNIST, Fashion-MNIST) download automatically into `data/` on first run. The notebook uses 2,000 training samples by default; change `TRAIN_SIZE` to use the full 60k set.

**Extended experiment pipeline:**

```bash
python step1_train_baselines.py      # ~20 min GPU / ~1 hr CPU
python step2_adversarial_experiment.py
python step3_results_and_plots.py
```

**CIFAR-10 analysis:**

```bash
python "CIFAR-10 full analysis.py"
```

Figures and JSON results are written to the directories listed above.

## License

MIT — see [LICENSE](LICENSE).
