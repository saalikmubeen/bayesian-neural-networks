"""Paper figures from saved results (no training).

    python scripts/make_figures_paper.py   -> figures/paper_steps_vs_n.pdf, figures/paper_cifar.pdf
"""

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
Path("figures").mkdir(exist_ok=True)

# ── Fig: E4, same data, two step budgets ──
p = pd.read_csv("results/E45/paired_dnn.csv")
budgets = [(500, "bbb", 4800), (500, "bbb_steps23550", 23550), (2000, "bbb", 4800),
           (2000, "bbb_steps23550", 23550), (10000, "bbb_steps4800", 4800), (10000, "bbb", 23550)]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
for ax, ood, title in zip(axes, ["fashion", "kmnist"], ["MNIST → Fashion-MNIST", "MNIST → KMNIST"]):
    for j, n in enumerate([500, 2000, 10000]):
        for k, (bn, m, steps) in enumerate([b for b in budgets if b[0] == n]):
            r = p[(p.n == n) & (p.method == m) & (p.metric == f"auroc_{ood}_entropy")].iloc[0]
            x = j + (k - 0.5) * 0.36
            color = BLUE if steps == 4800 else ORANGE
            ax.bar(x, r["diff"], width=0.32, color=color, label=f"{steps:,} steps" if j == 0 else None)
            ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], color=INK, lw=0.8, capsize=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["n = 500", "n = 2,000", "n = 10,000"])
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
axes[0].set_ylabel("Δ AUROC vs softness-matched DNN")
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("figures/paper_steps_vs_n.pdf")

# ── Fig: CIFAR trajectories (E7 CNN, E7b MLP if present) ──
sets = [("results/E7", "CNN, n = 10,000")]
for n in (2000, 10000):
    if glob.glob(f"results/E7b/dyn_n{n}_*.csv"):
        sets.append((f"results/E7b:{n}", f"MLP, n = {n:,}"))
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
colors = [ORANGE, BLUE, AQUA]
for (src, label), color in zip(sets, colors):
    d, _, nn_ = src.partition(":")
    files = glob.glob(f"{d}/dyn_n{nn_}_*.csv") if nn_ else glob.glob(f"{d}/dyn_*.csv")
    df = pd.concat([pd.read_csv(f) for f in files])
    m = df.groupby("step").mean(numeric_only=True)
    for ax, col in zip(axes, ["delta_svhn", "delta_cifar100"]):
        ax.plot(m.index / 1000, m[col], color=color, lw=1.8, label=label)
for ax, title in zip(axes, ["CIFAR-10 → SVHN", "CIFAR-10 → CIFAR-100"]):
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.set_xlabel("steps (×1000)")
axes[0].set_ylabel("Δ AUROC vs softness-matched DNN")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig("figures/paper_cifar.pdf")
print("wrote figures/paper_steps_vs_n.pdf, figures/paper_cifar.pdf")
