"""Figures for the E1 pilot, from results/E1_pilot/metrics.csv only.

    python scripts/make_figures_E1.py      -> figures/E1_kl_bug.pdf, figures/E1_paired_auroc.pdf
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/E1_pilot/metrics.csv")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# reference palette, categorical slots 1-3 (validated all-pairs in light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42,
})

df = pd.read_csv(SRC)


def dot_panel(ax, methods, col, title, ylabel=None, fmt="{:.3f}"):
    """Per-seed dots (lines join the same seed) and a labelled mean per method."""
    wide = df[df.method.isin([m for m, _, _ in methods])].pivot(index="seed", columns="method", values=col)
    xs = range(len(methods))
    for _, row in wide.iterrows():
        ax.plot(xs, [row[m] for m, _, _ in methods], color=GRID, lw=1, zorder=1)
    for x, (m, label, color) in zip(xs, methods):
        ax.scatter([x] * len(wide), wide[m], s=22, color=color, edgecolor="white", linewidth=1, zorder=3)
        mean = wide[m].mean()
        ax.plot([x - 0.18, x + 0.18], [mean, mean], color=INK, lw=1.5, zorder=4)
        ax.annotate(fmt.format(mean), (x + 0.22, mean), va="center", fontsize=7, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label, _ in methods])
    ax.set_xlim(-0.5, len(methods) - 0.2)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel)


# Fig 1: effect of the KL weight
bug = [("bbb_bug", "BBB\nKL / n_batches", ORANGE), ("bbb", "BBB\nKL / N", ORANGE),
       ("dnn", "DNN", BLUE)]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
dot_panel(axes[0], bug, "acc", "Test accuracy (higher is better)")
dot_panel(axes[1], bug, "ece", "Expected calibration error (lower is better)")
fig.tight_layout()
fig.savefig(OUT / "E1_kl_bug.pdf")

# Fig 2: OOD AUROC against the softness-matched DNN
ood = [("dnn", "DNN", BLUE), ("dnn_Tmatch[bbb]", "DNN, softness-\nmatched (T≈2.05)", AQUA),
       ("bbb", "BBB\n(KL / N)", ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), sharey=True)
dot_panel(axes[0], ood, "auroc_fashion_entropy", "MNIST → Fashion-MNIST", ylabel="OOD AUROC (entropy)")
dot_panel(axes[1], ood, "auroc_kmnist_entropy", "MNIST → KMNIST")
fig.tight_layout()
fig.savefig(OUT / "E1_paired_auroc.pdf")
print("wrote", *sorted(OUT.glob("E1_*.pdf")))
