"""Mechanism figures (no training).

    PYTHONPATH=. python scripts/make_figures_mechanism.py
      -> figures/E16_overpruning.pdf  (|mu| vs sigma per weight, BBB at 23,550 steps, n = 2000, seed 0)
      -> figures/E18_r1.pdf           (AUROC of the analytic first-layer score R1 vs BBB's sampled MI, seeds 5-9)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})

# Fig: overpruning
sd = torch.load("results/E45/bbb_steps23550_n2000_s0.pt", map_location="cpu")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
rng = np.random.default_rng(0)
for ax, layer in zip(axes, ["fc1", "fc2", "fc3"]):
    mu = sd[f"{layer}.mu"].abs().flatten().numpy()
    s = F.softplus(sd[f"{layer}.rho"]).flatten().numpy()
    idx = rng.choice(len(mu), min(20000, len(mu)), replace=False)
    ax.scatter(s[idx], mu[idx], s=1, alpha=0.25, color=BLUE, rasterized=True)
    lim = max(s.max(), mu.max()) * 1.05
    xs = np.linspace(max(s.min(), 1e-3), s.max(), 200)
    ax.plot(xs, xs, color=ORANGE, lw=1, label="|μ| = σ")
    ax.set_yscale("log")
    ax.set_xlim(0, s.max() * 1.05)
    ax.set_xlabel("posterior σ")
    frac = np.mean(mu > s)
    ax.set_title(f"{layer}: |μ| > σ for {100 * frac:.1f}% of weights", fontsize=8, loc="left")
axes[0].set_ylabel("|μ| (log)")
axes[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("figures/E16_overpruning.pdf", dpi=200)

# Fig: R1 vs MI
df = pd.read_csv("results/E18/per_seed.csv")
order = ["fashion", "kmnist", "cifar10_gray", "svhn_gray", "rotated"]
labels = ["Fashion", "KMNIST", "CIFAR-10\n(gray)", "SVHN\n(gray)", "rotated\ndigits"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
for ax, n in zip(axes, (2000, 10000)):
    m = df[df.n == n].groupby("ood")[["auroc_R1", "auroc_MI"]].mean().reindex(order)
    x = np.arange(len(order))
    ax.bar(x - 0.18, m.auroc_MI, width=0.34, color=BLUE, label="BBB, sampled MI (50 samples)")
    ax.bar(x + 0.18, m.auroc_R1, width=0.34, color=AQUA, label="first-layer R₁ (one pass)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.8, 1.0)
    ax.set_title(f"n = {n:,} (fresh seeds 5–9)", fontsize=8, loc="left")
axes[0].set_ylabel("OOD AUROC")
axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout()
fig.savefig("figures/E18_r1.pdf")
print("wrote figures/E16_overpruning.pdf, figures/E18_r1.pdf")
