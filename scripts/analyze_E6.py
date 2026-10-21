"""E6 analysis: score the pre-registered predictions and plot trajectories.

    python scripts/analyze_E6.py    -> results/E6/summary.txt, figures/E6_dynamics.pdf
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/E6")
df = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("dyn_*.csv"))], ignore_index=True)
lines = []
say = lambda s="": (print(s), lines.append(s))

# P1: delta tracks sigma (default init)
d5 = df[df.rho_init == -5]
rhos = [spearmanr(g.sigma_fc1, g.delta_kmnist)[0] for _, g in d5.groupby("seed")]
say(f"P1 Spearman(sigma_fc1, delta_kmnist) per seed, rho_init -5: {np.round(rhos, 3).tolist()}")

# P2: timing and plateau (default init)
at = lambda g, lo, hi, c: g[(g.step >= lo) & (g.step <= hi)][c].mean()
for s, g in d5.groupby("seed"):
    say(f"P2 seed {s}: dK@4-5k {at(g, 4000, 5000, 'delta_kmnist'):+.3f}  dK@25k {at(g, 25000, 25000, 'delta_kmnist'):+.3f}"
        f"  dK@30k {at(g, 30000, 30000, 'delta_kmnist'):+.3f}  dK@50k {at(g, 50000, 50000, 'delta_kmnist'):+.3f}"
        f"  peak dK {g.delta_kmnist.max():+.3f} at {int(g.step[g.delta_kmnist.idxmax()])}")

# P3: steps to first reach delta_kmnist >= 0.04
say()
for r, gr in df.groupby("rho_init"):
    first = []
    for _, g in gr.groupby("seed"):
        hit = g[g.delta_kmnist >= 0.04].step
        first.append(int(hit.min()) if len(hit) else np.inf)
    say(f"P3 rho_init {r:+g}: first step with dK >= 0.04 per seed {first}")

# P4: is the negative ELBO still decreasing at 4.8k?
e = d5.groupby("seed").apply(lambda g: pd.Series({"early": at(g, 4000, 5000, "neg_elbo"),
                                                   "late": at(g, 24000, 25000, "neg_elbo")}))
say(f"\nP4 neg ELBO 4-5k {e.early.mean():.2f} ± {e.early.std():.2f}  vs 24-25k {e.late.mean():.2f} ± {e.late.std():.2f}")

# End state per init
say("\nEnd state (step 50000), mean over seeds:")
end = df[df.step == df.step.max()].groupby("rho_init")[
    ["acc", "ece", "delta_fashion", "delta_kmnist", "sigma_fc1", "sigma_fc2", "sigma_fc3", "ce", "kl_per_n", "neg_elbo"]].mean()
say(end.round(4).to_string())
best = df.loc[df.groupby(["rho_init", "seed"]).delta_kmnist.idxmax()]
say("\nPeak delta_kmnist per run (step, acc, ece at the peak):")
say(best[["rho_init", "seed", "step", "delta_kmnist", "delta_fashion", "acc", "ece", "sigma_fc1"]].round(3).to_string(index=False))
(ROOT / "summary.txt").write_text("\n".join(lines) + "\n")

# Figure: small multiples, one line per sigma init (mean over seeds)
COLORS = {-5: "#2a78d6", -3: "#eb6834", -1: "#1baf7a", 0: "#eda100"}
LABEL = {-5: "σ₀ = 0.007", -3: "σ₀ = 0.049", -1: "σ₀ = 0.31", 0: "σ₀ = 0.69"}
panels = [("delta_kmnist", "Δ entropy-AUROC (KMNIST)"), ("delta_kmnist_mi", "Δ MI-AUROC (KMNIST)"),
          ("acc", "Test accuracy"),
          ("sigma_fc1", "Mean posterior σ, layer 1"), ("neg_elbo", "Negative ELBO per example")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 5, figsize=(12, 2.4))
mean = df.groupby(["rho_init", "step"]).mean(numeric_only=True).reset_index()
for ax, (col, title) in zip(axes, panels):
    for r, g in mean.groupby("rho_init"):
        ax.plot(g.step / 1000, g[col], color=COLORS[r], lw=2, label=LABEL[r])
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("steps (×1000)")
    if col == "neg_elbo":
        ax.set_yscale("log")
    if col.startswith("delta"):
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_ylim(-0.15, 0.12)
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/E6_dynamics.pdf")
print("wrote figures/E6_dynamics.pdf")
