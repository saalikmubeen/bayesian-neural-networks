"""Where the OOD window sits and how wide it is, as a function of n (from E6 + E9 logs).

    python scripts/analyze_window_scaling.py   -> results/E9/window_scaling.csv, figures/E9_window_vs_n.pdf

Per run (rho_init -5): d = (delta_fashion + delta_kmnist) / 2 per checkpoint, smoothed
with a 3-checkpoint moving average. Peak = step of max smoothed d. Window = the
contiguous steps around the peak where smoothed d >= half the peak value.
EXPLORATORY: not pre-registered (beyond E9's P3, the direction of the peak shift).
"""

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = glob.glob("results/E9/dyn_*.csv") + glob.glob("results/E6/dyn_n2000_rho-5_*.csv")
rows = []
for f in files:
    g = pd.read_csv(f).sort_values("step").reset_index(drop=True)
    if g.rho_init.iloc[0] != -5 or g.acc.max() < 0.5:
        continue
    d = ((g.delta_fashion + g.delta_kmnist) / 2).rolling(3, center=True, min_periods=1).mean().to_numpy()
    i = int(d.argmax())
    lo = i
    while lo > 0 and d[lo - 1] >= d[i] / 2:
        lo -= 1
    hi = i
    while hi < len(d) - 1 and d[hi + 1] >= d[i] / 2:
        hi += 1
    rows.append(dict(source=Path(f).parent.name, n=int(g.n.iloc[0]), seed=int(g.seed.iloc[0]),
                     peak_step=int(g.step[i]), peak_d=d[i], start=int(g.step[lo]), end=int(g.step[hi]),
                     width=int(g.step[hi] - g.step[lo]), closes=hi < len(d) - 1))
df = pd.DataFrame(rows).sort_values(["n", "source", "seed"])
df.to_csv("results/E9/window_scaling.csv", index=False, float_format="%.4f")
print(df.to_string(index=False))

agg = df.groupby("n")[["peak_step", "peak_d", "start", "end", "width"]].agg(["mean", "std"])
print(agg.round(0).to_string())
ns = df.n.to_numpy(float)
for col in ["peak_step", "width"]:
    slope, icpt = np.polyfit(np.log(ns), np.log(df[col].to_numpy(float)), 1)
    print(f"log-log slope of {col} vs n: {slope:.2f}  (per-run fit, {len(df)} runs)")

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.6, 2.6))
med = df.groupby("n").peak_step.median()
for n, g in df.groupby("n"):
    ax.vlines(np.full(len(g), n), g.start / 1000, g.end / 1000, color="#2a78d6", alpha=0.3, lw=7)
ax.scatter(df.n, df.peak_step / 1000, s=18, color="#2a78d6", edgecolor="white", zorder=3)
ax.plot(med.index, med / 1000, color="#0b0b0b", lw=1.2, label="median peak")
ax.axhline(40, color="#52514e", lw=0.8, ls="--")
ax.annotate("end of run (40k)", (520, 41), fontsize=7, color="#52514e")
ax.set_xscale("log")
ax.set_xticks([500, 2000, 10000])
ax.set_xticklabels(["500", "2,000", "10,000"])
ax.set_ylim(0, 45)
ax.set_xlabel("training set size n")
ax.set_ylabel("optimizer steps (×1000)")
ax.set_title("OOD window: peaks (dots), half-height spans (bars)", fontsize=8, loc="left")
ax.legend(frameon=False, fontsize=7, loc="center left")
fig.tight_layout()
fig.savefig("figures/E9_window_vs_n.pdf")
print("wrote figures/E9_window_vs_n.pdf")
