"""Score E15 (IVON with tempered likelihood, k in {10, 100}) against E14 (k = 1) and BBB (E9).

    python scripts/analyze_E15.py   -> results/E15/summary.txt, results/E15/per_run.csv, figures/E15_inflation.pdf
Protocol: experiments/E15_ivon_inflation/protocol.md
"""

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

frames = []
for f in sorted(glob.glob("results/E14/ivon_n*_s?.csv")):
    frames.append(pd.read_csv(f).assign(k=1))
empty = []
for f in sorted(glob.glob("results/E15/ivon_n*_s?_temp*.csv")):
    g = pd.read_csv(f)
    if g.empty:          # diverged before the first checkpoint
        empty.append(Path(f).name)
        continue
    frames.append(g.assign(k=int(Path(f).stem.split("_temp")[1])))
df = pd.concat(frames, ignore_index=True)
df = df[df.n.isin([2000, 10000])]
bbb = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/E9/dyn_n*_rho-5_s?.csv"))], ignore_index=True)

rows = []
for (n, k, s), g in df.groupby(["n", "k", "seed"]):
    end = g[g.step == g.step.max()].iloc[0]
    rows.append(dict(n=n, k=k, seed=s, sigma_40k=end.sigma_fc1, acc_40k=end.acc,
                     peak_dK=g.delta_kmnist.max(), peak_dF=g.delta_fashion.max(),
                     peak_dK_mi=g.delta_kmnist_mi.max(), dK_40k=end.delta_kmnist, dF_40k=end.delta_fashion,
                     fpr95_K_40k=end.fpr95_kmnist_entropy, fpr95_K_dnnT_40k=end.fpr95_kmnist_dnnT,
                     diverged=end.get("diverged_step")))
per = pd.DataFrame(rows)
per.to_csv("results/E15/per_run.csv", index=False, float_format="%.5f")
m = per.groupby(["n", "k"]).mean(numeric_only=True).drop(columns="seed")
lines = []
say = lambda s="": (print(s), lines.append(str(s)))
say(m.round(4).to_string())
say("\nRuns with no checkpoint (diverged before step 1000): " + str(empty))
expected = [f"ivon_n{n}_s{s}_temp{k}.csv" for n in (2000, 10000) for s in range(3) for k in (10, 100)]
say("Runs missing entirely (crashed): " + str([e for e in expected if not Path("results/E15", e).exists()]))
bb = bbb[bbb.n.isin([2000, 10000])].groupby(["n", "seed"]).agg(peak_dK=("delta_kmnist", "max")).groupby("n").mean()
say("\nBBB (E9) peak dK by n: " + str(bb.peak_dK.round(4).to_dict()))

for n in (2000, 10000):
    s1 = m.loc[(n, 1), "sigma_40k"]
    for k, need in ((10, 2), (100, 5)):
        if (n, k) not in m.index:
            say(f"P1 n={n} k={k}: no usable runs"); continue
        r = m.loc[(n, k), "sigma_40k"] / s1
        say(f"P1 n={n} k={k}: sigma ratio vs k=1 = {r:.2f} (need >= {need}) -> {'holds' if r >= need else 'fails'}")
for n in (2000, 10000):
    if (n, 100) not in m.index:
        say(f"P2 n={n} k=100: no usable runs -> fails"); continue
    pk = m.loc[(n, 100), "peak_dK"]
    say(f"P2 n={n} k=100: peak entropy dK = {pk:+.4f} (need >= +0.03) -> {'holds' if pk >= 0.03 else 'fails'}")
for n in (2000, 10000):
    d = m.loc[(n, 10), "acc_40k"] - m.loc[(n, 1), "acc_40k"]
    say(f"P3 n={n} k=10: accuracy change vs k=1 = {d:+.4f} (need |d| <= 0.02) -> {'holds' if abs(d) <= 0.02 else 'fails'}")
fals = all(m.loc[(n, k), "peak_dK"] < 0.02 for n in (2000, 10000) for k in (10, 100) if (n, k) in m.index)
say(f"Falsification condition (all peak dK < +0.02): {'MET' if fals else 'not met'}")
Path("results/E15/summary.txt").write_text("\n".join(lines) + "\n")

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "pdf.fonttype": 42})
C = {1: "#2a78d6", 10: "#eb6834", 100: "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
for ax, n in zip(axes, (2000, 10000)):
    for k in (1, 10, 100):
        g = df[(df.n == n) & (df.k == k)].groupby("step").mean(numeric_only=True)
        ax.plot(g.index / 1000, g.delta_kmnist, color=C[k], lw=1.8, label=f"IVON, k={k}")
    gb = bbb[bbb.n == n].groupby("step").mean(numeric_only=True)
    ax.plot(gb.index / 1000, gb.delta_kmnist, color="#52514e", lw=1.2, ls="--", label="BBB (Adam)")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_title(f"Δ entropy-AUROC (KMNIST), n = {n:,}", fontsize=8, loc="left")
    ax.set_xlabel("steps (×1000)")
axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig("figures/E15_inflation.pdf")
print("wrote figures/E15_inflation.pdf")
