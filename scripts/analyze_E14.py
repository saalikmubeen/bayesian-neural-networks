"""E14: pick IVON hyperparameters from the tuning runs, then score the main runs.

    python scripts/analyze_E14.py tune     -> prints the chosen (lr, hess_init) and writes results/E14/tune/choice.txt
    python scripts/analyze_E14.py score    -> results/E14/summary.txt, results/E14/per_run.csv, figures/E14_ivon_vs_bbb.pdf
Protocol: experiments/E14_ivon/protocol.md
"""

import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.argv[1] == "tune":
    rows = []
    for f in sorted(glob.glob("results/E14/tune/ivon_n2000_s99_*.csv")):
        g = pd.read_csv(f)
        last = g[g.step == g.step.max()].iloc[0]
        rows.append(dict(lr=last.lr, hess_init=last.hess_init, step=int(last.step), val_nll=last.val_nll,
                         diverged=last.get("diverged_step")))
    df = pd.DataFrame(rows).sort_values("val_nll")
    print(df.to_string(index=False))
    best = df.iloc[0]
    text = f"chosen by minimum validation NLL at the last checkpoint: lr={best.lr:g} hess_init={best.hess_init:g}\n\n{df.to_string(index=False)}\n"
    Path("results/E14/tune/choice.txt").write_text(text)
    print(text.splitlines()[0])
    sys.exit()

# ── score ─────────────────────────────────────────────────────────────────────
lines = []
say = lambda s="": (print(s), lines.append(str(s)))
iv = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/E14/ivon_n*_s?.csv"))], ignore_index=True)
bbb = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/E9/dyn_n*_rho-5_s?.csv"))], ignore_index=True)
dnn_acc = {}
for n in (500, 2000, 10000):
    for s in range(3):
        z = np.load(f"results/E45/dnn_n{n}_s{s}.npz")
        dnn_acc[(n, s)] = float((z["test_p"].argmax(-1) == z["test_y"]).mean())

rows = []
for (n, s), g in iv.groupby(["n", "seed"]):
    at = lambda st, c: float(g.loc[g.step == st, c].iloc[0]) if (g.step == st).any() else np.nan
    b = bbb[(bbb.n == n) & (bbb.seed == s)]
    rows.append(dict(n=n, seed=s, sigma_40k=at(40000, "sigma_fc1"), sigma_10k=at(10000, "sigma_fc1"),
                     dK_10k=at(10000, "delta_kmnist"), dK_40k=at(40000, "delta_kmnist"),
                     peak_dK=g.delta_kmnist.max(), peak_dF=g.delta_fashion.max(),
                     peak_dK_mi=g.delta_kmnist_mi.max(), acc_40k=at(40000, "acc"), dnn_acc=dnn_acc[(n, s)],
                     fpr95_K_40k=at(40000, "fpr95_kmnist_entropy"), fpr95_K_dnnT_40k=at(40000, "fpr95_kmnist_dnnT"),
                     bbb_peak_dK=b.delta_kmnist.max() if len(b) else np.nan,
                     bbb_sigma_40k=float(b.loc[b.step == 40000, "sigma_fc1"].iloc[0]) if len(b) else np.nan))
per = pd.DataFrame(rows)
per.to_csv("results/E14/per_run.csv", index=False, float_format="%.5f")
m = per.groupby("n").mean(numeric_only=True)
say(per.round(4).to_string(index=False))
say("\nmeans by n:\n" + m.round(4).to_string())

r = m.sigma_40k[500] / m.sigma_40k[10000]
say(f"\nP1: sigma_fc1 at 40k, n=500 / n=10000 = {m.sigma_40k[500]:.4f} / {m.sigma_40k[10000]:.4f} = {r:.2f}x -> "
    f"{'holds (>=2)' if r >= 2 else ('fails (<2)' if r >= 1.5 else 'FALSIFIED (<1.5)')}")
for n in (2000, 10000):
    say(f"P2 n={n}: sigma_fc1 at 40k = {m.sigma_40k[n]:.4f} (BBB {m.bbb_sigma_40k[n]:.3f}) -> {'holds (<=0.3)' if m.sigma_40k[n] <= 0.3 else 'fails'}")
for n in (500, 2000, 10000):
    d = abs(m.dK_40k[n] - m.dK_10k[n])
    say(f"P3 n={n}: |dK(40k) - dK(10k)| = |{m.dK_40k[n]:+.4f} - {m.dK_10k[n]:+.4f}| = {d:.4f} -> {'holds (<=0.02)' if d <= 0.02 else 'FAILS'}")
for n in (500, 2000, 10000):
    below = m.peak_dK[n] < m.bbb_peak_dK[n]
    nonneg = m.peak_dK[n] >= 0 if n >= 2000 else True
    say(f"P4 n={n}: IVON peak dK {m.peak_dK[n]:+.4f} vs BBB {m.bbb_peak_dK[n]:+.4f} -> "
        f"{'below BBB' if below else 'NOT below BBB'}{'' if n < 2000 else ('; >= 0' if nonneg else '; < 0')}")
for n in (500, 2000, 10000):
    d = m.acc_40k[n] - m.dnn_acc[n]
    say(f"P5 n={n}: IVON acc {m.acc_40k[n]:.4f} vs DNN {m.dnn_acc[n]:.4f} (diff {d:+.4f}) -> {'holds (|d|<=0.01)' if abs(d) <= 0.01 else 'fails'}")
say(f"\nFPR95 (KMNIST, entropy, 40k): IVON {m.fpr95_K_40k.round(3).to_dict()} vs softness-matched DNN {m.fpr95_K_dnnT_40k.round(3).to_dict()}")
Path("results/E14/summary.txt").write_text("\n".join(lines) + "\n")

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "pdf.fonttype": 42})
C = {500: "#2a78d6", 2000: "#eb6834", 10000: "#1baf7a"}
fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
for n in (500, 2000, 10000):
    gi = iv[iv.n == n].groupby("step").mean(numeric_only=True)
    gb = bbb[bbb.n == n].groupby("step").mean(numeric_only=True)
    axes[0].plot(gi.index / 1000, gi.sigma_fc1, color=C[n], lw=1.8, label=f"IVON, n={n:,}")
    axes[0].plot(gb.index / 1000, gb.sigma_fc1, color=C[n], lw=1.2, ls="--", label=f"BBB, n={n:,}")
    axes[1].plot(gi.index / 1000, gi.delta_kmnist, color=C[n], lw=1.8)
    axes[1].plot(gb.index / 1000, gb.delta_kmnist, color=C[n], lw=1.2, ls="--")
    axes[2].plot(gi.index / 1000, gi.acc, color=C[n], lw=1.8)
    axes[2].plot(gb.index / 1000, gb.acc, color=C[n], lw=1.2, ls="--")
axes[0].set_yscale("log")
for ax, t in zip(axes, ["mean posterior σ, layer 1 (log)", "Δ entropy-AUROC (KMNIST)", "test accuracy"]):
    ax.set_title(t, fontsize=8, loc="left")
    ax.set_xlabel("steps (×1000)")
axes[1].axhline(0, color="#52514e", lw=0.8)
axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig("figures/E14_ivon_vs_bbb.pdf")
print("wrote figures/E14_ivon_vs_bbb.pdf")
