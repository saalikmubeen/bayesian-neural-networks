"""Score E10 (optimizer clock) and E11 (prior scale) against their protocols.

    python scripts/analyze_E10_E11.py   -> results/E10/summary.txt, results/E11/summary.txt, figures/E10_clock.pdf

Reference arms come from E9 (Adam lr 1e-3, sigma_p 1, same seeds).
"""

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pattern, **extra):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        for k, v in extra.items():
            d[k] = v
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def first_step(g, col, thr):
    hit = g[g[col] >= thr].step
    return int(hit.min()) if len(hit) else np.inf


# ── E10 ───────────────────────────────────────────────────────────────────────
arms = {"adam 5e-4": ("results/E10/dyn_n{n}_rho-5_s*_lr5e-4.csv", 5e-4),
        "adam 1e-3": ("results/E9/dyn_n{n}_rho-5_s?.csv", 1e-3),
        "adam 2e-3": ("results/E10/dyn_n{n}_rho-5_s*_lr2e-3.csv", 2e-3),
        "sgd (rho)": ("results/E10/dyn_n{n}_rho-5_s*_sgd.csv", None)}
rows, curves = [], {}
for n in (500, 10000):
    for arm, (pat, lr) in arms.items():
        df = load(pat.format(n=n))
        if df.empty:
            continue
        curves[(n, arm)] = df.groupby("step").sigma_fc1.mean()
        for s, g in df.groupby("seed"):
            rows.append(dict(n=n, arm=arm, lr=lr, seed=s,
                             t03=first_step(g, "sigma_fc1", 0.3),
                             onset=first_step(g, "delta_kmnist", 0.04),
                             sigma_20k=float(g.loc[g.step == 20000, "sigma_fc1"].iloc[0]),
                             peak_dK=g.delta_kmnist.max(), peak_dK_mi=g.delta_kmnist_mi.max(),
                             acc_end=g.acc.iloc[-1]))
e10 = pd.DataFrame(rows)
lines = []
say = lambda x="": (print(x), lines.append(str(x)))
say("E10 per arm (mean over seeds):")
say(e10.groupby(["n", "arm"])[["t03", "onset", "sigma_20k", "peak_dK", "peak_dK_mi", "acc_end"]].mean().round(4).to_string())

adam = e10[e10.arm.str.startswith("adam")].copy()
adam["t03_lr"] = adam.t03 * adam.lr
say("\nP1: t0.3 x lr by n (mean over seeds per lr), spread = max/min - 1:")
for n, g in adam.groupby("n"):
    m = g.groupby("lr").t03_lr.mean()
    spread = (m.max() - m.min()) / m.mean()
    say(f"  n={n}: {dict(m.round(3))}  max deviation from mean {np.max(np.abs(m / m.mean() - 1)):.2f}  -> {'holds' if np.max(np.abs(m / m.mean() - 1)) <= 0.30 else 'FAILS'} (±30%)")
say("\nP2: t0.3 at n=10000 vs n=500, per Adam lr:")
for lr, g in adam.groupby("lr"):
    a, b = g[g.n == 500].t03.mean(), g[g.n == 10000].t03.mean()
    say(f"  lr {lr:g}: {a:.0f} vs {b:.0f}  ratio {b / a:.2f} -> {'holds' if abs(b / a - 1) <= 0.25 else 'FAILS'} (±25%)")
sg = e10[e10.arm == "sgd (rho)"].groupby("n")
say("\nP3: SGD on rho, mean sigma_fc1 at 20k: " + str(sg.sigma_20k.mean().round(4).to_dict()) +
    "; t0.3: " + str(sg.t03.mean().to_dict()))
if len(sg) == 2:
    r = sg.sigma_20k.mean()[500] / sg.sigma_20k.mean()[10000]
    say(f"  ratio n=500 / n=10000 = {r:.2f} -> {'holds' if r >= 3 else ('fails (<3)' if r >= 2 else 'FALSIFIED (<2)')}")
say("\nP4: onset (first dK >= 0.04) x lr at n=10000:")
g = adam[adam.n == 10000].copy()
g["onset_lr"] = g.onset * g.lr
m = g.groupby("lr").onset_lr.mean()
say(f"  {dict(m.round(3))}  max deviation {np.max(np.abs(m / m.mean() - 1)):.2f} -> {'holds' if np.max(np.abs(m / m.mean() - 1)) <= 0.40 else 'FAILS'} (±40%)")
Path("results/E10").mkdir(parents=True, exist_ok=True)
e10.to_csv("results/E10/per_run.csv", index=False, float_format="%.5f")
Path("results/E10/summary.txt").write_text("\n".join(lines) + "\n")

# figure: sigma vs lr*t (Adam) and vs t (SGD)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "pdf.fonttype": 42})
colors = {"adam 5e-4": "#2a78d6", "adam 1e-3": "#eb6834", "adam 2e-3": "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
for (n, arm), c in curves.items():
    if arm in colors:
        axes[0].plot(c.index * arms[arm][1], c.values, color=colors[arm], lw=1.6,
                     ls="-" if n == 500 else "--", label=f"{arm}, n={n:,}")
    else:
        axes[1].plot(c.index / 1000, c.values, color="#52514e" if n == 500 else "#e87ba4", lw=1.6,
                     label=f"SGD on ρ, n={n:,}")
axes[0].set_xlabel("learning rate × steps")
axes[0].set_ylabel("mean posterior σ, layer 1")
axes[0].set_title("Adam: σ grows on an lr × t clock", fontsize=8, loc="left")
axes[0].legend(frameon=False, fontsize=6)
axes[1].set_xlabel("steps (×1000)")
axes[1].set_title("SGD on ρ: growth depends on n", fontsize=8, loc="left")
axes[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig("figures/E10_clock.pdf")

# ── E11 ───────────────────────────────────────────────────────────────────────
e11 = pd.concat([load("results/E9/dyn_n2000_rho-5_s?.csv", prior_sigma=1.0),
                 load("results/E11/dyn_n2000_rho-5_s*_sp0.1.csv"),
                 load("results/E11/dyn_n2000_rho-5_s*_sp0.3.csv")], ignore_index=True)
lines = []
r = []
for (sp, s), g in e11.groupby(["prior_sigma", "seed"]):
    end = g[g.step == g.step.max()].iloc[0]
    r.append(dict(prior_sigma=sp, seed=s, acc_end=end.acc, dK_end=end.delta_kmnist, dF_end=end.delta_fashion,
                  dK_mi_end=end.delta_kmnist_mi, peak_dK=g.delta_kmnist.max(), peak_dK_mi=g.delta_kmnist_mi.max(),
                  peak_dF_mi=g.delta_fashion_mi.max(), sigma_end=end.sigma_fc1, H_id_end=end.H_id))
e11s = pd.DataFrame(r)
say("\nE11 per prior scale (mean over seeds; end = 40k steps):")
say(e11s.groupby("prior_sigma").mean().drop(columns="seed").round(4).to_string())
m = e11s.groupby("prior_sigma").mean()
if 1.0 in m.index:
    for sp in (0.1, 0.3):
        if sp in m.index:
            say(f"  sigma_p {sp}: P1 acc gain {m.acc_end[sp] - m.acc_end[1.0]:+.4f} ({'holds' if m.acc_end[sp] - m.acc_end[1.0] >= 0.01 else 'fails'});"
                f" P2 dK_end gain {m.dK_end[sp] - m.dK_end[1.0]:+.4f} ({'holds' if m.dK_end[sp] - m.dK_end[1.0] >= 0.02 else 'fails'});"
                f" P3 peak MI-dK {m.peak_dK_mi[sp]:+.4f} ({'holds' if m.peak_dK_mi[sp] >= 0.02 else 'FAILS'})")
Path("results/E11").mkdir(parents=True, exist_ok=True)
e11s.to_csv("results/E11/per_run.csv", index=False, float_format="%.5f")
Path("results/E11/summary.txt").write_text("\n".join(lines) + "\n")
print("wrote results/E10/summary.txt, results/E11/summary.txt, figures/E10_clock.pdf")
