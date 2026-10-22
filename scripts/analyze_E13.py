"""Score E13 (SGD on rho, clipping on mu only; lr_rho fixed vs scaled with N).

    python scripts/analyze_E13.py   -> results/E13/summary.txt, results/E13/per_run.csv
Protocol: experiments/E13_sgd_clean/protocol.md
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

rows = []
for f in sorted(glob.glob("results/E13/dyn_*.csv")):
    g = pd.read_csv(f)
    tag = "sgd_scaled" if f.endswith("_sgd_scaled.csv") else "sgd_mu"
    hit = g[g.sigma_fc1 >= 0.3].step
    s20 = g.loc[g.step == 20000, "sigma_fc1"]
    div = g["diverged_step"].iloc[0] if "diverged_step" in g else np.nan
    rows.append(dict(n=int(g.n.iloc[0]), arm=tag, seed=int(g.seed.iloc[0]), rho_lr=g.rho_lr.iloc[0],
                     t03=int(hit.min()) if len(hit) else np.inf,
                     sigma_20k=float(s20.iloc[0]) if len(s20) else np.nan,
                     peak_dK=g.delta_kmnist.max(), peak_dF=g.delta_fashion.max(),
                     peak_dK_mi=g.delta_kmnist_mi.max(), last_step=int(g.step.max()),
                     acc_last=g.acc.iloc[-1], diverged_step=div))
df = pd.DataFrame(rows).sort_values(["n", "arm", "seed"])
Path("results/E13").mkdir(exist_ok=True)
df.to_csv("results/E13/per_run.csv", index=False, float_format="%.5f")

lines = []
say = lambda s="": (print(s), lines.append(str(s)))
say(df.to_string(index=False))
m = df.groupby(["n", "arm"]).mean(numeric_only=True)
say("\n" + m.round(4).to_string())
a, b = m.loc[(500, "sgd_mu"), "sigma_20k"], m.loc[(10000, "sgd_mu"), "sigma_20k"]
r = a / b
say(f"\nP1: sigma_fc1 at 20k, n=500 / n=10000 (lr_rho 0.05, no joint clipping) = {a:.4f} / {b:.4f} = {r:.1f}x -> "
    f"{'holds (>=3)' if r >= 3 else ('fails (<3)' if r >= 2 else 'FALSIFIED (<2)')}")
t500, t10k = m.loc[(500, "sgd_mu"), "t03"], m.loc[(10000, "sgd_scaled"), "t03"]
ratio = t10k / t500
say(f"P2: t0.3 n=10000 (lr_rho 1.0) / n=500 (lr_rho 0.05) = {t10k:.0f} / {t500:.0f} = {ratio:.2f} -> "
    f"{'holds (within +-25%)' if abs(ratio - 1) <= 0.25 else ('fails' if 0.5 <= ratio <= 2 else 'FALSIFIED (outside 0.5-2)')}")
pk = df[(df.n == 10000) & (df.arm == "sgd_scaled")].peak_dK
say(f"P3: peak entropy dK at n=10000 with scaled lr_rho: {pk.round(3).tolist()} mean {pk.mean():+.3f} -> "
    f"{'holds (>=0.04)' if pk.mean() >= 0.04 else 'fails'}")
say(f"Divergence: {df.dropna(subset=['diverged_step'])[['n', 'arm', 'seed', 'diverged_step']].to_dict('records')}")
Path("results/E13/summary.txt").write_text("\n".join(lines) + "\n")
