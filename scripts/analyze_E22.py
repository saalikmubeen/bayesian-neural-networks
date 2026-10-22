"""E22: MNIST digits 0-4 as in-distribution; held-out digits 5-9 (near OOD), Fashion-MNIST, KMNIST.

    PYTHONPATH=. python -m bnnstudy.evaluate results/E22
    PYTHONPATH=. python scripts/analyze_E22.py   -> results/E22/E22_summary.txt, results/E22/E22_deltas.csv
Protocol: experiments/E22_heldout_digits/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bnnstudy import data
from bnnstudy.metrics import auroc

ROOT = Path("results/E22")
OODS = ["mnist59", "fashion", "kmnist"]
lines = []
say = lambda s="": (print(s), lines.append(str(s)))

df = pd.read_csv(ROOT / "metrics.csv")
rows = []
for (n, m), g in df[~df.method.str.contains("_T") & (df.method != "dnn")].groupby(["n", "method"]):
    a = g.set_index("seed")
    b = df[(df.n == n) & (df.method == f"dnn_Tmatch[{m}]")].set_index("seed")
    for seed in a.index.intersection(b.index):
        r = dict(n=n, method=m, seed=seed, acc=a.loc[seed, "acc"])
        for o in OODS:
            r[f"auroc_{o}"] = a.loc[seed, f"auroc_{o}_entropy"]
            r[f"d_{o}"] = a.loc[seed, f"auroc_{o}_entropy"] - b.loc[seed, f"auroc_{o}_entropy"]
            r[f"dmi_{o}"] = a.loc[seed, f"auroc_{o}_mi"] - b.loc[seed, f"auroc_{o}_entropy"]
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "E22_deltas.csv", index=False, float_format="%.5f")
dnn = df[df.method == "dnn"]
say(f"DNN: acc {dnn.acc.mean():.4f}, entropy AUROC "
    + ", ".join(f"{o} {dnn[f'auroc_{o}_entropy'].mean():.3f}" for o in OODS))
say("Mean over seeds (Delta vs softness-matched 40-epoch DNN; seeds with Delta > 0):")
for m, g in d.groupby("method"):
    say(f"{m:20s} acc {g.acc.mean():.4f} | " + " | ".join(
        f"{o}: AUROC {g[f'auroc_{o}'].mean():.3f} d {g[f'd_{o}'].mean():+.4f} ({int((g[f'd_{o}'] > 0).sum())}/{len(g)}) "
        f"MI d {g[f'dmi_{o}'].mean():+.4f}" for o in OODS))

get = lambda m: d[d.method == m].set_index("seed")
long_, std = get("bbb_steps23550"), get("bbb")
p1 = long_.d_mnist59.mean() >= 0.02 and (long_.d_mnist59 > 0).sum() >= 4
say(f"\nP1 (long BBB held-out d >= +0.02, >= 4/5 seeds): {'holds' if p1 else 'fails'} "
    f"({long_.d_mnist59.mean():+.4f}, {int((long_.d_mnist59 > 0).sum())}/5)")
far = (long_.d_fashion.mean() + long_.d_kmnist.mean()) / 2
say(f"P2 (held-out d < mean far-OOD d): {long_.d_mnist59.mean():+.4f} vs {far:+.4f} -> "
    f"{'holds' if long_.d_mnist59.mean() < far else 'fails'}")
for base in ("dnn_steps23550", "ensemble_steps23550"):
    b = get(base)
    s = long_.index.intersection(b.index)
    w = int((long_.loc[s, "d_mnist59"] > b.loc[s, "d_mnist59"]).sum())
    say(f"P3 vs {base}: BBB higher in {w}/{len(s)} -> {'holds' if w >= 4 else 'fails'} "
        f"({base} mean d {b.d_mnist59.mean():+.4f})")

# pixel rule: ink (above background) on any pixel that is constant across the training subset
data.ID = "mnist04"
pix = []
for seed in range(5):
    sp = data.load(2000, seed)
    x_tr = sp["train"][0].flatten(1)
    dead = x_tr.max(0).values == x_tr.min(0).values
    bg = x_tr.min(0).values
    score = lambda x: ((x.flatten(1)[:, dead] > bg[dead] + 1e-6).sum(1)).float().numpy()
    pix.append({o: auroc(score(sp["test"][0]), score(sp[o][0])) for o in OODS})
pix = pd.DataFrame(pix)
say("P4 (pixel rule AUROC on held-out digits <= 0.80): "
    f"{'holds' if pix.mnist59.mean() <= 0.80 else 'fails'} (mean {pix.mnist59.mean():.3f}, "
    f"range {pix.mnist59.min():.3f}-{pix.mnist59.max():.3f}; far OOD: fashion {pix.fashion.mean():.3f}, "
    f"kmnist {pix.kmnist.mean():.3f})")
say(f"P5 (standard BBB held-out d <= +0.02): {'holds' if std.d_mnist59.mean() <= 0.02 else 'fails'} "
    f"({std.d_mnist59.mean():+.4f})")
say(f"Falsification (P1 fails): {'MET' if not p1 else 'not met'}")
(ROOT / "E22_summary.txt").write_text("\n".join(lines) + "\n")
