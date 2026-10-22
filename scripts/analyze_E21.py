"""E21: core comparisons with Fashion-MNIST as in-distribution (OOD: MNIST, KMNIST).

    PYTHONPATH=. python -m bnnstudy.evaluate results/E21
    PYTHONPATH=. python scripts/analyze_E21.py   -> results/E21/E21_summary.txt, results/E21/E21_deltas.csv
Protocol: experiments/E21_fashion_id/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/E21")
OODS = ["mnist", "kmnist"]
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
            r[f"d_{o}"] = a.loc[seed, f"auroc_{o}_entropy"] - b.loc[seed, f"auroc_{o}_entropy"]
            r[f"dmi_{o}"] = a.loc[seed, f"auroc_{o}_mi"] - b.loc[seed, f"auroc_{o}_entropy"]
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "E21_deltas.csv", index=False, float_format="%.5f")
dnn_acc = df[df.method == "dnn"].groupby("n").acc.mean()

say("Mean over seeds (Delta vs softness-matched 40-epoch DNN; + = method better):")
for (n, m), g in d.groupby(["n", "method"]):
    say(f"n={n} {m:20s} seeds {len(g)} acc {g.acc.mean():.4f} (DNN {dnn_acc[n]:.4f}) | "
        + " | ".join(f"{o}: d {g[f'd_{o}'].mean():+.4f} ({int((g[f'd_{o}'] > 0).sum())}/{len(g)} >0) "
                     f"MI d {g[f'dmi_{o}'].mean():+.4f}" for o in OODS))


def get(n, m):
    return d[(d.n == n) & (d.method == m)].set_index("seed")


say()
std, long_ = get(2000, "bbb"), get(2000, "bbb_steps23550")
p1 = all(std[f"d_{o}"].mean() <= 0.02 for o in OODS)
means = ", ".join(f"{o} {std['d_' + o].mean():+.4f}" for o in OODS)
say(f"P1 (standard-budget bbb, mean d <= +0.02 on both): {'holds' if p1 else 'fails'} ({means})")
p2 = {o: long_[f"d_{o}"].mean() >= 0.03 and (long_[f"d_{o}"] > 0).sum() >= 4 for o in OODS}
say(f"P2 (bbb_steps23550 mean d >= +0.03 and >= 4/5 seeds > 0): "
    + ", ".join(f"{o} {'holds' if p2[o] else 'fails'} ({long_[f'd_{o}'].mean():+.4f}, {int((long_[f'd_{o}'] > 0).sum())}/5)"
                for o in OODS))
for base in ("dnn_steps23550", "ensemble_steps23550"):
    b = get(2000, base)
    seeds = long_.index.intersection(b.index)
    wins = {o: int((long_.loc[seeds, f"d_{o}"] > b.loc[seeds, f"d_{o}"]).sum()) for o in OODS}
    say(f"P3 vs {base}: BBB d higher in " + ", ".join(f"{o} {wins[o]}/{len(seeds)}" for o in OODS)
        + f" -> {'holds' if all(w >= 4 for w in wins.values()) else 'fails'} "
        + f"({base} mean d: " + ", ".join(f"{o} {b[f'd_{o}'].mean():+.4f}" for o in OODS) + ")")
big = get(10000, "bbb")
p4 = all(big[f"d_{o}"].mean() >= 0.03 for o in OODS)
means = ", ".join(f"{o} {big['d_' + o].mean():+.4f}" for o in OODS)
say(f"P4 (n=10000 bbb mean d >= +0.03 on both): {'holds' if p4 else 'fails'} ({means})")
p5 = all(long_[f"dmi_{o}"].mean() >= 0.03 for o in OODS)
means = ", ".join(f"{o} {long_['dmi_' + o].mean():+.4f}" for o in OODS)
say(f"P5 (bbb_steps23550 mean MI d >= +0.03 on both): {'holds' if p5 else 'fails'} ({means})")
say(f"Falsification (P2 fails on both OOD sets): {'MET' if not any(p2.values()) else 'not met'}")
(ROOT / "E21_summary.txt").write_text("\n".join(lines) + "\n")
