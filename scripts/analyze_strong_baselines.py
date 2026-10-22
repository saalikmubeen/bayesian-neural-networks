"""Stronger deterministic OOD scores vs longer-trained BBB (no training).

    PYTHONPATH=. python scripts/analyze_strong_baselines.py   -> results/E45/strong_baselines.csv

For each saved DNN (results/E45, 5 seeds) computes OOD AUROC with: predictive
entropy (T = 1), max softmax probability, max logit, energy (logsumexp of the
logits), and entropy at the best temperature chosen with oracle knowledge of the
OOD set (an upper bound for any temperature). Compares with BBB at 23,550 steps
(`bbb_steps23550` at n = 2000, `bbb` at n = 10000), entropy and MI scores.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

from bnnstudy.metrics import auroc, entropy

TEMPS = np.logspace(-1, 1.5, 26)
rows = []
for n, bbb in [(2000, "bbb_steps23550"), (10000, "bbb")]:
    for seed in range(5):
        d = np.load(f"results/E45/dnn_n{n}_s{seed}.npz")
        b = np.load(f"results/E45/{bbb}_n{n}_s{seed}.npz")
        for o in ["fashion", "kmnist"]:
            zi, zo = d["test_logits"].astype(float), d[f"{o}_logits"].astype(float)
            row = dict(n=n, seed=seed, ood=o)
            row["dnn_entropy"] = auroc(entropy(softmax(zi, -1)), entropy(softmax(zo, -1)))
            row["dnn_msp"] = auroc(-softmax(zi, -1).max(-1), -softmax(zo, -1).max(-1))
            row["dnn_maxlogit"] = auroc(-zi.max(-1), -zo.max(-1))
            row["dnn_energy"] = auroc(-logsumexp(zi, -1), -logsumexp(zo, -1))
            row["dnn_best_T_oracle"] = max(auroc(entropy(softmax(zi / t, -1)), entropy(softmax(zo / t, -1))) for t in TEMPS)
            pi, po = b["test_p"].astype(float), b[f"{o}_p"].astype(float)
            row["bbb_entropy"] = auroc(entropy(pi), entropy(po))
            row["bbb_mi"] = auroc(entropy(pi) - b["test_ment"], entropy(po) - b[f"{o}_ment"])
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("results/E45/strong_baselines.csv", index=False, float_format="%.4f")
cols = [c for c in df if c.startswith(("dnn", "bbb"))]
print(df.groupby(["n", "ood"])[cols].mean().round(3).to_string())
best_dnn = df[[c for c in cols if c.startswith("dnn")]].max(1)
print("\nBBB entropy minus best DNN score per run: min", (df.bbb_entropy - best_dnn).min().round(3),
      "mean", (df.bbb_entropy - best_dnn).mean().round(3), "| BBB wins in", int((df.bbb_entropy > best_dnn).sum()), "of", len(df))
