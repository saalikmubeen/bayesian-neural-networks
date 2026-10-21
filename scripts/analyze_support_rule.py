"""Model-free OOD rule: flag any image with ink on a pixel that is constant in every training image.

    PYTHONPATH=. python scripts/analyze_support_rule.py   -> results/E45/support_rule.csv

Score = total ink (above the constant background value) on the training subset's
constant pixels; values below 1e-3 count as zero (float noise).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bnnstudy import data

rows = []
for n in [500, 2000, 10000]:
    for seed in range(5):
        d = data.load(n, seed)
        xtr = d["train"][0].flatten(1).numpy()
        dead = xtr.max(0) <= xtr.min(0)
        bg = xtr.min()
        score = {k: np.clip(d[k][0].flatten(1).numpy()[:, dead] - bg, 0, None).sum(1) for k in ["test", "fashion", "kmnist"]}
        score = {k: np.where(v > 1e-3, v, 0.0) for k, v in score.items()}
        row = dict(n=n, seed=seed, dead_px=int(dead.sum()), mnist_flagged=float((score["test"] > 0).mean()))
        for o in ["fashion", "kmnist"]:
            y = np.r_[np.zeros(len(score["test"])), np.ones(len(score[o]))]
            row[f"auroc_{o}"] = roc_auc_score(y, np.r_[score["test"], score[o]])
            row[f"{o}_flagged"] = float((score[o] > 0).mean())
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("results/E45/support_rule.csv", index=False, float_format="%.4f")
print(df.groupby("n").mean().drop(columns="seed").round(4).to_string())
print("range over all runs: AUROC fashion", df.auroc_fashion.min().round(3), "-", df.auroc_fashion.max().round(3),
      "| kmnist", df.auroc_kmnist.min().round(3), "-", df.auroc_kmnist.max().round(3))
