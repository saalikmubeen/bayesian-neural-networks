"""Paired comparison of each uncertainty method with its softness-matched DNN.

    python -m bnnstudy.compare results/E2 [dnn|map]      (base model, default dnn)

For every method M and training size n, per seed:
    diff = metric(M) - metric(<base>_Tmatch[M])
reported as mean, 95% bootstrap CI over seeds, wins (M better / DNN better)
and a two-sided Wilcoxon signed-rank p-value. Writes <dir>/paired_<base>.csv and
<dir>/paired_<base>.txt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon



def bootstrap_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return np.quantile(boots, [0.025, 0.975])


def main(root, base="dnn"):
    root = Path(root)
    df = pd.read_csv(root / "metrics.csv")
    # (metric of M, metric of the matched DNN): entropy vs entropy, and M's MI vs DNN entropy
    ood = [c[len("auroc_"):-len("_entropy")] for c in df.columns
           if c.startswith("auroc_") and c.endswith("_entropy")]
    pairs = [(f"auroc_{o}_{s}", f"auroc_{o}_entropy") for o in ood for s in ("entropy", "mi")]
    pairs += [("acc", "acc"), ("nll", "nll"), ("ece", "ece"), ("aurc", "aurc")]

    rows = []
    methods = [m for m in df.method.unique() if "_T" not in m and m != base]
    for n in sorted(df.n.unique()):
        for m in methods:
            a = df[(df.n == n) & (df.method == m)].set_index("seed")
            b = df[(df.n == n) & (df.method == f"{base}_Tmatch[{m}]")].set_index("seed")
            seeds = a.index.intersection(b.index)
            if len(seeds) == 0:
                continue
            for ca, cb in pairs:
                d = (a.loc[seeds, ca] - b.loc[seeds, cb]).to_numpy()
                lo, hi = bootstrap_ci(d)
                p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
                rows.append(dict(n=n, method=m, metric=ca, vs=cb, seeds=len(d),
                                 mean_M=a.loc[seeds, ca].mean(), mean_dnnT=b.loc[seeds, cb].mean(),
                                 diff=d.mean(), ci_lo=lo, ci_hi=hi,
                                 M_higher=int((d > 0).sum()), dnnT_higher=int((d < 0).sum()),
                                 wilcoxon_p=p, T=b.loc[seeds, "T"].mean()))
    out = pd.DataFrame(rows)
    out.to_csv(root / f"paired_{base}.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        text = out.round(4).to_string(index=False)
    (root / f"paired_{base}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:3])
