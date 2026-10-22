"""Metrics for every saved run, plus post-hoc temperature variants of the DNN.

    python -m bnnstudy.evaluate results/E1_pilot

Variants added per (n, seed) from the logits of each deterministic base
model (dnn, and map if present), using validation data only:
  <base>_Tnll            temperature minimising validation NLL
  <base>_Tmatch[<ref>]   temperature giving <base> the same mean validation
                         entropy as method <ref> (a softness-matched control)
Writes <dir>/metrics.csv (one row per run x variant) and <dir>/summary.txt.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import ALL_OOD
from .metrics import (auroc, classification, entropy, fit_temperature_entropy,
                      fit_temperature_nll, scores, selective)


BASES = ("dnn", "map")  # deterministic models that get temperature variants


def evaluate(d, ood_names):
    """d: dict split -> (p, ment or None, y). Returns one metrics row."""
    p, ment, y = d["test"]
    row = classification(p, y)
    s_id = scores(p, ment)
    correct = p.argmax(-1) == y
    row["aurc"], row["acc@80"] = selective(s_id["entropy"], correct)
    row["H_id"] = float(s_id["entropy"].mean())
    for o in ood_names:
        s_ood = scores(*d[o][:2])
        row[f"H_{o}"] = float(s_ood["entropy"].mean())
        for k in s_id:
            row[f"auroc_{o}_{k}"] = auroc(s_id[k], s_ood[k])
    return row


def load_run(f, splits):
    z = np.load(f)
    return z, {s: (z[f"{s}_p"].astype(np.float64), z[f"{s}_ment"].astype(np.float64), z[f"{s}_y"])
               for s in splits}


def main(root):
    root = Path(root)
    runs = {}
    for f in sorted(root.glob("*.npz")):
        m = re.match(r"(.+)_n(\d+)_s(\d+)\.npz", f.name)
        runs[(m[1], int(m[2]), int(m[3]))] = f
    keys = set(np.load(next(iter(runs.values()))).files)
    ood = [o for name in ALL_OOD for o in (name, f"{name}_onsupp") if f"{o}_p" in keys]
    splits = ["val", "test"] + ood

    rows = []
    for (method, n, seed), f in runs.items():
        z, d = load_run(f, splits)
        rows.append(dict(method=method, n=n, seed=seed, T=np.nan, **evaluate(d, ood)))
        if method not in BASES:
            continue
        logits = {s: z[f"{s}_logits"].astype(np.float64) for s in splits}
        y = {s: z[f"{s}_y"] for s in splits}

        def tempered(t):
            return {s: (softmax(logits[s] / t, -1), None, y[s]) for s in splits}

        t = fit_temperature_nll(logits["val"], y["val"])
        rows.append(dict(method=f"{method}_Tnll", n=n, seed=seed, T=t, **evaluate(tempered(t), ood)))
        for (ref, rn, rs), rf in runs.items():
            if ref in BASES or (rn, rs) != (n, seed):
                continue
            target = entropy(np.load(rf)["val_p"].astype(np.float64)).mean()
            t = fit_temperature_entropy(logits["val"], target)
            rows.append(dict(method=f"{method}_Tmatch[{ref}]", n=n, seed=seed, T=t,
                             **evaluate(tempered(t), ood)))

    df = pd.DataFrame(rows).sort_values(["n", "method", "seed"])
    df.to_csv(root / "metrics.csv", index=False, float_format="%.4f")

    cols = ["T", "acc", "nll", "ece", "aurc", "acc@80", "H_id"] + \
           [f"auroc_{o}_{k}" for o in ood for k in ("entropy", "msp", "mi")]
    cols = [c for c in cols if c in df]
    g = df.groupby(["n", "method"])[cols]
    summary = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    summary.insert(0, "seeds", g.size())
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        text = summary.to_string()
    (root / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
