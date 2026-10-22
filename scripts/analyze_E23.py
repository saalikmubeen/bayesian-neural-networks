"""E23 part A: controls for the HMC comparison (prior N(0, 0.3^2), width 100, data seeds 0-2).

    PYTHONPATH=. python scripts/analyze_E23.py   -> results/E23/summary.txt, results/E23/controls.csv
Protocol: experiments/E23_hmc_controls_r1_baselines/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bnnstudy import data
from bnnstudy import models as M
from bnnstudy.metrics import auroc, entropy, fit_temperature_entropy
from bnnstudy.softmatch import summarize

M.HIDDEN = 100
OUT = Path("results/E23")
OODS = ["fashion", "kmnist"]
lines, rows = [], []
say = lambda s="": (print(s), lines.append(str(s)))


def delta_from_npz(z, dnn):
    """Entropy and MI Delta of a saved run (mean probs + mean per-sample entropy) vs the softened DNN."""
    H = {k: entropy(z[f"{k}_p"].astype(np.float64)) for k in ["val", "test"] + OODS}
    MI = {k: H[k] - z[f"{k}_ment"].astype(np.float64) for k in ["test"] + OODS}
    T = fit_temperature_entropy(dnn["val_logits"].astype(np.float64), H["val"].mean())
    from scipy.special import softmax
    Hd = {k: entropy(softmax(dnn[f"{k}_logits"].astype(np.float64) / T, -1)) for k in ["test"] + OODS}
    r = dict(acc=float((z["test_p"].argmax(-1) == z["test_y"]).mean()))
    for o in OODS:
        base = auroc(Hd["test"], Hd[o])
        r[f"delta_{o}"] = auroc(H["test"], H[o]) - base
        r[f"delta_{o}_mi"] = auroc(MI["test"], MI[o]) - base
    return r


hmc = pd.read_csv("results/E20/seeds.csv")
for n in (500, 2000):
    for seed in (0, 1, 2):
        dnn = np.load(f"results/E19/dnn_n{n}_s{seed}.npz")
        for m in ("map_sp0.3", "map_ensemble_sp0.3"):
            rows.append(dict(n=n, seed=seed, method=m, **delta_from_npz(np.load(OUT / f"{m}_n{n}_s{seed}.npz"), dnn)))
        # BBB with 800 posterior samples
        d = data.load(n, seed)
        splits = {k: (x.cuda(), y) for k, (x, y) in d.items() if k != "train"}
        model = M.BBB(mc_samples=800, prior_sigma=0.3).cuda()
        model.load_state_dict(torch.load(f"results/E20/bbb_steps23550_sp0.3_n{n}_s{seed}.pt"))
        torch.manual_seed(0)
        with torch.no_grad():
            ps = {k: model.predict(x).cpu().numpy().astype(np.float64) for k, (x, _) in splits.items()}
        r = summarize(ps, {k: y.numpy() for k, (_, y) in splits.items()}, dnn, OODS)
        rows.append(dict(n=n, seed=seed, method="bbb_800", **{k: r[k] for k in
                         ["acc", "delta_fashion", "delta_kmnist", "delta_fashion_mi", "delta_kmnist_mi"]}))
        rows.append(dict(n=n, seed=seed, method="dnn", acc=float((dnn["test_p"].argmax(-1) == dnn["test_y"]).mean())))
        for _, h in hmc[(hmc.n == n) & (hmc.seed == seed)].iterrows():
            rows.append(dict(n=n, seed=seed, method={"HMC": "hmc", "BBB": "bbb_50"}[h.method],
                             **{k: h[k] for k in ["acc", "delta_fashion", "delta_kmnist", "delta_fashion_mi", "delta_kmnist_mi"]}))
df = pd.DataFrame(rows)
df.to_csv(OUT / "controls.csv", index=False, float_format="%.5f")

cols = ["acc", "delta_fashion", "delta_kmnist", "delta_fashion_mi", "delta_kmnist_mi"]
for n in (500, 2000):
    say(f"\nn={n} (mean [range] over data seeds 0-2):")
    for m in ("dnn", "hmc", "bbb_50", "bbb_800", "map_sp0.3", "map_ensemble_sp0.3"):
        g = df[(df.n == n) & (df.method == m)]
        say(f"  {m:20s} " + " | ".join(f"{c} {g[c].mean():+.3f} [{g[c].min():+.3f},{g[c].max():+.3f}]"
                                         for c in cols if g[c].notna().any()))
    h = df[(df.n == n) & (df.method == "hmc")].set_index("seed")
    e = df[(df.n == n) & (df.method == "map_ensemble_sp0.3")].set_index("seed")
    p1 = all((h[f"delta_{o}"] > e[f"delta_{o}"]).all() for o in OODS)
    counts = ", ".join(f"{o}: {int((h['delta_' + o] > e['delta_' + o]).sum())}/3" for o in OODS)
    say(f"  P1 (HMC entropy dK, dF > MAP ensemble on every subset): {'holds' if p1 else 'fails'} ({counts})")
    mp = df[(df.n == n) & (df.method == "map_sp0.3")]
    say(f"  P2 (single MAP mean dK <= +0.02): {mp.delta_kmnist.mean():+.4f} -> {'holds' if mp.delta_kmnist.mean() <= 0.02 else 'fails'}")
    b50 = df[(df.n == n) & (df.method == "bbb_50")][cols[1:]].mean()
    b800 = df[(df.n == n) & (df.method == "bbb_800")][cols[1:]].mean()
    ch = {c: abs(b800[c] - b50[c]) for c in ["delta_kmnist", "delta_fashion", "delta_kmnist_mi"]}
    say(f"  P3 (BBB 800 vs 50 samples, |change| <= 0.01): " + ", ".join(f"{c} {v:.4f}" for c, v in ch.items())
        + f" -> {'holds' if max(ch.values()) <= 0.01 else 'fails'}")
fals = all(df[(df.n == n) & (df.method == "map_ensemble_sp0.3")][f"delta_{o}"].mean()
           >= df[(df.n == n) & (df.method == "hmc")][f"delta_{o}"].mean() for n in (500, 2000) for o in OODS)
say(f"\nFalsification (MAP ensemble mean >= HMC on both OOD sets at both n): {'MET' if fals else 'not met'}")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
