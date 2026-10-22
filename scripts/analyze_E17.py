"""E17 mechanism test: per-image noise-to-signal ratio of BBB's hidden layers.

    PYTHONPATH=. python scripts/analyze_E17.py   -> results/E17/per_seed.csv, results/E17/summary.txt
Protocol: experiments/E17_mechanism/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from bnnstudy import data
from bnnstudy.metrics import auroc, entropy
from bnnstudy.models import BBB

OUT = Path("results/E17")
OUT.mkdir(parents=True, exist_ok=True)
BBB_RUN = {2000: "bbb_steps23550", 10000: "bbb"}
OODS = ["fashion", "kmnist"]


@torch.no_grad()
def noise_to_signal(sd, x):
    """R_1, R_2 per image: sum of noise variances / sum of squared mean pre-activations."""
    a = x.flatten(1)
    out = []
    for layer in ("fc1", "fc2"):
        mu, s = sd[f"{layer}.mu"], F.softplus(sd[f"{layer}.rho"])
        bmu, bs = sd[f"{layer}.bias_mu"], F.softplus(sd[f"{layer}.bias_rho"])
        m = a @ mu.T + bmu
        v = (a ** 2) @ (s ** 2).T + bs ** 2
        out.append(((v.sum(1)) / (m ** 2).sum(1).clamp_min(1e-12)).cpu().numpy())
        a = F.relu(m)
    return out


rows = []
for n, run in BBB_RUN.items():
    for seed in range(5):
        splits = {k: (x.cuda(), y) for k, (x, y) in data.load(n, seed).items() if k in ["test"] + OODS}
        sd = {k: v.cuda() for k, v in torch.load(f"results/E45/{run}_n{n}_s{seed}.pt").items()}
        model = BBB(mc_samples=50).cuda()
        model.load_state_dict(sd)
        torch.manual_seed(seed)
        R, MI = {}, {}
        for k, (x, _) in splits.items():
            R[k] = noise_to_signal(sd, x)
            ps = model.predict(x).cpu().numpy().astype(np.float64)
            MI[k] = entropy(ps.mean(0)) - entropy(ps).mean(0)
        for o in OODS:
            pooled_R2 = np.r_[R["test"][1], R[o][1]]
            pooled_MI = np.r_[MI["test"], MI[o]]
            rows.append(dict(n=n, seed=seed, ood=o,
                             auroc_R1=auroc(R["test"][0], R[o][0]), auroc_R2=auroc(R["test"][1], R[o][1]),
                             auroc_MI=auroc(MI["test"], MI[o]),
                             spearman_R2_MI=spearmanr(pooled_R2, pooled_MI)[0],
                             median_R2_id=float(np.median(R["test"][1])), median_R2_ood=float(np.median(R[o][1]))))
        print(f"n={n} s={seed} " + " | ".join(f"{r['ood']}: R1 {r['auroc_R1']:.3f} R2 {r['auroc_R2']:.3f} MI {r['auroc_MI']:.3f} rho {r['spearman_R2_MI']:.2f}"
                                              for r in rows[-2:]), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "per_seed.csv", index=False, float_format="%.5f")
m = df.groupby(["n", "ood"]).mean(numeric_only=True).drop(columns="seed")
lines = []
say = lambda s="": (print(s), lines.append(str(s)))
say(m.round(4).to_string())
for (n, o), r in m.iterrows():
    say(f"n={n} {o}: P1 AUROC(R2)={r.auroc_R2:.3f} ({'holds' if r.auroc_R2 >= 0.90 else 'fails'}; "
        f"falsified if < 0.75: {'YES' if r.auroc_R2 < 0.75 else 'no'}) | P3 R2 > R1: {r.auroc_R2:.3f} vs {r.auroc_R1:.3f} "
        f"({'holds' if r.auroc_R2 > r.auroc_R1 else 'fails'})")
for n in BBB_RUN:
    rho = m.loc[n].spearman_R2_MI.mean()
    say(f"n={n}: P2 mean Spearman(R2, MI) = {rho:.3f} ({'holds' if rho >= 0.6 else 'fails'})")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
