"""E23 part B: R1 against trivial input statistics as an account of BBB's per-image MI (E18 models).

    PYTHONPATH=. python scripts/analyze_E23_r1.py   -> results/E23/r1_baselines.txt, results/E23/r1_baselines.csv
Protocol: experiments/E23_hmc_controls_r1_baselines/protocol.md
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from bnnstudy import data
from bnnstudy.metrics import auroc, entropy
from bnnstudy.models import BBB

OUT = Path("results/E23")
OUT.mkdir(parents=True, exist_ok=True)
BBB_RUN = {2000: "bbb_steps23550", 10000: "bbb"}
FAR = ["fashion", "kmnist", "cifar10_gray", "svhn_gray"]

# reuse E18's OOD construction and R1 without running its analysis
src = open("scripts/analyze_E18.py").read()
ns = {}
exec(src[:src.index("EXTRA = extra_ood()")], ns)
EXTRA, r1 = ns["extra_ood"](), ns["r1"]


def baselines(x, x_train):
    """Input-only scores on images normalised with MNIST statistics; higher = more OOD-like."""
    raw = (x.flatten(1) * data.MNIST_STD + data.MNIST_MEAN).clamp(0, 1)
    tr = (x_train.flatten(1) * data.MNIST_STD + data.MNIST_MEAN).clamp(0, 1)
    dead = tr.max(0).values == tr.min(0).values
    return {"l1": raw.sum(1).numpy(), "ink_count": (raw > 0.5).sum(1).float().numpy(),
            "pixel_rule": (raw[:, dead] > tr.min(0).values[dead] + 1e-6).sum(1).float().numpy()}


rows = []
for n, run in BBB_RUN.items():
    for seed in range(5, 10):
        d = data.load(n, seed)
        sets = {"test": d["test"][0], "fashion": d["fashion"][0], "kmnist": d["kmnist"][0],
                "rotated": torch.rot90(d["test"][0], 1, dims=(2, 3)), **EXTRA}
        sd = {k: v.cuda() for k, v in torch.load(f"results/E18/{run}_n{n}_s{seed}.pt").items()}
        model = BBB(mc_samples=50).cuda()
        model.load_state_dict(sd)
        torch.manual_seed(seed)
        S, MI = {}, {}
        for k, x in sets.items():
            ps = model.predict(x.cuda()).cpu().numpy().astype(np.float64)
            MI[k] = entropy(ps.mean(0)) - entropy(ps).mean(0)
            S[k] = dict(R1=r1(sd, x.cuda()), **baselines(x, d["train"][0]))
        for o in FAR + ["rotated"]:
            r = dict(n=n, seed=seed, ood=o)
            for name in S["test"]:
                a, b = S["test"][name], S[o][name]
                au = auroc(a, b)
                r[f"auroc_{name}"] = max(au, 1 - au) if name != "R1" else au  # baselines: either direction
                r[f"spearman_{name}"] = abs(spearmanr(np.r_[a, b], np.r_[MI["test"], MI[o]])[0])
            rows.append(r)
        print(f"n={n} s={seed} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "r1_baselines.csv", index=False, float_format="%.5f")
m = df.groupby(["n", "ood"]).mean(numeric_only=True).drop(columns="seed")
lines = []
say = lambda s="": (print(s), lines.append(str(s)))
names = ["R1", "l1", "ink_count", "pixel_rule"]
say("Means over seeds 5-9. |Spearman| with BBB's per-image MI (test + OOD pooled); AUROC (baselines in their better direction).")
for (n, o), r in m.iterrows():
    say(f"n={n:5d} {o:12s} | " + " | ".join(f"{k}: rho {r[f'spearman_{k}']:.3f} AUROC {r[f'auroc_{k}']:.3f}" for k in names))
fail = {}
for n in BBB_RUN:
    for o in FAR:
        r = m.loc[(n, o)]
        fail[(n, o)] = max(r[f"spearman_{k}"] for k in names[1:]) >= r["spearman_R1"]
    p4 = not any(fail[(n, o)] for o in FAR)
    say(f"P4 n={n} (R1 rho > every baseline on all 4 far-OOD sets): {'holds' if p4 else 'fails'} "
        f"(baseline >= R1 on {[o for o in FAR if fail[(n, o)]]})")
    r = m.loc[(n, "rotated")]
    p5 = all(r["auroc_R1"] > r[f"auroc_{k}"] for k in names[1:])
    say(f"P5 n={n} (R1 AUROC > every baseline on rotated digits): {'holds' if p5 else 'fails'} "
        f"(R1 {r['auroc_R1']:.3f}; " + ", ".join(f"{k} {r[f'auroc_{k}']:.3f}" for k in names[1:]) + ")")
fals = any(sum(fail[(n, o)] for o in FAR) >= 2 for n in BBB_RUN)
say(f"Falsification (a baseline's rho >= R1's on >= 2 far-OOD sets at either n): {'MET' if fals else 'not met'}")
(OUT / "r1_baselines.txt").write_text("\n".join(lines) + "\n")
