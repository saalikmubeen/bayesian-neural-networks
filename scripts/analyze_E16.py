"""E16 swap test: BBB means vs deterministic means under BBB's posterior noise.

    PYTHONPATH=. python scripts/analyze_E16.py   -> results/E16/variants.csv, results/E16/summary.txt
Protocol: experiments/E16_swap/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bnnstudy import data
from bnnstudy.models import BBB
from bnnstudy.softmatch import summarize

OUT = Path("results/E16")
OUT.mkdir(parents=True, exist_ok=True)
BBB_RUN = {2000: "bbb_steps23550", 10000: "bbb"}
OODS = ["fashion", "kmnist"]
DNN_KEYS = {"fc1": "net.1", "fc2": "net.3", "fc3": "net.5"}


def variant_state(bbb_sd, dnn_sd, which, gen):
    sd = {k: v.clone() for k, v in bbb_sd.items()}
    if which == "B_swap":
        for layer, dk in DNN_KEYS.items():
            sd[f"{layer}.mu"] = dnn_sd[f"{dk}.weight"].clone()
            sd[f"{layer}.bias_mu"] = dnn_sd[f"{dk}.bias"].clone()
    elif which == "C_shuffled":
        for k in sd:
            if k.endswith("rho"):
                flat = sd[k].flatten()
                sd[k] = flat[torch.randperm(flat.numel(), generator=gen)].view_as(sd[k])
    return sd


rows = []
for n, run in BBB_RUN.items():
    for seed in range(5):
        splits = {k: (x.cuda(), y) for k, (x, y) in data.load(n, seed).items() if k != "train"}
        ys = {k: y.numpy() for k, (_, y) in splits.items()}
        dnn_npz = np.load(f"results/E45/dnn_n{n}_s{seed}.npz")
        bbb_sd = torch.load(f"results/E45/{run}_n{n}_s{seed}.pt", map_location="cpu")
        dnn_sd = torch.load(f"results/E12/dnn_steps23550_n{n}_s{seed}.pt", map_location="cpu")
        dnn_acc = float((np.load(f"results/E12/dnn_steps23550_n{n}_s{seed}.npz")["test_p"].argmax(-1)
                         == ys["test"]).mean())
        gen = torch.Generator().manual_seed(seed)
        for which in ["A_bbb", "B_swap", "C_shuffled"]:
            torch.manual_seed(seed)
            model = BBB(mc_samples=50).cuda()
            model.load_state_dict(variant_state(bbb_sd, dnn_sd, which, gen))
            with torch.no_grad():
                ps = {k: model.predict(x).cpu().numpy().astype(np.float64) for k, (x, _) in splits.items()}
            r = summarize(ps, ys, dnn_npz, OODS)
            rows.append(dict(n=n, seed=seed, variant=which, dnn_long_acc=dnn_acc, **r))
            print(f"n={n} s={seed} {which:11s} acc {r['acc']:.4f} dF {r['delta_fashion']:+.3f} dK {r['delta_kmnist']:+.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "variants.csv", index=False, float_format="%.5f")
lines = []
say = lambda s="": (print(s), lines.append(str(s)))
cols = ["acc", "dnn_long_acc", "delta_fashion", "delta_kmnist", "delta_fashion_mi", "delta_kmnist_mi",
        "auroc_fashion_entropy", "auroc_kmnist_entropy", "fpr95_kmnist_entropy"]
say(df.groupby(["n", "variant"])[cols].mean().round(4).to_string())
piv = df.pivot_table(index=["n", "seed"], columns="variant", values=["acc", "delta_fashion", "delta_kmnist"])
for n in BBB_RUN:
    g = piv.loc[n]
    acc_drop = (df[(df.n == n) & (df.variant == "B_swap")].set_index("seed").dnn_long_acc - g[("acc", "B_swap")])
    say(f"\nn={n}")
    say(f"P1: swap accuracy {g[('acc', 'B_swap')].round(4).tolist()} vs step-matched DNN; drop >= 0.05 in "
        f"{int((acc_drop >= 0.05).sum())}/5 seeds -> {'holds' if (acc_drop >= 0.05).sum() >= 4 else 'fails'}")
    for o in OODS:
        gap = (g[(f"delta_{o}", "A_bbb")] - g[(f"delta_{o}", "B_swap")]).mean()
        say(f"P2 {o}: Delta(A) - Delta(B) = {gap:+.4f} (need >= 0.05) -> {'holds' if gap >= 0.05 else 'fails'}")
    for o in OODS:
        a, c = g[(f"delta_{o}", "A_bbb")].mean(), g[(f"delta_{o}", "C_shuffled")].mean()
        frac = c / a if a != 0 else np.nan
        say(f"P3 {o}: Delta(C) / Delta(A) = {c:+.4f} / {a:+.4f} = {frac:.2f} (need >= 0.80)")
    dacc = (g[("acc", "C_shuffled")] - g[("acc", "A_bbb")]).mean()
    say(f"P3 accuracy: C - A = {dacc:+.4f} (need |d| <= 0.01) -> {'holds' if abs(dacc) <= 0.01 else 'fails'}")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
