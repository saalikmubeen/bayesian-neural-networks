"""E18: confirm the first-layer noise-to-signal account (R1) on fresh BBB models and new OOD sets.

    PYTHONPATH=. python scripts/analyze_E18.py   -> results/E18/per_seed.csv, results/E18/summary.txt
Protocol: experiments/E18_r1_confirm/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchvision import datasets

from bnnstudy import data
from bnnstudy.metrics import auroc, entropy
from bnnstudy.models import BBB

OUT = Path("results/E18")
BBB_RUN = {2000: "bbb_steps23550", 10000: "bbb"}
FAR = ["fashion", "kmnist", "cifar10_gray", "svhn_gray"]
SEEDS = range(5, 10)


def to_mnist_like(rgb_nchw_uint8):
    x = torch.as_tensor(rgb_nchw_uint8).float() / 255.0
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gray = F.interpolate(gray[:, None], size=(28, 28), mode="bilinear", align_corners=False)
    return (gray - data.MNIST_MEAN) / data.MNIST_STD


def extra_ood():
    c10 = datasets.CIFAR10(data.DATA_ROOT, train=False, download=False)
    sv = datasets.SVHN(data.DATA_ROOT, split="test", download=False)
    idx = np.random.RandomState(0).choice(len(sv.labels), 10000, replace=False)
    return {"cifar10_gray": to_mnist_like(c10.data.transpose(0, 3, 1, 2)),
            "svhn_gray": to_mnist_like(sv.data[idx])}


@torch.no_grad()
def r1(sd, x):
    a = x.flatten(1)
    mu, s = sd["fc1.mu"], F.softplus(sd["fc1.rho"])
    bmu, bs = sd["fc1.bias_mu"], F.softplus(sd["fc1.bias_rho"])
    m = a @ mu.T + bmu
    v = (a ** 2) @ (s ** 2).T + bs ** 2
    return (v.sum(1) / (m ** 2).sum(1).clamp_min(1e-12)).cpu().numpy()


EXTRA = extra_ood()
rows = []
for n, run in BBB_RUN.items():
    for seed in SEEDS:
        d = data.load(n, seed)
        sets = {"test": d["test"][0], "fashion": d["fashion"][0], "kmnist": d["kmnist"][0],
                "rotated": torch.rot90(d["test"][0], 1, dims=(2, 3)), **EXTRA}
        sd = {k: v.cuda() for k, v in torch.load(OUT / f"{run}_n{n}_s{seed}.pt").items()}
        model = BBB(mc_samples=50).cuda()
        model.load_state_dict(sd)
        torch.manual_seed(seed)
        R, MI = {}, {}
        for k, x in sets.items():
            x = x.cuda()
            R[k] = r1(sd, x)
            ps = model.predict(x).cpu().numpy().astype(np.float64)
            MI[k] = entropy(ps.mean(0)) - entropy(ps).mean(0)
        acc = float((model.predict(d["test"][0].cuda()).mean(0).argmax(-1).cpu() == d["test"][1]).float().mean())
        for o in FAR + ["rotated"]:
            rows.append(dict(n=n, seed=seed, ood=o, acc=acc,
                             auroc_R1=auroc(R["test"], R[o]), auroc_MI=auroc(MI["test"], MI[o]),
                             spearman_R1_MI=spearmanr(np.r_[R["test"], R[o]], np.r_[MI["test"], MI[o]])[0]))
        print(f"n={n} s={seed} acc {acc:.4f} " + " | ".join(f"{r['ood']}: R1 {r['auroc_R1']:.3f} MI {r['auroc_MI']:.3f}"
                                                         for r in rows[-5:]), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "per_seed.csv", index=False, float_format="%.5f")
m = df.groupby(["n", "ood"]).mean(numeric_only=True).drop(columns="seed")
lines = []
say = lambda s="": (print(s), lines.append(str(s)))
say(m.round(4).to_string())
fails_p1, fails_p3 = {}, {}
for n in BBB_RUN:
    fails_p1[n] = fails_p3[n] = 0
    for o in FAR:
        r = m.loc[(n, o)]
        p1, p2, p3 = r.auroc_R1 >= 0.90, abs(r.auroc_R1 - r.auroc_MI) <= 0.03, r.spearman_R1_MI >= 0.6
        fails_p1[n] += not p1
        fails_p3[n] += r.spearman_R1_MI < 0.4
        say(f"n={n} {o:13s}: P1 AUROC(R1) {r.auroc_R1:.3f} {'ok' if p1 else 'FAIL'} | P2 |R1-MI| "
            f"{abs(r.auroc_R1 - r.auroc_MI):.3f} {'ok' if p2 else 'FAIL'} | P3 rho {r.spearman_R1_MI:.2f} {'ok' if p3 else 'FAIL'}")
    r = m.loc[(n, "rotated")]
    say(f"n={n} rotated (near OOD): P4 |AUROC(R1) - AUROC(MI)| = |{r.auroc_R1:.3f} - {r.auroc_MI:.3f}| = "
        f"{abs(r.auroc_R1 - r.auroc_MI):.3f} {'ok' if abs(r.auroc_R1 - r.auroc_MI) <= 0.05 else 'FAIL'}")
falsified = any(fails_p1[n] >= 2 or fails_p3[n] >= 2 for n in BBB_RUN)
say(f"\nFalsification condition: {'MET' if falsified else 'not met'} (P1 failures {fails_p1}, P3 < 0.4 counts {fails_p3})")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
