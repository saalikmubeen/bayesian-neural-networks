"""E12 controls: weight noise, step-matched DNN/ensemble, feature-space detectors.

    PYTHONPATH=. python scripts/analyze_E12.py [noise|features|matched|all]
      -> results/E12/{noise,features,matched}.csv and results/E12/summary.txt

BBB reference: results/E45 bbb_steps23550 (n = 2000) and bbb (n = 10000), 23,550 steps.
Protocol: experiments/E12_controls/protocol.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax

from bnnstudy import data
from bnnstudy.metrics import auroc, entropy, fit_temperature_entropy
from bnnstudy.models import DNN

OUT = Path("results/E12")
OUT.mkdir(parents=True, exist_ok=True)
BBB = {2000: "bbb_steps23550", 10000: "bbb"}
OODS = ["fashion", "kmnist"]
SEEDS = range(5)
GRID = np.logspace(-2, np.log10(3), 25)


def bbb_ref(n, seed):
    z = np.load(f"results/E45/{BBB[n]}_n{n}_s{seed}.npz")
    h = {k: entropy(z[f"{k}_p"].astype(float)) for k in ["val", "test"] + OODS}
    mi = {k: h[k] - z[f"{k}_ment"] for k in ["test"] + OODS}
    return h, mi


def load_dnn(path):
    m = DNN().cuda()
    m.load_state_dict(torch.load(path))
    m.eval()
    return m


@torch.no_grad()
def noisy_probs(model, x, scale, samples, gen):
    base = {k: v.clone() for k, v in model.state_dict().items()}
    out = []
    for _ in range(samples):
        for k, v in model.state_dict().items():
            std = base[k].std() if base[k].dim() > 1 else base[k].abs().mean() + 1e-8
            v.copy_(base[k] + scale * std * torch.randn(v.shape, device=v.device, generator=gen))
        out.append(torch.cat([F.softmax(model(x[i:i + 4096]), -1) for i in range(0, len(x), 4096)]))
    model.load_state_dict(base)
    return torch.stack(out).cpu().numpy().astype(np.float64)


def run_noise():
    rows = []
    for n in BBB:
        for seed in SEEDS:
            d = {k: (x.cuda(), y) for k, (x, y) in data.load(n, seed).items() if k != "train"}
            model = load_dnn(f"results/E45/dnn_n{n}_s{seed}.pt")
            gen = torch.Generator(device="cuda").manual_seed(seed)
            h_bbb, mi_bbb = bbb_ref(n, seed)
            target = h_bbb["val"].mean()
            val_h = [entropy(noisy_probs(model, d["val"][0], s, 10, gen).mean(0)).mean() for s in GRID]
            s_match = float(np.interp(target, val_h, GRID)) if val_h[-1] > target else float(GRID[-1])
            res = {}
            for label, s in [("matched", s_match)] + [(f"g{i}", g) for i, g in enumerate(GRID[::3])]:
                ps = {k: noisy_probs(model, d[k][0], s, 50 if label == "matched" else 20, gen) for k in ["test"] + OODS}
                hm = {k: entropy(v.mean(0)) for k, v in ps.items()}
                mim = {k: hm[k] - entropy(v).mean(0) for k, v in ps.items()}
                res[label] = {o: (auroc(hm["test"], hm[o]), auroc(mim["test"], mim[o])) for o in OODS}
            for o in OODS:
                best = max(max(r[o]) for r in res.values())
                rows.append(dict(n=n, seed=seed, ood=o, s_matched=s_match,
                                 noise_entropy=res["matched"][o][0], noise_mi=res["matched"][o][1],
                                 noise_best_oracle=best,
                                 bbb_entropy=auroc(h_bbb["test"], h_bbb[o]), bbb_mi=auroc(mi_bbb["test"], mi_bbb[o])))
            print(f"noise n={n} s={seed} s_match={s_match:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "noise.csv", index=False, float_format="%.4f")


@torch.no_grad()
def feats(model, x):
    body = model.net[:-1]
    return torch.cat([body(x[i:i + 4096]) for i in range(0, len(x), 4096)]).double()


def run_features():
    rows = []
    for n in BBB:
        for seed in SEEDS:
            dd = data.load(n, seed)
            d = {k: (x.cuda(), y) for k, (x, y) in dd.items()}
            h_bbb, mi_bbb = bbb_ref(n, seed)
            for which, path in [("dnn40ep", f"results/E45/dnn_n{n}_s{seed}.pt"),
                                ("dnn_steps23550", f"results/E12/dnn_steps23550_n{n}_s{seed}.pt")]:
                if not Path(path).exists():
                    continue
                model = load_dnn(path)
                ftr, ytr = feats(model, d["train"][0]), d["train"][1].cuda()
                mus = torch.stack([ftr[ytr == c].mean(0) for c in range(10)])
                cen = ftr - mus[ytr]
                prec = torch.linalg.inv(cen.T @ cen / len(ftr) + 1e-6 * torch.eye(ftr.shape[1], device=ftr.device, dtype=ftr.dtype))
                ntr = F.normalize(ftr, dim=1)

                def maha(f):
                    diff = f[:, None, :] - mus[None]
                    return torch.einsum("nkd,de,nke->nk", diff, prec, diff).min(1).values.cpu().numpy()

                def knn(f):
                    fn = F.normalize(f, dim=1)
                    out = []
                    for i in range(0, len(fn), 2048):
                        dist = torch.cdist(fn[i:i + 2048], ntr)
                        out.append(dist.kthvalue(10, dim=1).values)
                    return torch.cat(out).cpu().numpy()

                f = {k: feats(model, d[k][0]) for k in ["test"] + OODS}
                for o in OODS:
                    rows.append(dict(n=n, seed=seed, ood=o, model=which,
                                     mahalanobis=auroc(maha(f["test"]), maha(f[o])),
                                     knn=auroc(knn(f["test"]), knn(f[o])),
                                     bbb_entropy=auroc(h_bbb["test"], h_bbb[o]), bbb_mi=auroc(mi_bbb["test"], mi_bbb[o])))
            print(f"features n={n} s={seed}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "features.csv", index=False, float_format="%.4f")


def run_matched():
    rows = []
    for n in BBB:
        for seed in SEEDS:
            h_bbb, mi_bbb = bbb_ref(n, seed)
            dn = np.load(f"results/E12/dnn_steps23550_n{n}_s{seed}.npz")
            T = fit_temperature_entropy(dn["val_logits"].astype(float), h_bbb["val"].mean())
            hd = {k: entropy(softmax(dn[f"{k}_logits"].astype(float) / T, -1)) for k in ["test"] + OODS}
            en = np.load(f"results/E12/ensemble_steps23550_n{n}_s{seed}.npz")
            he = {k: entropy(en[f"{k}_p"].astype(float)) for k in ["test"] + OODS}
            mie = {k: he[k] - en[f"{k}_ment"] for k in ["test"] + OODS}
            for o in OODS:
                rows.append(dict(n=n, seed=seed, ood=o, T=T,
                                 dnn_long_matched=auroc(hd["test"], hd[o]),
                                 ens_long_entropy=auroc(he["test"], he[o]), ens_long_mi=auroc(mie["test"], mie[o]),
                                 bbb_entropy=auroc(h_bbb["test"], h_bbb[o]), bbb_mi=auroc(mi_bbb["test"], mi_bbb[o]),
                                 acc_dnn_long=float((dn["test_logits"].argmax(-1) == dn["test_y"]).mean()),
                                 acc_ens_long=float((en["test_p"].argmax(-1) == en["test_y"]).mean())))
    pd.DataFrame(rows).to_csv(OUT / "matched.csv", index=False, float_format="%.4f")


def ci(x, seed=0):
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return np.quantile(b, [0.025, 0.975])


def summarize():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    for name, col_pairs in [("noise.csv", [("bbb_entropy", "noise_entropy"), ("bbb_mi", "noise_mi"), ("bbb_entropy", "noise_best_oracle")]),
                            ("matched.csv", [("bbb_entropy", "dnn_long_matched"), ("bbb_entropy", "ens_long_entropy"), ("bbb_mi", "ens_long_mi")]),
                            ("features.csv", [("bbb_entropy", "mahalanobis"), ("bbb_entropy", "knn"), ("bbb_mi", "mahalanobis")])]:
        f = OUT / name
        if not f.exists():
            continue
        df = pd.read_csv(f)
        keys = ["n", "ood"] + (["model"] if "model" in df else [])
        say(f"\n{name}")
        say(df.groupby(keys).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
        for a, b in col_pairs:
            for k, g in df.groupby(keys):
                diff = (g[a] - g[b]).to_numpy()
                lo, hi = ci(diff)
                say(f"  {a} - {b} {k}: {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] wins {int((diff > 0).sum())}/{len(diff)}"
                    f"  (>= +0.02 in {int((diff >= 0.02).sum())})")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("noise", "all"):
        run_noise()
    if what in ("features", "all"):
        run_features()
    if what in ("matched", "all"):
        run_matched()
    summarize()
