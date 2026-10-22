"""E6 / E9: BBB training dynamics. Constant learning rate; evaluates every --every steps.

    python -m bnnstudy.dynamics --n 2000 --seed 0 --rho-init -5 --out results/E6

Writes <out>/dyn_n<n>_rho<r>_s<seed>.csv: one row per checkpoint with accuracy,
calibration, OOD AUROCs, delta vs the softness-matched DNN (results/E45 DNN run on
the same subset), mean sigma per layer, and running CE and KL/N.

With --proxies it also scores synthetic outliers built from the *validation*
images only (never test or real OOD data), for checkpoint selection (E9):
  noise     uniform noise over the pixel range
  flip      validation digits flipped left-right and up-down
  permute   validation digits with one fixed random pixel permutation
and logs validation NLL (for NLL-based early stopping).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from . import data
from .data import batches
from .models import BBB
from .metrics import auroc, entropy
from .softmatch import summarize

BATCH = 64
WARMUP_STEPS = 640  # 20 epochs at n = 2000, as in train.py
OODS = list(data.OOD_SETS)


def make_proxies(x_val, seed):
    g = torch.Generator().manual_seed(10_000 + seed)
    lo, hi = x_val.min().item(), x_val.max().item()
    perm = torch.randperm(784, generator=g)
    return {
        "proxy_noise": lo + (hi - lo) * torch.rand(x_val.shape, generator=g),
        "proxy_flip": x_val.flip(-1).flip(-2),
        "proxy_permute": x_val.flatten(1)[:, perm].view_as(x_val),
    }


@torch.no_grad()
def predict_all(model, splits, samples):
    model.mc_samples = samples
    return {k: model.predict(x).cpu().numpy().astype(np.float64)
            for k, (x, _) in splits.items() if k != "train"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rho-init", type=float, default=-5.0)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dnn-dir", default="results/E45")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proxies", action="store_true", help="score synthetic outliers (E9)")
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--rho-optimizer", choices=["adam", "sgd"], default="adam",
                    help="sgd: rho parameters use SGD+momentum 0.9 with --rho-lr; mu stays on Adam (E10)")
    ap.add_argument("--rho-lr", type=float, default=0.05)
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--clip", choices=["all", "mu"], default="all",
                    help="mu: clip only the mean parameters, leave rho unclipped (E13)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"dyn_n{a.n}_rho{a.rho_init:g}_s{a.seed}{'_' + a.tag if a.tag else ''}.csv"
    if path.exists():
        print("exists, skipping:", path)
        return

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits = {k: (x.cuda(), y.cuda()) for k, (x, y) in data.load(a.n, a.seed).items()}
    if a.proxies:
        x_val, y_val = splits["val"]
        for k, xp in make_proxies(x_val.cpu(), a.seed).items():
            splits[k] = (xp.cuda(), y_val)
    ys = {k: y.cpu().numpy() for k, (_, y) in splits.items()}
    dnn = np.load(Path(a.dnn_dir) / f"dnn_n{a.n}_s{a.seed}.npz")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(a.seed)

    x, y = splits["train"]
    model = BBB(rho_init=a.rho_init, prior_sigma=a.prior_sigma).cuda()
    if a.rho_optimizer == "sgd":
        rho = [p for name, p in model.named_parameters() if "rho" in name]
        mu = [p for name, p in model.named_parameters() if "rho" not in name]
        opts = [torch.optim.Adam(mu, lr=a.lr), torch.optim.SGD(rho, lr=a.rho_lr, momentum=0.9)]
    else:
        opts = [torch.optim.Adam(model.parameters(), lr=a.lr)]
    rows, step, ce_sum, kl_sum, cnt, t0 = [], 0, 0.0, 0.0, 0, time.time()
    diverged = None
    while step < a.steps and diverged is None:
        model.train()
        for xb, yb in batches(x, y, BATCH, gen):
            w = min(1.0, (step + 1) / WARMUP_STEPS) / len(x)
            for o in opts:
                o.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            kl = model.kl()
            (ce + w * kl).backward()
            clip_params = model.parameters() if a.clip == "all" else \
                [p for name, p in model.named_parameters() if "rho" not in name]
            nn.utils.clip_grad_norm_(clip_params, 1.0)
            for o in opts:
                o.step()
            step += 1
            ce_sum, kl_sum, cnt = ce_sum + ce.item(), kl_sum + kl.item() / len(x), cnt + 1
            if step % a.every == 0:
                ps = predict_all(model, splits, a.samples)
                if not all(np.isfinite(v).all() for v in ps.values()):
                    # diverged (e.g. unclipped SGD on rho): keep earlier checkpoints, record where it stopped
                    print(f"non-finite predictions at step {step}; stopping this run", flush=True)
                    diverged = step
                    break
                row = summarize({k: v for k, v in ps.items() if not k.startswith("proxy")},
                                ys, dnn, OODS)
                if a.proxies:
                    pv = ps["val"].mean(0)
                    row["val_nll"] = float(-np.log(pv[np.arange(len(ys["val"])), ys["val"]] + 1e-12).mean())
                    h_val = entropy(pv)
                    mi_val = h_val - entropy(ps["val"]).mean(0)
                    for k in [k for k in ps if k.startswith("proxy")]:
                        hp = entropy(ps[k].mean(0))
                        row[f"auroc_{k}_entropy"] = auroc(h_val, hp)
                        row[f"auroc_{k}_mi"] = auroc(mi_val, hp - entropy(ps[k]).mean(0))
                for name in ["fc1", "fc2", "fc3"]:
                    row[f"sigma_{name}"] = float(F.softplus(getattr(model, name).rho).mean())
                row.update(step=step, ce=ce_sum / cnt, kl_per_n=kl_sum / cnt,
                           neg_elbo=(ce_sum + kl_sum) / cnt, seconds=round(time.time() - t0))
                rows.append(row)
                ce_sum, kl_sum, cnt = 0.0, 0.0, 0
                print(f"step {step:6d} acc {row['acc']:.4f} ece {row['ece']:.3f} "
                      f"dK {row['delta_kmnist']:+.3f} dF {row['delta_fashion']:+.3f} "
                      f"s1 {row['sigma_fc1']:.3f} -elbo {row['neg_elbo']:.4f}", flush=True)
            if step >= a.steps:
                break
    pd.DataFrame(rows).assign(n=a.n, seed=a.seed, rho_init=a.rho_init, lr=a.lr, prior_sigma=a.prior_sigma,
                              rho_optimizer=a.rho_optimizer, rho_lr=a.rho_lr, clip=a.clip,
                              diverged_step=diverged).to_csv(
        path, index=False, float_format="%.5f")
    print("wrote", path)


if __name__ == "__main__":
    main()
