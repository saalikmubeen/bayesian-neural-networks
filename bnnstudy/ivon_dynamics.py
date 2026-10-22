"""E14: training dynamics of IVON (Shen et al., ICML 2024) on the MNIST MLP.

    python -m bnnstudy.ivon_dynamics --n 2000 --seed 0 --lr 0.1 --hess-init 1.0 --out results/E14

Same evaluation as bnnstudy.dynamics (Δ vs the softness-matched DNN, proxies,
validation NLL), plus FPR95. Posterior: N(m, 1/(ess (h + wd))) with ess = N and
wd = 1/(N sigma_p^2), i.e. the same N(0, sigma_p^2) prior as BBB.
Writes <out>/ivon_n<n>_s<seed>[_<tag>].csv.
"""

import argparse
import time
from pathlib import Path

import ivon
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import data
from .data import batches
from .dynamics import make_proxies
from .metrics import auroc, entropy
from .models import DNN
from .softmatch import summarize

BATCH = 64
OODS = list(data.OOD_SETS)


@torch.no_grad()
def predict_all(model, opt, splits, samples):
    model.eval()
    out = {k: [] for k in splits if k != "train"}
    for _ in range(samples):
        with opt.sampled_params():
            for k in out:
                x = splits[k][0]
                out[k].append(torch.cat([F.softmax(model(x[i:i + 4096]), -1) for i in range(0, len(x), 4096)]))
    return {k: torch.stack(v).cpu().numpy().astype(np.float64) for k, v in out.items()}


def layer_sigmas(model, opt):
    """Mean posterior std per Linear layer (weights and bias together)."""
    group = opt.param_groups[0]
    std = 1.0 / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
    out, offset = {}, 0
    names = {id(p): n for n, p in model.named_parameters()}
    for p in group["params"]:
        layer = names[id(p)].rsplit(".", 1)[0]          # e.g. "net.1"
        out.setdefault(layer, []).append(std[offset:offset + p.numel()])
        offset += p.numel()
    keys = sorted(out, key=lambda k: int(k.split(".")[1]))
    return {f"sigma_fc{i + 1}": float(torch.cat(out[k]).mean()) for i, k in enumerate(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--hess-init", type=float, required=True)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="likelihood temperature k: ess = N/k, prior unchanged (E15)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--dnn-dir", default="results/E45")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ivon_n{a.n}_s{a.seed}{'_' + a.tag if a.tag else ''}.csv"
    if path.exists():
        print("exists, skipping:", path)
        return

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits = {k: (x.cuda(), y.cuda()) for k, (x, y) in data.load(a.n, a.seed).items()}
    x_val, y_val = splits["val"]
    for k, xp in make_proxies(x_val.cpu(), a.seed).items():
        splits[k] = (xp.cuda(), y_val)
    ys = {k: y.cpu().numpy() for k, (_, y) in splits.items()}
    dnn_file = Path(a.dnn_dir) / f"dnn_n{a.n}_s{a.seed}.npz"
    dnn = np.load(dnn_file) if dnn_file.exists() else None
    gen = torch.Generator(device="cuda")
    gen.manual_seed(a.seed)

    x, y = splits["train"]
    model = DNN().cuda()
    ess = len(x) / a.temp
    opt = ivon.IVON(model.parameters(), lr=a.lr, ess=ess, hess_init=a.hess_init,
                    weight_decay=1.0 / (ess * a.prior_sigma ** 2), beta1=0.9, beta2=0.99999, mc_samples=1)
    rows, step, ce_sum, cnt, t0, diverged = [], 0, 0.0, 0, time.time(), None
    while step < a.steps and diverged is None:
        model.train()
        for xb, yb in batches(x, y, BATCH, gen):
            with opt.sampled_params(train=True):
                opt.zero_grad()
                ce = F.cross_entropy(model(xb), yb)
                ce.backward()
            opt.step()
            step += 1
            ce_sum, cnt = ce_sum + ce.item(), cnt + 1
            if step % a.every == 0:
                ps = predict_all(model, opt, splits, a.samples)
                if not all(np.isfinite(v).all() for v in ps.values()):
                    print(f"non-finite predictions at step {step}; stopping this run", flush=True)
                    diverged = step
                    break
                pv = ps["val"].mean(0)
                row = dict(val_nll=float(-np.log(pv[np.arange(len(ys["val"])), ys["val"]] + 1e-12).mean()))
                if dnn is not None:
                    row.update(summarize({k: v for k, v in ps.items() if not k.startswith("proxy")}, ys, dnn, OODS))
                h_val = entropy(pv)
                mi_val = h_val - entropy(ps["val"]).mean(0)
                for k in [k for k in ps if k.startswith("proxy")]:
                    hp = entropy(ps[k].mean(0))
                    row[f"auroc_{k}_mi"] = auroc(mi_val, hp - entropy(ps[k]).mean(0))
                row.update(layer_sigmas(model, opt))
                row.update(step=step, ce=ce_sum / cnt, seconds=round(time.time() - t0))
                rows.append(row)
                ce_sum, cnt = 0.0, 0
                print(f"step {step:6d} acc {row.get('acc', float('nan')):.4f} val_nll {row['val_nll']:.4f} "
                      f"dK {row.get('delta_kmnist', float('nan')):+.3f} dF {row.get('delta_fashion', float('nan')):+.3f} "
                      f"s1 {row['sigma_fc1']:.4f}", flush=True)
            if step >= a.steps:
                break
    pd.DataFrame(rows).assign(n=a.n, seed=a.seed, lr=a.lr, hess_init=a.hess_init, prior_sigma=a.prior_sigma, temp=a.temp,
                              diverged_step=diverged).to_csv(path, index=False, float_format="%.5f")
    print("wrote", path)


if __name__ == "__main__":
    main()
