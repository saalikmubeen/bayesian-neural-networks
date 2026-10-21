"""E7: CIFAR-10 replication. Small CNN; OOD = SVHN, CIFAR-100.

    python -m bnnstudy.cifar dnn      --n 10000 --seed 0 --out results/E7
    python -m bnnstudy.cifar ensemble --n 10000 --seed 0 --out results/E7
    python -m bnnstudy.cifar bbb      --n 10000 --seed 0 --out results/E7 --steps 60000

`dnn` / `ensemble` save per-sample outputs (.npz, logits for the DNN).
`bbb` trains Bayes by Backprop with a constant learning rate and, every --every
steps, logs accuracy, ECE, OOD AUROCs, delta vs the softness-matched DNN from
the same directory, validation-vs-proxy AUROCs, mean sigma per layer and the
running -ELBO (as bnnstudy.dynamics does for MNIST). No data augmentation, so
the ELBO stays a proper bound for the training set.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

from .data import batches, stratified_indices
from .metrics import auroc, entropy
from .models import BayesianLinear
from .softmatch import summarize

DATA_ROOT = "./data"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
N_VAL, VAL_SEED, N_OOD = 5000, 12345, 10000
BATCH = 128
OODS = ["svhn", "cifar100"]


def _norm(x_uint8_nhwc):
    x = torch.as_tensor(x_uint8_nhwc).float().div(255).permute(0, 3, 1, 2)
    return (x - MEAN) / STD


def load(n, seed):
    tr = datasets.CIFAR10(DATA_ROOT, train=True, download=True)
    te = datasets.CIFAR10(DATA_ROOT, train=False, download=True)
    y_all = np.array(tr.targets)
    val_idx = stratified_indices(y_all, N_VAL, np.random.RandomState(VAL_SEED))
    tr_idx = stratified_indices(y_all, n, np.random.RandomState(seed), exclude=val_idx)
    x_all = _norm(tr.data)
    y_all = torch.as_tensor(y_all)
    out = {"train": (x_all[tr_idx], y_all[tr_idx]), "val": (x_all[val_idx], y_all[val_idx]),
           "test": (_norm(te.data), torch.as_tensor(te.targets))}
    rng = np.random.RandomState(0)  # fixed OOD subsets
    sv = datasets.SVHN(DATA_ROOT, split="test", download=True)
    i = rng.choice(len(sv.labels), N_OOD, replace=False)
    out["svhn"] = (_norm(sv.data[i].transpose(0, 2, 3, 1)), torch.as_tensor(sv.labels[i]))
    c100 = datasets.CIFAR100(DATA_ROOT, train=False, download=True)
    out["cifar100"] = (_norm(c100.data), torch.as_tensor(c100.targets))
    return out


def make_proxies(x_val, seed):
    g = torch.Generator().manual_seed(10_000 + seed)
    lo, hi = x_val.min().item(), x_val.max().item()
    perm = torch.randperm(32 * 32, generator=g)
    flat = x_val.flatten(2)
    return {"proxy_noise": lo + (hi - lo) * torch.rand(x_val.shape, generator=g),
            "proxy_flip": x_val.flip(-2),                                   # upside down
            "proxy_permute": flat[:, :, perm].view_as(x_val)}               # same permutation per channel


# ── Models ─────────────────────────────────────────────────────────────────────


def cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, 10))


class BayesianConv2d(nn.Module):
    def __init__(self, i, o, k, rho_init=-5.0, prior_sigma=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(o, i, k, k))
        nn.init.kaiming_uniform_(self.mu, a=5 ** 0.5)
        self.rho = nn.Parameter(torch.full((o, i, k, k), rho_init))
        self.bias_mu = nn.Parameter(torch.zeros(o))
        self.bias_rho = nn.Parameter(torch.full((o,), rho_init))
        self.prior_sigma = prior_sigma

    def forward(self, x):
        w = self.mu + F.softplus(self.rho) * torch.randn_like(self.mu)
        b = self.bias_mu + F.softplus(self.bias_rho) * torch.randn_like(self.bias_mu)
        return F.conv2d(x, w, b, padding=1)

    kl = BayesianLinear.kl


class BayesCNN(nn.Module):
    LAYERS = ["c1", "c2", "c3", "f1", "f2"]

    def __init__(self, rho_init=-5.0):
        super().__init__()
        self.c1 = BayesianConv2d(3, 32, 3, rho_init)
        self.c2 = BayesianConv2d(32, 64, 3, rho_init)
        self.c3 = BayesianConv2d(64, 128, 3, rho_init)
        self.f1 = BayesianLinear(128 * 16, 256, rho_init=rho_init)
        self.f2 = BayesianLinear(256, 10, rho_init=rho_init)

    def forward(self, x):
        for c in (self.c1, self.c2, self.c3):
            x = F.max_pool2d(F.relu(c(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))

    def kl(self):
        return sum(getattr(self, l).kl() for l in self.LAYERS)


@torch.no_grad()
def sample_probs(model, x, samples, chunk=2500):
    model.eval()
    return torch.stack([torch.cat([F.softmax(model(x[i:i + chunk]), -1) for i in range(0, len(x), chunk)])
                        for _ in range(samples)])


def train_dnn(model, x, y, epochs, gen, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(x, y, BATCH, gen):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()


# ── Commands ───────────────────────────────────────────────────────────────────


def run_deterministic(a, splits, gen):
    members = 5 if a.method == "ensemble" else 1
    nets = []
    for _ in range(members):
        m = cnn().cuda()
        train_dnn(m, *splits["train"], a.epochs, gen)
        nets.append(m)
    arrays = {}
    for k, (x, y) in splits.items():
        if k == "train":
            continue
        ps = torch.cat([sample_probs(m, x, 1) for m in nets]).cpu().numpy().astype(np.float64)
        arrays[f"{k}_p"] = ps.mean(0).astype(np.float32)
        arrays[f"{k}_ment"] = entropy(ps).mean(0).astype(np.float32)
        arrays[f"{k}_y"] = y.cpu().numpy()
        if members == 1:
            with torch.no_grad():
                nets[0].eval()
                arrays[f"{k}_logits"] = torch.cat([nets[0](x[i:i + 2500]) for i in range(0, len(x), 2500)]).cpu().numpy()
    path = Path(a.out) / f"{a.method}_n{a.n}_s{a.seed}.npz"
    np.savez_compressed(path, **arrays)
    acc = (arrays["test_p"].argmax(-1) == arrays["test_y"]).mean()
    print(f"{a.method} n={a.n} s={a.seed}: test acc {acc:.4f} -> {path}")


def run_bbb(a, splits, gen):
    ys = {k: y.cpu().numpy() for k, (_, y) in splits.items()}
    dnn = np.load(Path(a.out) / f"dnn_n{a.n}_s{a.seed}.npz")
    x, y = splits["train"]
    model = BayesCNN(a.rho_init).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    warmup = a.warmup
    rows, step, ce_sum, kl_sum, cnt, t0 = [], 0, 0.0, 0.0, 0, time.time()
    while step < a.steps:
        model.train()
        for xb, yb in batches(x, y, BATCH, gen):
            w = min(1.0, (step + 1) / warmup) / len(x)
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            kl = model.kl()
            (ce + w * kl).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            ce_sum, kl_sum, cnt = ce_sum + ce.item(), kl_sum + kl.item() / len(x), cnt + 1
            if step % a.every == 0:
                ps = {k: sample_probs(model, xx, a.samples).cpu().numpy().astype(np.float64)
                      for k, (xx, _) in splits.items() if k != "train"}
                row = summarize({k: v for k, v in ps.items() if not k.startswith("proxy")}, ys, dnn, OODS)
                pv = ps["val"].mean(0)
                row["val_nll"] = float(-np.log(pv[np.arange(len(ys["val"])), ys["val"]] + 1e-12).mean())
                h_val = entropy(pv)
                mi_val = h_val - entropy(ps["val"]).mean(0)
                for k in [k for k in ps if k.startswith("proxy")]:
                    hp = entropy(ps[k].mean(0))
                    row[f"auroc_{k}_entropy"] = auroc(h_val, hp)
                    row[f"auroc_{k}_mi"] = auroc(mi_val, hp - entropy(ps[k]).mean(0))
                for l in BayesCNN.LAYERS:
                    row[f"sigma_{l}"] = float(F.softplus(getattr(model, l).rho).mean())
                row.update(step=step, ce=ce_sum / cnt, kl_per_n=kl_sum / cnt,
                           neg_elbo=(ce_sum + kl_sum) / cnt, seconds=round(time.time() - t0))
                rows.append(row)
                ce_sum, kl_sum, cnt = 0.0, 0.0, 0
                print(f"step {step:6d} acc {row['acc']:.4f} ece {row['ece']:.3f} "
                      f"dS {row['delta_svhn']:+.3f} dC {row['delta_cifar100']:+.3f} "
                      f"sf1 {row['sigma_f1']:.3f} -elbo {row['neg_elbo']:.3f} ({row['seconds']}s)", flush=True)
            if step >= a.steps:
                break
    path = Path(a.out) / f"dyn_n{a.n}_rho{a.rho_init:g}_s{a.seed}.csv"
    pd.DataFrame(rows).assign(n=a.n, seed=a.seed, rho_init=a.rho_init).to_csv(path, index=False, float_format="%.5f")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dnn", "ensemble", "bbb"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--rho-init", type=float, default=-5.0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits = {k: (x.cuda(), y.cuda()) for k, (x, y) in load(a.n, a.seed).items()}
    if a.method == "bbb":
        x_val, y_val = splits["val"]
        for k, xp in make_proxies(x_val.cpu(), a.seed).items():
            splits[k] = (xp.cuda(), y_val)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(a.seed)
    (run_bbb if a.method == "bbb" else run_deterministic)(a, splits, gen)


if __name__ == "__main__":
    main()
