"""E19: full-batch Hamiltonian Monte Carlo for the MNIST MLP (reference posterior).

    python -m bnnstudy.hmc --n 2000 --chain 0 --out results/E19

Model: MLP 784-W-W-10 (default W = 100), N(0, prior_sigma^2) prior on all weights
and biases, full-batch log-likelihood (sum over the training subset). Each chain
starts from a MAP estimate (full-batch Adam from its own random init), runs
`--warmup` iterations with step-size adaptation towards 0.75 acceptance, then
`--iters` iterations at a frozen step size, keeping every `--thin`-th sample.
Writes <out>/hmc_n<n>_c<chain>.pt with the kept parameter vectors (layout of
bnnstudy.models.DNN, flattened) and the sampler trace.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data


def shapes(width):
    return [(width, 784), (width,), (width, width), (width,), (10, width), (10,)]


def unflatten(theta, width):
    out, o = [], 0
    for sh in shapes(width):
        k = int(np.prod(sh))
        out.append(theta[..., o:o + k].reshape(theta.shape[:-1] + sh))
        o += k
    return out


def forward(theta, x, width):
    w1, b1, w2, b2, w3, b3 = unflatten(theta, width)
    h = F.relu(x @ w1.T + b1)
    h = F.relu(h @ w2.T + b2)
    return h @ w3.T + b3


def potential_and_grad(theta, x, y, width, prior_sigma):
    theta = theta.detach().requires_grad_(True)
    u = F.cross_entropy(forward(theta, x, width), y, reduction="sum") + 0.5 * (theta ** 2).sum() / prior_sigma ** 2
    (g,) = torch.autograd.grad(u, theta)
    return u.detach(), g


def init_theta(width, gen):
    parts = []
    for sh in shapes(width):
        fan_in = sh[1] if len(sh) == 2 else (784 if sh[0] == width and len(parts) == 0 else width)
        bound = 1 / math.sqrt(fan_in)
        parts.append((torch.rand(sh, generator=gen) * 2 - 1).mul(bound).flatten())
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--chain", type=int, required=True)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--leapfrog", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--map-steps", type=int, default=3000)
    ap.add_argument("--eps0", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hmc_n{a.n}_c{a.chain}.pt"
    if path.exists():
        print("exists, skipping:", path)
        return
    x, y = data.load(a.n, a.data_seed)["train"]
    x, y = x.flatten(1).cuda(), y.cuda()
    gen = torch.Generator().manual_seed(1000 + a.chain)
    torch.manual_seed(1000 + a.chain)

    # MAP initialisation
    theta = init_theta(a.width, gen).cuda().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-3)
    for _ in range(a.map_steps):
        opt.zero_grad()
        u = F.cross_entropy(forward(theta, x, a.width), y, reduction="sum") + 0.5 * (theta ** 2).sum() / a.prior_sigma ** 2
        (u / len(x)).backward()
        opt.step()
    theta = theta.detach()
    acc_map = float((forward(theta, x, a.width).argmax(-1) == y).float().mean())
    print(f"chain {a.chain} n={a.n}: MAP train acc {acc_map:.4f}, dim {theta.numel()}", flush=True)

    log_eps = math.log(a.eps0)
    u, g = potential_and_grad(theta, x, y, a.width, a.prior_sigma)
    kept, trace, t0 = [], [], time.time()
    total = a.warmup + a.iters
    for it in range(total):
        eps = math.exp(log_eps)
        p0 = torch.randn_like(theta)
        p = p0 - 0.5 * eps * g
        q = theta.clone()
        for l in range(a.leapfrog):
            q = q + eps * p
            u_new, g_new = potential_and_grad(q, x, y, a.width, a.prior_sigma)
            if l < a.leapfrog - 1:
                p = p - eps * g_new
        p = p - 0.5 * eps * g_new
        h0 = float(u + 0.5 * (p0 ** 2).sum())
        h1 = float(u_new + 0.5 * (p ** 2).sum())
        log_acc = h0 - h1
        acc_prob = 0.0 if not math.isfinite(log_acc) else min(1.0, math.exp(min(log_acc, 0.0)))
        accepted = torch.rand(1).item() < acc_prob
        if accepted:
            theta, u, g = q, u_new, g_new
        if it < a.warmup:  # Robbins-Monro step-size adaptation, frozen afterwards
            log_eps += (acc_prob - 0.75) / math.sqrt(it + 10)
        elif (it - a.warmup) % a.thin == a.thin - 1:
            kept.append(theta.detach().cpu().clone())
        trace.append(dict(it=it, eps=eps, accept_prob=acc_prob, accepted=accepted, potential=float(u)))
        if it % 100 == 0 or it == total - 1:
            recent = np.mean([t["accept_prob"] for t in trace[-100:]])
            print(f"chain {a.chain} n={a.n} it {it:5d} eps {eps:.2e} acc(100) {recent:.2f} U {float(u):.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(samples=torch.stack(kept), trace=trace, width=a.width, n=a.n, chain=a.chain,
                    data_seed=a.data_seed, prior_sigma=a.prior_sigma, leapfrog=a.leapfrog,
                    warmup=a.warmup, iters=a.iters, thin=a.thin, map_train_acc=acc_map), path)
    print("wrote", path, "samples", len(kept))


if __name__ == "__main__":
    main()
