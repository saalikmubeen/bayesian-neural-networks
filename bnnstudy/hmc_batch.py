"""E20: batched full-batch HMC (several chains in one process) with diagonal mass adaptation.

    python -m bnnstudy.hmc_batch --n 500 --chains 8 --out results/E20

Same model, prior and data as bnnstudy.hmc (E19). Differences: all chains run
in one process with batched matrix products (the E19 sampler was limited by
Python overhead, not by the GPU); much longer trajectories (up to
--max-leapfrog steps); each chain has its own dual-averaging step size, and
with --adapt-mass its own diagonal inverse mass matrix from the sample variance
in expanding Stan-style windows (off by default: in the pilot it shrank the
step size, because short windows underestimate the many prior-scale
directions). The number of leapfrog steps is shared,
L = min(--max-leapfrog, ceil(--traj-len / median step size)). Step sizes are
jittered by +-10% while sampling. Writes <out>/hmcb_n<n>.pt (data seed 0; E20)
or <out>/hmcb_n<n>_d<seed>.pt (E20b) with samples of shape (chains, kept, dim)
and the per-chain trace.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .hmc import init_theta, shapes, warmup_windows


def forward_batch(theta, x, width):
    """theta: (C, dim), x: (n, 784) -> logits (C, n, 10)."""
    parts, o = [], 0
    for sh in shapes(width):
        k = int(np.prod(sh))
        parts.append(theta[:, o:o + k].reshape((theta.shape[0],) + sh))
        o += k
    w1, b1, w2, b2, w3, b3 = parts
    h = F.relu(torch.matmul(x, w1.transpose(1, 2)) + b1[:, None])
    h = F.relu(torch.matmul(h, w2.transpose(1, 2)) + b2[:, None])
    return torch.matmul(h, w3.transpose(1, 2)) + b3[:, None]


def potential_and_grad(theta, x, y, width, prior_sigma):
    """Per-chain potential (C,) and gradient (C, dim)."""
    theta = theta.detach().requires_grad_(True)
    logits = forward_batch(theta, x, width)
    nll = F.cross_entropy(logits.reshape(-1, 10), y.repeat(theta.shape[0]), reduction="none")
    u = nll.view(theta.shape[0], -1).sum(1) + 0.5 * (theta ** 2).sum(1) / prior_sigma ** 2
    (g,) = torch.autograd.grad(u.sum(), theta)
    return u.detach(), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--map-steps", type=int, default=3000)
    ap.add_argument("--eps0", type=float, default=1e-4)
    ap.add_argument("--traj-len", type=float, default=1.5)
    ap.add_argument("--max-leapfrog", type=int, default=10000)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--adapt-mass", action="store_true")
    ap.add_argument("--window-base", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"hmcb_n{a.n}.pt" if a.data_seed == 0 else f"hmcb_n{a.n}_d{a.data_seed}.pt")
    if path.exists():
        print("exists, skipping:", path)
        return
    x, y = data.load(a.n, a.data_seed)["train"]
    x, y = x.flatten(1).cuda(), y.cuda()
    C = a.chains
    torch.manual_seed(2000 + a.n + 100 * a.data_seed)

    # MAP initialisation, one random init per chain (seeds 1000 + chain, as in E19)
    theta = torch.stack([init_theta(a.width, torch.Generator().manual_seed(1000 + c)) for c in range(C)]).cuda()
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-3)
    for _ in range(a.map_steps):
        opt.zero_grad()
        logits = forward_batch(theta, x, a.width)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.repeat(C), reduction="sum") \
            + 0.5 * (theta ** 2).sum() / a.prior_sigma ** 2
        (loss / len(x)).backward()
        opt.step()
    theta = theta.detach()
    with torch.no_grad():
        acc_map = (forward_batch(theta, x, a.width).argmax(-1) == y).float().mean(1).cpu().numpy()
    print(f"n={a.n}: MAP train acc {np.round(acc_map, 4).tolist()}, dim {theta.shape[1]}", flush=True)

    ends = set(warmup_windows(a.warmup, first=a.window_base, last=a.window_base, base=a.window_base))
    if a.adapt_mass:
        print("adaptation windows end at", sorted(ends), flush=True)
    v = torch.ones_like(theta)
    u, g = potential_and_grad(theta, x, y, a.width, a.prior_sigma)
    win_sum, win_sq, win_k = torch.zeros_like(theta), torch.zeros_like(theta), 0

    def da_reset(eps):
        return dict(mu=torch.log(10 * eps), hbar=torch.zeros(C), log_eps=torch.log(eps),
                    log_eps_bar=torch.zeros(C), t=0)

    da = da_reset(torch.full((C,), a.eps0))
    eps = torch.full((C,), a.eps0)
    kept, trace, t0 = [], [], time.time()
    total = a.warmup + a.iters
    for it in range(total):
        if it < a.warmup:
            eps = da["log_eps"].exp()
            step = eps
        else:
            step = eps * (1 + 0.1 * (2 * torch.rand(C) - 1))
        n_steps = min(a.max_leapfrog, max(1, math.ceil(a.traj_len / float(step.median()))))
        e = step.cuda()[:, None]
        p0 = torch.randn_like(theta) / v.sqrt()
        p = p0 - 0.5 * e * g
        q = theta.clone()
        for l in range(n_steps):
            q = q + e * v * p
            u_new, g_new = potential_and_grad(q, x, y, a.width, a.prior_sigma)
            if l < n_steps - 1:
                p = p - e * g_new
        p = p - 0.5 * e * g_new
        log_acc = (u + 0.5 * (v * p0 ** 2).sum(1)) - (u_new + 0.5 * (v * p ** 2).sum(1))
        log_acc = torch.where(torch.isfinite(log_acc), log_acc, torch.full_like(log_acc, -float("inf")))
        acc_prob = log_acc.clamp(max=0).exp().cpu()
        accept = torch.rand(C) < acc_prob
        m = accept.cuda()
        theta = torch.where(m[:, None], q, theta)
        g = torch.where(m[:, None], g_new, g)
        u = torch.where(m, u_new, u)
        if it < a.warmup:
            da["t"] += 1
            t = da["t"]
            da["hbar"] = (1 - 1 / (t + 10)) * da["hbar"] + (a.target_accept - acc_prob) / (t + 10)
            da["log_eps"] = da["mu"] - math.sqrt(t) / 0.05 * da["hbar"]
            w = t ** -0.75
            da["log_eps_bar"] = w * da["log_eps"] + (1 - w) * da["log_eps_bar"]
            if a.adapt_mass and a.window_base <= it < max(ends):
                win_sum += theta
                win_sq += theta ** 2
                win_k += 1
            if a.adapt_mass and it + 1 in ends:
                mean = win_sum / win_k
                var = (win_sq / win_k - mean ** 2).clamp_min(0) * win_k / (win_k - 1)
                v = (win_k / (win_k + 5)) * var + 1e-3 * (5 / (win_k + 5))
                win_sum.zero_(), win_sq.zero_()
                win_k = 0
                da = da_reset(da["log_eps"].exp())
            if it == a.warmup - 1:
                eps = da["log_eps_bar"].exp()
        elif (it - a.warmup) % a.thin == a.thin - 1:
            kept.append(theta.cpu().clone())
        trace.append(dict(it=it, eps=step.numpy().copy(), n_leapfrog=n_steps, accept_prob=acc_prob.numpy().copy(),
                          accepted=accept.numpy().copy(), potential=u.cpu().numpy().copy()))
        if it % 10 == 0 or it == total - 1:
            recent = np.mean([t["accept_prob"] for t in trace[-10:]], 0)
            print(f"n={a.n} it {it:4d} L {n_steps} eps med {float(step.median()):.2e} acc(10) "
                  f"{np.round(recent, 2).tolist()} U med {float(u.median()):.0f} "
                  f"sqrt(v) med {float(v.sqrt().median()):.3f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(samples=torch.stack(kept, 1), trace=trace, width=a.width, n=a.n, chains=C,
                    data_seed=a.data_seed, prior_sigma=a.prior_sigma, adapt_mass=a.adapt_mass, inv_mass=v.cpu(), eps=eps,
                    traj_len=a.traj_len, max_leapfrog=a.max_leapfrog, target_accept=a.target_accept,
                    warmup=a.warmup, iters=a.iters, thin=a.thin, map_train_acc=acc_map), path)
    print("wrote", path, "samples", tuple(torch.stack(kept, 1).shape))


if __name__ == "__main__":
    main()
