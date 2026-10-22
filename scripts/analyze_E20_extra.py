"""E20 follow-ups (exploratory, inference only): what the exact posterior looks like next to BBB.

    PYTHONPATH=. python scripts/analyze_E20_extra.py   -> results/E20/extra.txt, results/E20/extra.csv
1. per-layer spread: median within-chain std and |mean| of HMC weights vs BBB's sigma and |mu| (same prior);
2. per-chain deltas and a single posterior sample;
3. mean-field projection of the exact posterior: each weight drawn independently from a Gaussian with
   its within-chain mean and std (8 chains x 100 draws) -- do the marginals alone carry the OOD signal?
4. first-layer noise-to-signal R1 (as in E17/E18) from HMC moments and from BBB;
5. checks added after an independent code review: Delta on test images that both HMC and the DNN
   classify correctly (accuracy confound); equipartition E[theta . grad U] / dim = 1 at stationarity
   (sampler self-test); drift of the potential between halves of sampling with a batch-means standard
   error; range of leapfrog steps actually used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bnnstudy import data
from bnnstudy.hmc import shapes
from scipy.special import softmax

from bnnstudy.hmc_batch import forward_batch, potential_and_grad
from bnnstudy.metrics import auroc, entropy
from bnnstudy.softmatch import summarize

OUT = Path("results/E20")
OODS = ["fashion", "kmnist"]
W = 100
SIZES = [int(np.prod(sh)) for sh in shapes(W)]
NAMES = ["fc1", "fc1_b", "fc2", "fc2_b", "fc3", "fc3_b"]
lines, rows = [], []
say = lambda s="": (print(s), lines.append(str(s)))


@torch.no_grad()
def predict(S, x):
    """S: (draws, dim) on CPU -> probabilities (draws, n, 10) as float64 numpy."""
    return np.concatenate([F.softmax(forward_batch(S[j:j + 25].cuda(), x, W), -1).cpu().numpy()
                           for j in range(0, len(S), 25)]).astype(np.float64)


@torch.no_grad()
def r1(mean, var, x):
    """First-layer noise-to-signal ratio per image from per-weight means and variances (flat layout)."""
    w_m, b_m = mean[:SIZES[0]].view(W, 784), mean[SIZES[0]:SIZES[0] + W]
    w_v, b_v = var[:SIZES[0]].view(W, 784), var[SIZES[0]:SIZES[0] + W]
    m = x @ w_m.T + b_m
    v = (x ** 2) @ w_v.T + b_v
    return (v.sum(1) / (m ** 2).sum(1).clamp_min(1e-12)).cpu().numpy()


torch.manual_seed(0)
for n in (500, 2000):
    run = torch.load(OUT / f"hmcb_n{n}.pt")
    S = run["samples"]  # (chains, kept, dim)
    C, K, _ = S.shape
    d = data.load(n, 0)
    splits = {k: (x.flatten(1).cuda(), y) for k, (x, y) in d.items() if k != "train"}
    ys = {k: y.numpy() for k, (_, y) in splits.items()}
    dnn = np.load(f"results/E19/dnn_n{n}_s0.npz")
    sd = torch.load(OUT / f"bbb_steps23550_sp0.3_n{n}_s0.pt", map_location="cpu")

    # 1. spread per layer
    mean_c, std_c = S.mean(1), S.std(1)  # (chains, dim)
    say(f"\nn={n} per-layer medians (HMC: within-chain, averaged over chains | BBB seed 0) -- prior std 0.3")
    for name, a, b in zip(NAMES, np.cumsum([0] + SIZES[:-1]), np.cumsum(SIZES)):
        if name.endswith("_b"):
            continue
        layer = name
        hs, hm = float(std_c[:, a:b].median(1).values.mean()), float(mean_c[:, a:b].abs().median(1).values.mean())
        bs = float(F.softplus(sd[f"{layer}.rho"]).median())
        bm = float(sd[f"{layer}.mu"].abs().median())
        say(f"  {layer}: HMC std {hs:.3f}, |mean| {hm:.3f} | BBB sigma {bs:.3f}, |mu| {bm:.3f}")
        rows.append(dict(n=n, layer=layer, hmc_std=hs, hmc_absmean=hm, bbb_sigma=bs, bbb_absmu=bm))

    # 2. per-chain deltas and a single draw
    per = []
    for c in range(C):
        ps = {k: predict(S[c], x) for k, (x, _) in splits.items()}
        per.append(summarize(ps, ys, dnn, OODS))
    f = lambda key: f"{min(r[key] for r in per):+.3f} to {max(r[key] for r in per):+.3f}"
    say(f"n={n} single chains (100 draws each): acc {min(r['acc'] for r in per):.4f}-{max(r['acc'] for r in per):.4f}; "
        f"dF {f('delta_fashion')}; dK {f('delta_kmnist')}; MI dK {f('delta_kmnist_mi')}")
    one = summarize({k: predict(S[0, -1:], x) for k, (x, _) in splits.items()}, ys, dnn, OODS)
    say(f"n={n} one posterior draw: acc {one['acc']:.4f}, dF {one['delta_fashion']:+.3f}, dK {one['delta_kmnist']:+.3f}")

    # 3. mean-field projection: same per-weight means and stds, correlations removed
    mf = torch.cat([mean_c[c] + std_c[c] * torch.randn(K, mean_c.shape[1]) for c in range(C)])
    r = summarize({k: predict(mf, x) for k, (x, _) in splits.items()}, ys, dnn, OODS)
    say(f"n={n} mean-field projection of HMC (8 x {K} independent draws): acc {r['acc']:.4f}, "
        f"dF {r['delta_fashion']:+.3f}, dK {r['delta_kmnist']:+.3f}, MI dK {r['delta_kmnist_mi']:+.3f}")
    rows.append(dict(n=n, layer="mean-field projection", acc=r["acc"], delta_fashion=r["delta_fashion"],
                     delta_kmnist=r["delta_kmnist"], delta_kmnist_mi=r["delta_kmnist_mi"]))

    # 4. R1 from HMC moments (per chain, AUROC averaged) and from BBB
    xs = {k: splits[k][0] for k in ["test"] + OODS}
    for o in OODS:
        a_h = np.mean([auroc(r1(mean_c[c].cuda(), std_c[c].cuda() ** 2, xs["test"]),
                             r1(mean_c[c].cuda(), std_c[c].cuda() ** 2, xs[o])) for c in range(C)])
        mu = torch.cat([sd["fc1.mu"].flatten(), sd["fc1.bias_mu"]]).cuda()
        var = torch.cat([F.softplus(sd["fc1.rho"]).flatten(), F.softplus(sd["fc1.bias_rho"])]).cuda() ** 2
        a_b = auroc(r1(mu, var, xs["test"]), r1(mu, var, xs[o]))
        say(f"n={n} R1 AUROC {o}: from HMC moments {a_h:.3f} | from BBB {a_b:.3f}")
        rows.append(dict(n=n, layer=f"R1_{o}", hmc=a_h, bbb=a_b))

    # 5a. accuracy confound: in-distribution images correct for both HMC and the DNN
    ps = {k: np.concatenate([predict(S[c], x) for c in range(C)]) for k, (x, _) in splits.items()}
    full = summarize(ps, ys, dnn, OODS)
    H = {k: entropy(v.mean(0)) for k, v in ps.items()}
    Hd = {k: entropy(softmax(dnn[f"{k}_logits"].astype(np.float64) / full["T"], -1)) for k in ["test"] + OODS}
    both = (ps["test"].mean(0).argmax(-1) == ys["test"]) & (dnn["test_logits"].argmax(-1) == ys["test"])
    cc = {o: auroc(H["test"][both], H[o]) - auroc(Hd["test"][both], Hd[o]) for o in OODS}
    say(f"n={n} Delta on test images correct for both HMC and DNN ({both.mean():.3f} of test): "
        f"dF {cc['fashion']:+.3f}, dK {cc['kmnist']:+.3f} (all images: {full['delta_fashion']:+.3f}, {full['delta_kmnist']:+.3f})")
    rows.append(dict(n=n, layer="correct_only", delta_fashion=cc["fashion"], delta_kmnist=cc["kmnist"]))

    # 5b. equipartition: E[theta . grad U] = dim under the target
    x_tr, y_tr = d["train"]
    x_tr, y_tr = x_tr.flatten(1).cuda(), y_tr.cuda()
    eq = []
    for k in range(0, K, 10):
        th = S[:, k].cuda()
        _, g = potential_and_grad(th, x_tr, y_tr, W, run["prior_sigma"])
        eq.append(((th * g).sum(1) / th.shape[1]).cpu().numpy())
    eq = np.array(eq).mean(0)
    say(f"n={n} equipartition E[theta.gradU]/dim (should be 1): {eq.mean():.4f} (chains {eq.min():.4f}-{eq.max():.4f})")

    # 5c. drift of the potential, batch-means standard error (batches of 50 iterations)
    U = np.stack([t["potential"] for t in run["trace"]])[run["warmup"]:]  # (iters, chains)
    h = len(U) // 2
    se = lambda u: u.reshape(-1, 50, u.shape[1]).mean(1).std(0, ddof=1) / np.sqrt(len(u) // 50)
    diff = U[h:].mean() - U[:h].mean()
    se_diff = np.sqrt((se(U[:h]) ** 2 + se(U[h:]) ** 2).sum()) / U.shape[1]
    say(f"n={n} potential drift between halves {diff:+.1f} (batch-means s.e. {se_diff:.1f}, "
        f"{diff / se_diff:+.1f} s.e.; chains with positive drift {int((U[h:].mean(0) > U[:h].mean(0)).sum())}/{U.shape[1]})")
    # 5d. per-weight lag-1 autocorrelation of kept samples (hidden layers, 5,000 random weights per chain)
    rng = np.random.default_rng(0)
    idx = torch.as_tensor(rng.choice(SIZES[0] + SIZES[1] + SIZES[2], 5000, replace=False))
    ac = []
    for c in range(C):
        w = S[c][:, idx].double()
        w = w - w.mean(0)
        ac.append(float(((w[1:] * w[:-1]).sum(0) / (w ** 2).sum(0)).median()))
    say(f"n={n} median per-weight lag-1 autocorrelation of kept samples: {np.mean(ac):+.3f} "
        f"(chains {min(ac):+.3f} to {max(ac):+.3f})")
    Ls = [t["n_leapfrog"] for t in run["trace"][run["warmup"]:]]
    say(f"n={n} leapfrog steps per iteration while sampling: {min(Ls)}-{max(Ls)}")

pd.DataFrame(rows).to_csv(OUT / "extra.csv", index=False, float_format="%.5f")
(OUT / "extra.txt").write_text("\n".join(lines) + "\n")
