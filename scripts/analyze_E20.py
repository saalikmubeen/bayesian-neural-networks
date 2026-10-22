"""E20: HMC (prior N(0, 0.3^2), 8 batched chains) vs BBB under the same prior, width 100.

    PYTHONPATH=. python scripts/analyze_E20.py   -> results/E20/summary.txt, results/E20/compare.csv
Protocol: experiments/E20_hmc_sp03/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bnnstudy import data
from bnnstudy import models as M
from bnnstudy.hmc import shapes
from bnnstudy.hmc_batch import forward_batch
from bnnstudy.softmatch import summarize

M.HIDDEN = 100
OUT = Path("results/E20")
OODS = ["fashion", "kmnist"]
lines = []
say = lambda s="": (print(s), lines.append(str(s)))


def split_rhat(q):
    """q: (chains, draws, items) -> split-R-hat per item."""
    c, d, _ = q.shape
    h = d // 2
    s = np.concatenate([q[:, :h], q[:, h:2 * h]], 0)
    n = s.shape[1]
    w = s.var(1, ddof=1).mean(0)
    b = n * s.mean(1).var(0, ddof=1)
    var = (n - 1) / n * w + b / n
    return np.sqrt(var / np.maximum(w, 1e-12))


def hidden_frac(mean, std, width):
    sizes = [int(np.prod(sh)) for sh in shapes(width)]
    m, s = torch.split(mean, sizes), torch.split(std, sizes)
    hm, hs = torch.cat([m[0], m[2]]), torch.cat([s[0], s[2]])
    return float((hm.abs() > hs).float().mean())


def bbb_eval(n, seed, splits, ys, dnn):
    sd = torch.load(OUT / f"bbb_steps23550_sp0.3_n{n}_s{seed}.pt")
    model = M.BBB(mc_samples=50, prior_sigma=0.3).cuda()
    model.load_state_dict(sd)
    torch.manual_seed(0)
    with torch.no_grad():
        ps = {k: model.predict(x).cpu().numpy().astype(np.float64) for k, (x, _) in splits.items()}
    r = summarize(ps, ys, dnn, OODS)
    mu = torch.cat([sd["fc1.mu"].flatten(), sd["fc2.mu"].flatten()])
    sg = F.softplus(torch.cat([sd["fc1.rho"].flatten(), sd["fc2.rho"].flatten()]))
    return r, float((mu.abs() > sg).float().mean())


rows = []
for n in (500, 2000):
    path = OUT / f"hmcb_n{n}.pt"
    if not path.exists():
        continue
    run = torch.load(path)
    S, width = run["samples"], run["width"]  # (chains, kept, dim)
    C, K, _ = S.shape
    d = data.load(n, 0)
    splits = {k: (x.flatten(1).cuda(), y) for k, (x, y) in d.items() if k != "train"}
    ys = {k: y.numpy() for k, (_, y) in splits.items()}
    probs = {}
    with torch.no_grad():
        for k, (x, _) in splits.items():
            out = []
            for c in range(C):
                for j in range(0, K, 25):
                    out.append(F.softmax(forward_batch(S[c, j:j + 25].cuda(), x, width), -1).cpu())
            probs[k] = torch.cat(out).numpy()  # (C*K, n_k, 10), chain-major
    idx = np.arange(1000)
    q = probs["test"][:, idx, ys["test"][idx]].reshape(C, K, -1)
    rhat = split_rhat(q.astype(np.float64))
    U = np.stack([t["potential"] for t in run["trace"]])  # (iters, chains)
    samp = U[run["warmup"]:]
    h = len(samp) // 2
    acc_rate = np.stack([t["accept_prob"] for t in run["trace"][run["warmup"]:]]).mean(0)
    L = run["trace"][-1]["n_leapfrog"]
    conv = np.quantile(rhat, 0.9) < 1.1
    say(f"\nn={n}: {C} chains x {K} samples; L = {L} leapfrog steps; step size (frozen) "
        f"{np.round(run['eps'].numpy(), 6).tolist()}; acceptance {np.round(acc_rate, 2).tolist()}")
    say(f"n={n} split-R-hat on p(true class), 1000 test images: median {np.median(rhat):.3f}, "
        f"90th pct {np.quantile(rhat, 0.9):.3f}, max {rhat.max():.3f} -> "
        f"{'CONVERGED (confirmatory)' if conv else 'NOT converged (exploratory)'}")
    say(f"n={n} potential energy, mean over chains: first half of sampling {samp[:h].mean():.1f}, "
        f"second half {samp[h:].mean():.1f} (change {samp[h:].mean() - samp[:h].mean():+.1f}); "
        f"per chain spread {samp.mean(0).min():.0f}-{samp.mean(0).max():.0f}")

    ps = {k: v.astype(np.float64) for k, v in probs.items()}
    dnn = np.load(f"results/E19/dnn_n{n}_s0.npz")
    hmc = summarize(ps, ys, dnn, OODS)
    hmc_frac = float(np.mean([hidden_frac(S[c].mean(0), S[c].std(0), width) for c in range(C)]))
    bbb, bbb_frac = bbb_eval(n, 0, splits, ys, dnn)
    dnn_acc = float((dnn["test_p"].argmax(-1) == dnn["test_y"]).mean())
    for name, r, frac in [("HMC", hmc, hmc_frac), ("BBB", bbb, bbb_frac)]:
        rows.append(dict(n=n, method=name, seed=0, acc=r["acc"], ece=r["ece"], nll=r["nll"], frac_signal=frac,
                         **{k: r[k] for k in r if k.startswith(("delta_", "auroc_", "fpr95_"))}))
    say(f"n={n} accuracy: HMC {hmc['acc']:.4f} | BBB {bbb['acc']:.4f} | DNN {dnn_acc:.4f}; "
        f"NLL HMC {hmc['nll']:.3f} BBB {bbb['nll']:.3f}; ECE HMC {hmc['ece']:.3f} BBB {bbb['ece']:.3f}")
    for name, r in [("HMC", hmc), ("BBB", bbb)]:
        say(f"n={n} {name}: dF {r['delta_fashion']:+.4f} dK {r['delta_kmnist']:+.4f} | MI: dF {r['delta_fashion_mi']:+.4f} "
            f"dK {r['delta_kmnist_mi']:+.4f} | AUROC entropy F/K {r['auroc_fashion_entropy']:.3f}/"
            f"{r['auroc_kmnist_entropy']:.3f} MI F/K {r['auroc_fashion_mi']:.3f}/{r['auroc_kmnist_mi']:.3f} "
            f"| FPR95 K {r['fpr95_kmnist_entropy']:.3f} (DNN_T {r['fpr95_kmnist_dnnT']:.3f})")
    for seed in (1, 2):  # context only: different subsets, no HMC on them
        if (OUT / f"bbb_steps23550_sp0.3_n{n}_s{seed}.pt").exists():
            ds = data.load(n, seed)
            sp = {k: (x.flatten(1).cuda(), y) for k, (x, y) in ds.items() if k != "train"}
            r, frac = bbb_eval(n, seed, sp, {k: y.numpy() for k, (_, y) in sp.items()},
                               np.load(f"results/E19/dnn_n{n}_s{seed}.npz"))
            rows.append(dict(n=n, method="BBB", seed=seed, acc=r["acc"], ece=r["ece"], nll=r["nll"], frac_signal=frac,
                             **{k: r[k] for k in r if k.startswith(("delta_", "auroc_", "fpr95_"))}))
            say(f"n={n} BBB seed {seed} (context): acc {r['acc']:.4f} dF {r['delta_fashion']:+.4f} "
                f"dK {r['delta_kmnist']:+.4f} MI dK {r['delta_kmnist_mi']:+.4f}")

    say(f"n={n} precondition (BBB entropy dK >= +0.03): {'met' if bbb['delta_kmnist'] >= 0.03 else 'NOT met'}")
    p1 = ("exact posterior also has an OOD advantage beyond softness"
          if hmc["delta_kmnist"] >= 0.03 and hmc["delta_fashion"] >= 0.03
          else "BBB's advantage is not a property of the exact posterior"
          if hmc["delta_kmnist"] <= 0.01 and hmc["delta_fashion"] <= 0.01 else "inconclusive")
    say(f"n={n} P1 (decision rule): {p1}")
    say(f"n={n} P2: fraction |mean| > std in hidden layers: HMC {hmc_frac:.4f} vs BBB {bbb_frac:.4f} -> "
        f"{'holds (>= 2x)' if hmc_frac >= 2 * bbb_frac else ('fails' if hmc_frac >= bbb_frac else 'FALSIFIED (< BBB)')}")
    say(f"n={n} P3: HMC accuracy >= BBB: {hmc['acc']:.4f} vs {bbb['acc']:.4f} -> "
        f"{'holds' if hmc['acc'] >= bbb['acc'] else 'fails'}")
    gap = abs(hmc["delta_kmnist_mi"] - bbb["delta_kmnist_mi"])
    say(f"n={n} P4: |MI dK HMC - BBB| = {gap:.4f} -> {'holds' if gap <= 0.03 else 'fails'}")

pd.DataFrame(rows).to_csv(OUT / "compare.csv", index=False, float_format="%.5f")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
