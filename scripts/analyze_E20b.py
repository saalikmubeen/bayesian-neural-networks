"""E20b: E20's HMC vs BBB comparison over data seeds 0-2 (prior N(0, 0.3^2), width 100).

    PYTHONPATH=. python scripts/analyze_E20b.py   -> results/E20/seeds_summary.txt, results/E20/seeds.csv
Protocol: experiments/E20b_hmc_seeds/protocol.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bnnstudy import data
from bnnstudy import models as M
from bnnstudy.hmc_batch import forward_batch
from bnnstudy.softmatch import summarize

M.HIDDEN = 100
OUT = Path("results/E20")
OODS = ["fashion", "kmnist"]
KEYS = ["acc", "delta_fashion", "delta_kmnist", "delta_fashion_mi", "delta_kmnist_mi"]
lines, rows = [], []
say = lambda s="": (print(s), lines.append(str(s)))


def split_rhat(q):
    c, d, _ = q.shape
    h = d // 2
    s = np.concatenate([q[:, :h], q[:, h:2 * h]], 0)
    n = s.shape[1]
    w = s.var(1, ddof=1).mean(0)
    b = n * s.mean(1).var(0, ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / np.maximum(w, 1e-12))


for n in (500, 2000):
    for seed in (0, 1, 2):
        path = OUT / (f"hmcb_n{n}.pt" if seed == 0 else f"hmcb_n{n}_d{seed}.pt")
        if not path.exists():
            say(f"n={n} seed {seed}: missing {path}")
            continue
        run = torch.load(path)
        S = run["samples"]
        C, K, _ = S.shape
        d = data.load(n, seed)
        splits = {k: (x.flatten(1).cuda(), y) for k, (x, y) in d.items() if k != "train"}
        ys = {k: y.numpy() for k, (_, y) in splits.items()}
        with torch.no_grad():
            ps = {k: np.concatenate([F.softmax(forward_batch(S[c, j:j + 25].cuda(), x, 100), -1).cpu().numpy()
                                     for c in range(C) for j in range(0, K, 25)]).astype(np.float64)
                  for k, (x, _) in splits.items()}
        idx = np.arange(1000)
        rhat = split_rhat(ps["test"][:, idx, ys["test"][idx]].reshape(C, K, -1))
        conv = bool(np.quantile(rhat, 0.9) < 1.1)
        dnn = np.load(f"results/E19/dnn_n{n}_s{seed}.npz")
        hmc = summarize(ps, ys, dnn, OODS)
        sd = torch.load(OUT / f"bbb_steps23550_sp0.3_n{n}_s{seed}.pt")
        model = M.BBB(mc_samples=50, prior_sigma=0.3).cuda()
        model.load_state_dict(sd)
        torch.manual_seed(0)
        with torch.no_grad():
            bps = {k: model.predict(x).cpu().numpy().astype(np.float64) for k, (x, _) in splits.items()}
        bbb = summarize(bps, ys, dnn, OODS)
        for name, r in [("HMC", hmc), ("BBB", bbb)]:
            rows.append(dict(n=n, seed=seed, method=name, converged=conv, rhat90=float(np.quantile(rhat, 0.9)),
                             **{k: r[k] for k in KEYS}))
        say(f"n={n} seed {seed}: R-hat 90th pct {np.quantile(rhat, 0.9):.3f} max {rhat.max():.3f} "
            f"({'converged' if conv else 'NOT converged'}) | HMC acc {hmc['acc']:.4f} dF {hmc['delta_fashion']:+.4f} "
            f"dK {hmc['delta_kmnist']:+.4f} MI dK {hmc['delta_kmnist_mi']:+.4f} | BBB acc {bbb['acc']:.4f} "
            f"dF {bbb['delta_fashion']:+.4f} dK {bbb['delta_kmnist']:+.4f} MI dK {bbb['delta_kmnist_mi']:+.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "seeds.csv", index=False, float_format="%.5f")
new_q1_fail = 0
for n in (500, 2000):
    h = df[(df.n == n) & (df.method == "HMC") & df.converged].set_index("seed")
    b = df[(df.n == n) & (df.method == "BBB")].set_index("seed").loc[h.index]
    if len(h) == 0:
        continue
    say(f"\nn={n}: {len(h)} converged subsets {h.index.tolist()}")
    for k in KEYS:
        say(f"  {k}: HMC {h[k].mean():+.4f} (range {h[k].min():+.4f} to {h[k].max():+.4f}) | "
            f"BBB {b[k].mean():+.4f} (range {b[k].min():+.4f} to {b[k].max():+.4f})")
    q1 = (h.delta_kmnist >= 0.03) & (h.delta_fashion >= 0.03)
    new_q1_fail += int((~q1[[s for s in h.index if s != 0]]).sum())
    say(f"  Q1 (HMC dK, dF >= +0.03 on every subset): {'holds' if q1.all() else 'fails'} ({int(q1.sum())}/{len(q1)})")
    gap = (h.delta_kmnist_mi - b.delta_kmnist_mi).abs()
    say(f"  Q2 (mean |MI dK HMC - BBB| <= 0.03): {gap.mean():.4f} (per subset {np.round(gap.values, 4).tolist()}) -> "
        f"{'holds' if gap.mean() <= 0.03 else 'fails'}")
    q3 = h.acc >= b.acc
    say(f"  Q3 (HMC acc >= BBB on every subset): {'holds' if q3.all() else 'fails'} ({int(q3.sum())}/{len(q3)})")
    diff = h.delta_kmnist.mean() - b.delta_kmnist.mean()
    say(f"  Q4 (mean entropy dK HMC - BBB >= 0.02): {diff:+.4f} -> {'holds' if diff >= 0.02 else 'fails'}")
say(f"\nFalsification (Q1 fails on >= 2 of the 4 new runs): {'MET' if new_q1_fail >= 2 else 'not met'} "
    f"({new_q1_fail} new runs fail Q1)")
(OUT / "seeds_summary.txt").write_text("\n".join(lines) + "\n")
