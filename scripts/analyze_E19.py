"""E19: HMC reference vs BBB (both width 100, same N(0,1) prior and data subset).

    PYTHONPATH=. python scripts/analyze_E19.py   -> results/E19/summary.txt, results/E19/compare.csv
Protocol: experiments/E19_hmc/protocol.md
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bnnstudy import data
from bnnstudy import models as M
from bnnstudy.hmc import forward
from bnnstudy.softmatch import summarize

M.HIDDEN = 100
OUT = Path("results/E19")
OODS = ["fashion", "kmnist"]
lines = []
say = lambda s="": (print(s), lines.append(str(s)))


def split_rhat(q):
    """q: (chains, draws, items) -> split-R-hat per item."""
    c, d, _ = q.shape
    h = d // 2
    s = np.concatenate([q[:, :h], q[:, h:2 * h]], 0)
    m, n = s.shape[0], s.shape[1]
    w = s.var(1, ddof=1).mean(0)
    b = n * s.mean(1).var(0, ddof=1)
    var = (n - 1) / n * w + b / n
    return np.sqrt(var / np.maximum(w, 1e-12))


rows = []
for n in (2000, 500):
    files = sorted(glob.glob(str(OUT / f"hmc_n{n}_c*.pt")))
    if not files:
        continue
    chains = [torch.load(f) for f in files]
    width = chains[0]["width"]
    d = data.load(n, 0)
    splits = {k: (x.flatten(1).cuda(), y) for k, (x, y) in d.items() if k != "train"}
    ys = {k: y.numpy() for k, (_, y) in splits.items()}
    probs = {k: [] for k in splits}
    per_chain_q = []
    with torch.no_grad():
        for ch in chains:
            S = ch["samples"].cuda()
            for k, (x, _) in splits.items():
                probs[k].append(torch.stack([F.softmax(forward(th, x, width), -1) for th in S]).cpu().numpy())
            idx = np.arange(1000)
            pt = probs["test"][-1][:, idx, ys["test"][idx]]
            per_chain_q.append(pt)
    rhat = split_rhat(np.stack(per_chain_q))
    ps = {k: np.concatenate(v, 0).astype(np.float64) for k, v in probs.items()}
    dnn = np.load(OUT / f"dnn_n{n}_s0.npz")
    hmc = summarize(ps, ys, dnn, OODS)
    acc_rates = [np.mean([t["accept_prob"] for t in ch["trace"][ch["warmup"]:]]) for ch in chains]
    say(f"\nn={n}: {len(chains)} chains x {len(chains[0]['samples'])} samples; mean acceptance after warm-up "
        f"{np.round(acc_rates, 2).tolist()}; split-R-hat on p(true class), 1000 test images: median "
        f"{np.median(rhat):.3f}, 90th pct {np.quantile(rhat, 0.9):.3f} -> "
        f"{'CONVERGED (confirmatory)' if np.quantile(rhat, 0.9) < 1.1 else 'NOT converged (exploratory)'}")

    # overpruning: within-chain marginal mean and std per weight, hidden layers
    fr = []
    for ch in chains:
        S = ch["samples"]
        mean, std = S.mean(0), S.std(0)
        w1, _, w2, _, _, _ = [(mean_part, std_part) for mean_part, std_part in
                              zip(*[torch.split(t, [int(np.prod(sh)) for sh in
                                                    [(width, 784), (width,), (width, width), (width,), (10, width), (10,)]])
                                    for t in (mean, std)])]
        hid_mean = torch.cat([w1[0], w2[0]])
        hid_std = torch.cat([w1[1], w2[1]])
        fr.append(float((hid_mean.abs() > hid_std).float().mean()))
    hmc_frac = float(np.mean(fr))

    # BBB (width 100) on the same subset
    bbb_sd = torch.load(OUT / f"bbb_steps23550_n{n}_s0.pt")
    model = M.BBB(mc_samples=50).cuda()
    model.load_state_dict(bbb_sd)
    torch.manual_seed(0)
    with torch.no_grad():
        bps = {k: model.predict(x).cpu().numpy().astype(np.float64) for k, (x, _) in splits.items()}
    bbb = summarize(bps, ys, dnn, OODS)
    mu = torch.cat([bbb_sd["fc1.mu"].flatten(), bbb_sd["fc2.mu"].flatten()])
    sg = F.softplus(torch.cat([bbb_sd["fc1.rho"].flatten(), bbb_sd["fc2.rho"].flatten()]))
    bbb_frac = float((mu.abs() > sg).float().mean())
    dnn_acc = float((dnn["test_p"].argmax(-1) == dnn["test_y"]).mean())

    for name, r, frac in [("HMC", hmc, hmc_frac), ("BBB", bbb, bbb_frac)]:
        rows.append(dict(n=n, method=name, acc=r["acc"], ece=r["ece"], frac_signal=frac,
                         **{k: r[k] for k in r if k.startswith(("delta_", "auroc_", "fpr95_"))}))
    say(f"n={n} accuracy: HMC {hmc['acc']:.4f} | BBB {bbb['acc']:.4f} | DNN {dnn_acc:.4f}")
    for name, r in [("HMC", hmc), ("BBB", bbb)]:
        say(f"n={n} {name}: dF {r['delta_fashion']:+.4f} dK {r['delta_kmnist']:+.4f} | MI: dF {r['delta_fashion_mi']:+.4f} "
            f"dK {r['delta_kmnist_mi']:+.4f} | AUROC F/K {r['auroc_fashion_entropy']:.3f}/{r['auroc_kmnist_entropy']:.3f} "
            f"| FPR95 K {r['fpr95_kmnist_entropy']:.3f} (DNN_T {r['fpr95_kmnist_dnnT']:.3f})")
    p1 = ("exact posterior also has an advantage beyond softness" if hmc["delta_kmnist"] >= 0.03 and hmc["delta_fashion"] >= 0.03
          else "BBB's advantage is not a property of the exact posterior" if hmc["delta_kmnist"] <= 0.01 and hmc["delta_fashion"] <= 0.01
          else "inconclusive")
    say(f"n={n} P1 (decision rule): {p1}")
    say(f"n={n} P2: fraction |mean| > std in hidden layers: HMC {hmc_frac:.4f} vs BBB {bbb_frac:.4f} -> "
        f"{'holds (>= 2x)' if hmc_frac >= 2 * bbb_frac else ('fails' if hmc_frac >= bbb_frac else 'FALSIFIED (< BBB)')}")
    say(f"n={n} P3: HMC accuracy >= BBB: {hmc['acc']:.4f} vs {bbb['acc']:.4f} -> {'holds' if hmc['acc'] >= bbb['acc'] else 'fails'}")

pd.DataFrame(rows).to_csv(OUT / "compare.csv", index=False, float_format="%.5f")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
