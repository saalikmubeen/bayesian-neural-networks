"""Post-hoc analyses requested by an independent review (exploratory, no training).

    PYTHONPATH=. python scripts/analyze_posthoc_review.py   -> results/posthoc/summary.txt, results/posthoc/*.csv

1. Temperature sweep: entropy AUROC of the plain 40-epoch DNN over T in [0.25, 50] (log grid), its value
   at T = 1, at the softness-matched T, and the best T chosen with knowledge of the OOD set (oracle),
   against long-trained BBB's entropy AUROC. How much can softness alone buy?
2. Accuracy-controlled Delta: BBB's entropy AUROC minus the softness-matched DNN's, with the
   in-distribution test set restricted to images that both classify correctly.
Settings: MNIST as ID (results/E45: bbb_steps23550 at n = 500, 2000; bbb at n = 10000, i.e. 23,550 steps),
Fashion-MNIST as ID (results/E21: bbb_steps23550 at n = 2000; bbb at n = 10000).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from bnnstudy.metrics import auroc, entropy, fit_temperature_entropy

OUT = Path("results/posthoc")
OUT.mkdir(parents=True, exist_ok=True)
SETTINGS = [("MNIST", "results/E45", ["fashion", "kmnist"], {500: "bbb_steps23550", 2000: "bbb_steps23550", 10000: "bbb"}),
            ("Fashion-MNIST", "results/E21", ["mnist", "kmnist"], {2000: "bbb_steps23550", 10000: "bbb"})]
TS = np.exp(np.linspace(np.log(0.25), np.log(50), 40))
lines, rows = [], []
say = lambda s="": (print(s), lines.append(str(s)))

for id_name, root, oods, runs in SETTINGS:
    for n, bbb_name in runs.items():
        for seed in range(5):
            fd, fb = Path(root) / f"dnn_n{n}_s{seed}.npz", Path(root) / f"{bbb_name}_n{n}_s{seed}.npz"
            if not (fd.exists() and fb.exists()):
                continue
            z, b = np.load(fd), np.load(fb)
            logit = {k: z[f"{k}_logits"].astype(np.float64) for k in ["val", "test"] + oods}
            y = z["test_y"]
            Hb = {k: entropy(b[f"{k}_p"].astype(np.float64)) for k in ["val", "test"] + oods}
            T_match = fit_temperature_entropy(logit["val"], Hb["val"].mean())
            Hd = lambda t, k: entropy(softmax(logit[k] / t, -1))
            both = (b["test_p"].argmax(-1) == y) & (logit["test"].argmax(-1) == y)
            for o in oods:
                sweep = [auroc(Hd(t, "test"), Hd(t, o)) for t in TS]
                bbb_auc = auroc(Hb["test"], Hb[o])
                match_auc = auroc(Hd(T_match, "test"), Hd(T_match, o))
                rows.append(dict(id=id_name, n=n, seed=seed, ood=o, bbb=bbb_auc, dnn_T1=auroc(Hd(1.0, "test"), Hd(1.0, o)),
                                 dnn_Tmatch=match_auc, dnn_best=max(sweep), T_best=TS[int(np.argmax(sweep))],
                                 dnn_min=min(sweep), T_match=T_match,
                                 delta=bbb_auc - match_auc,
                                 delta_correct=auroc(Hb["test"][both], Hb[o]) - auroc(Hd(T_match, "test")[both], Hd(T_match, o)),
                                 frac_both=both.mean(),
                                 bbb_acc=(b["test_p"].argmax(-1) == y).mean(), dnn_acc=(logit["test"].argmax(-1) == y).mean()))
df = pd.DataFrame(rows)
df.to_csv(OUT / "softness_and_accuracy.csv", index=False, float_format="%.5f")
g = df.groupby(["id", "n", "ood"]).mean().drop(columns="seed")
say("Means over 5 seeds. AUROCs: long-trained BBB; DNN at T = 1, at the softness-matched T, worst and best T on a")
say("log grid 0.25-50 (best = oracle choice using the OOD set). Delta = BBB - DNN_Tmatch; Delta_correct on images both")
say("classify correctly.")
for (i, n, o), r in g.iterrows():
    say(f"{i:13s} n={n:5d} {o:7s} | BBB {r.bbb:.3f} | DNN T=1 {r.dnn_T1:.3f}, T_match {r.dnn_Tmatch:.3f} "
        f"(T {r.T_match:.2f}), range over T {r.dnn_min:.3f}-{r.dnn_best:.3f} (best T {r.T_best:.2f}) | "
        f"BBB - best-T DNN {r.bbb - r.dnn_best:+.3f} | Delta {r.delta:+.3f}, Delta_correct {r.delta_correct:+.3f} "
        f"(both correct {r.frac_both:.3f}; acc BBB {r.bbb_acc:.3f}, DNN {r.dnn_acc:.3f})")
wins = (df.bbb > df.dnn_best).groupby([df.id, df.n, df.ood]).sum()
say("\nSeeds where BBB beats the oracle-temperature DNN: " + ", ".join(f"{i} n={n} {o}: {int(w)}/5" for (i, n, o), w in wins.items()))
say(f"Spread of DNN AUROC over the whole T grid (max - min), mean over settings: {(df.dnn_best - df.dnn_min).mean():.3f}; "
    f"gain of the oracle T over T = 1: {(df.dnn_best - df.dnn_T1).mean():+.3f}")
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
