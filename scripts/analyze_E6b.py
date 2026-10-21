"""E6b: which layer's posterior noise drives converged BBB's OOD advantage?

    PYTHONPATH=. python scripts/analyze_E6b.py   -> results/E6b/layers.csv

Loads results/E45/bbb_steps23550_n2000_s*.pt and switches posterior noise off
(rho -> -30, sigma ~ 1e-13) in all layers except the kept ones.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bnnstudy import data
from bnnstudy.models import BBB
from bnnstudy.softmatch import summarize

ROOT, OUT = Path("results/E45"), Path("results/E6b")
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = {"all": ("fc1", "fc2", "fc3"), "keep_fc1": ("fc1",), "keep_fc2": ("fc2",),
            "keep_fc3": ("fc3",), "none": (),
            # exploratory follow-up (not pre-registered): drop one layer's noise
            "drop_fc1": ("fc2", "fc3"), "drop_fc2": ("fc1", "fc3"), "drop_fc3": ("fc1", "fc2")}
OODS = list(data.OOD_SETS)

rows = []
for f in sorted(ROOT.glob("bbb_steps23550_n2000_s*.pt")):
    seed = int(f.stem.split("_s")[-1])
    splits = {k: (x.cuda(), y) for k, (x, y) in data.load(2000, seed).items() if k != "train"}
    ys = {k: y.numpy() for k, (_, y) in splits.items()}
    dnn = np.load(ROOT / f"dnn_n2000_s{seed}.npz")
    for name, keep in VARIANTS.items():
        torch.manual_seed(seed)
        model = BBB(mc_samples=50).cuda()
        model.load_state_dict(torch.load(f))
        with torch.no_grad():
            for layer in ["fc1", "fc2", "fc3"]:
                if layer not in keep:
                    getattr(model, layer).rho.fill_(-30.0)
                    getattr(model, layer).bias_rho.fill_(-30.0)
            ps = {k: model.predict(x).cpu().numpy().astype(np.float64) for k, (x, _) in splits.items()}
        rows.append(dict(seed=seed, variant=name, **summarize(ps, ys, dnn, OODS)))
        r = rows[-1]
        print(f"s{seed} {name:9s} acc {r['acc']:.4f} ece {r['ece']:.3f} "
              f"dF {r['delta_fashion']:+.3f} dK {r['delta_kmnist']:+.3f} "
              f"dF_mi {r['delta_fashion_mi']:+.3f} dK_mi {r['delta_kmnist_mi']:+.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "layers.csv", index=False, float_format="%.5f")
cols = ["acc", "ece", "delta_fashion", "delta_kmnist", "delta_fashion_mi", "delta_kmnist_mi"]
summary = df.groupby("variant")[cols].mean()
full = summary.loc["all"]
for o in ["fashion", "kmnist"]:
    summary[f"retained_{o}"] = summary[f"delta_{o}"] / full[f"delta_{o}"]
print(summary.round(3).to_string())
(OUT / "summary.txt").write_text(summary.round(4).to_string() + "\n")
