"""E4 mechanism analysis: first-layer posterior sigma per input pixel, dead vs live.

    python scripts/analyze_E4.py      -> results/E45/sigma_by_pixel.csv, figures/E4_sigma_maps.pdf

Dead pixels = constant across the training subset of that run (bnnstudy.data.support_mask).
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bnnstudy import data

ROOT = Path("results/E45")
RUNS = [("bbb", 500), ("bbb_steps23550", 500), ("bbb", 2000), ("bbb_steps23550", 2000),
        ("bbb_steps4800", 10000), ("bbb", 10000)]
STEPS = {("bbb", 500): 4800, ("bbb", 2000): 4800, ("bbb", 10000): 23550,
         ("bbb_steps23550", 500): 23550, ("bbb_steps23550", 2000): 23550, ("bbb_steps4800", 10000): 4800}

rows, maps = [], {}
for method, n in RUNS:
    for f in sorted(ROOT.glob(f"{method}_n{n}_s*.pt")):
        seed = int(f.stem.split("_s")[-1])
        sd = torch.load(f, map_location="cpu")
        sigma = F.softplus(sd["fc1.rho"]).mean(0).view(28, 28).numpy()   # mean over hidden units
        mu_abs = sd["fc1.mu"].abs().mean(0).view(28, 28).numpy()
        live = data.support_mask(data.load(n, seed)["train"][0])[0].numpy()
        rows.append(dict(method=method, n=n, seed=seed, steps=STEPS[(method, n)],
                         dead_px=int((~live).sum()),
                         sigma_dead=float(sigma[~live].mean()), sigma_live=float(sigma[live].mean()),
                         ratio=float(sigma[~live].mean() / sigma[live].mean()),
                         mu_dead=float(mu_abs[~live].mean()), mu_live=float(mu_abs[live].mean())))
        if seed == 0:
            maps[(method, n)] = (sigma, live)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "sigma_by_pixel.csv", index=False, float_format="%.5f")
print(df.groupby(["n", "method", "steps"])[["dead_px", "sigma_dead", "sigma_live", "ratio", "mu_dead", "mu_live"]]
      .mean().round(4).to_string())

fig, axes = plt.subplots(1, len(RUNS), figsize=(1.6 * len(RUNS), 2.1))
vmax = max(m[0].max() for m in maps.values())
for ax, key in zip(axes, RUNS):
    if key not in maps:
        ax.axis("off")
        continue
    sigma, live = maps[key]
    im = ax.imshow(sigma, cmap="Blues", vmin=0, vmax=vmax)
    ax.contour(~live, levels=[0.5], colors="#eb6834", linewidths=0.6)
    ax.set_title(f"n={key[1]}\n{STEPS[key]} steps", fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="mean posterior σ (fc1)")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/E4_sigma_maps.pdf", bbox_inches="tight")
print("wrote figures/E4_sigma_maps.pdf")
