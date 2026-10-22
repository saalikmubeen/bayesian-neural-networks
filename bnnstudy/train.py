"""Train one (method, train_size, seed) and save per-sample outputs on every split.

    python -m bnnstudy.train --method bbb --n 2000 --seed 0 --out results/E1_pilot

Writes <out>/<method>_n<n>_s<seed>.npz with, per split (val/test/ood sets):
  <split>_p      mean predictive probabilities (n, 10)
  <split>_ment   mean over samples of per-sample entropy (for mutual information)
  <split>_logits logits (deterministic models only, for temperature scaling)
  <split>_y      labels
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from . import data
from .metrics import entropy
from .models import BBB, DNN, Ensemble, LastLayerLaplace, train_bbb, train_dnn

BATCH = 64
REF_N = 2000  # epochs below are for n=2000; smaller n trains proportionally longer
METHODS = {
    "dnn":          dict(kind="dnn", epochs=40, lr=1e-3),
    "laplace":      dict(kind="laplace", epochs=40, lr=1e-3),
    # MAP under BBB's N(0,1) prior with BBB's budget (150 epochs, clipping)
    "map":          dict(kind="dnn", epochs=150, lr=1e-3, prior_sigma=1.0, clip=1.0),
    "mc_dropout":   dict(kind="dnn", epochs=40, lr=1e-3, p_drop=0.1, mc=50),
    "ensemble":     dict(kind="ensemble", epochs=40, lr=1e-3, members=5),
    "bbb":          dict(kind="bbb", epochs=150, lr=1e-3, kl="1/N"),
    "bbb_bug":      dict(kind="bbb", epochs=150, lr=1e-3, kl="1/n_batches"),
    "bbb_legacy":   dict(kind="bbb", epochs=120, lr=1e-4, kl="1/n_batches", legacy_init=True),
}
# BBB with a fixed number of optimizer steps instead of fixed epochs
# (4800 = default budget at n = 500 and 2000; 23550 = default budget at n = 10000)
for steps in [4800, 23550]:
    METHODS[f"bbb_steps{steps}"] = dict(kind="bbb", steps=steps, lr=1e-3, kl="1/N")
# deterministic baselines with BBB's long step budget (E12)
METHODS["dnn_steps23550"] = dict(kind="dnn", steps=23550, lr=1e-3)
METHODS["ensemble_steps23550"] = dict(kind="ensemble", steps=23550, lr=1e-3, members=5)
# BBB under an N(0, 0.3^2) prior, the HMC reference prior of E20
METHODS["bbb_steps23550_sp0.3"] = dict(kind="bbb", steps=23550, lr=1e-3, kl="1/N", prior_sigma=0.3)
for beta in [0.1, 0.3, 3, 10, 30]:  # KL weight beta/N (bbb is beta = 1)
    METHODS[f"bbb_beta{beta}"] = dict(kind="bbb", epochs=150, lr=1e-3, kl="1/N", beta=beta)


def build_and_train(cfg, splits, gen):
    x, y = splits["train"]
    n_batches = int(np.ceil(len(x) / BATCH))
    if "steps" in cfg:  # fixed step budget; warm-up keeps BBB's 20/150 fraction
        epochs = int(np.ceil(cfg["steps"] / n_batches))
        scale = epochs / 150
    else:
        scale = max(1, REF_N // len(x))
        epochs = cfg["epochs"] * scale
    if cfg["kind"] in ("dnn", "laplace"):
        m = DNN(cfg.get("p_drop", 0.0), cfg.get("mc", 1)).cuda()
        train_dnn(m, x, y, epochs, cfg["lr"], BATCH, gen,
                  prior_sigma=cfg.get("prior_sigma"), clip=cfg.get("clip"))
        if cfg["kind"] == "laplace":
            m = LastLayerLaplace(m).fit(x, *splits["val"])
    elif cfg["kind"] == "ensemble":
        members = []
        for _ in range(cfg["members"]):  # different init and batch order per member
            mem = DNN().cuda()
            train_dnn(mem, x, y, epochs, cfg["lr"], BATCH, gen)
            members.append(mem)
        m = Ensemble(members)
    else:
        m = BBB(legacy_init=cfg.get("legacy_init", False), prior_sigma=cfg.get("prior_sigma", 1.0)).cuda()
        kl_scale = cfg.get("beta", 1) / len(x) if cfg["kl"] == "1/N" else 1 / n_batches
        train_bbb(m, x, y, epochs, cfg["lr"], BATCH, gen, kl_scale,
                  anneal_epochs=max(1, round(20 * scale)))
    return m, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--on-support", action="store_true", help="also evaluate on-support OOD variants")
    ap.add_argument("--save-model", action="store_true", help="save the state_dict next to the outputs")
    ap.add_argument("--hidden", type=int, default=256, help="hidden-layer width (E19 uses 100)")
    a = ap.parse_args()
    from . import models as _models
    _models.HIDDEN = a.hidden

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.method}_n{a.n}_s{a.seed}.npz"
    if path.exists():
        print("exists, skipping:", path)
        return

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits = {k: (x.cuda(), y.cuda()) for k, (x, y) in data.load(a.n, a.seed, a.on_support).items()}
    gen = torch.Generator(device="cuda")
    gen.manual_seed(a.seed)

    t0 = time.time()
    cfg = METHODS[a.method]
    model, epochs = build_and_train(cfg, splits, gen)
    t_train = time.time() - t0

    arrays = {}
    for name, (x, y) in splits.items():
        if name == "train":
            continue
        ps = model.predict(x).cpu().numpy().astype(np.float64)
        arrays[f"{name}_p"] = ps.mean(0).astype(np.float32)
        arrays[f"{name}_ment"] = entropy(ps).mean(0).astype(np.float32)
        arrays[f"{name}_y"] = y.cpu().numpy()
        if cfg["kind"] == "dnn" and cfg.get("mc", 1) == 1:
            arrays[f"{name}_logits"] = model.logits(x).cpu().numpy()
    np.savez_compressed(path, **arrays)
    if a.save_model and isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path.with_suffix(".pt"))

    acc = (arrays["test_p"].argmax(-1) == arrays["test_y"]).mean()
    info = dict(method=a.method, n=a.n, seed=a.seed, config=cfg, epochs=epochs, batch=BATCH, hidden=a.hidden,
                prior_precision=getattr(model, "prior_precision", None),
                train_seconds=round(t_train, 1), test_acc=float(acc),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                host=platform.node(), date=time.strftime("%Y-%m-%d %H:%M"))
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"{a.method} n={a.n} s={a.seed}: test acc {acc:.4f} ({t_train:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
