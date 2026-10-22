"""Compare a sampled model's OOD scores with a DNN softened to its validation entropy.

Used by analyses that create model variants on the fly (dynamics, layer ablations),
where the saved-run pipeline (train.py -> evaluate.py) does not apply."""

import numpy as np
from scipy.special import softmax

from .metrics import auroc, classification, entropy, fit_temperature_entropy, fpr95


def summarize(ps, ys, dnn_npz, oods):
    """ps: split -> per-sample probabilities (S, n, 10) as numpy; ys: split -> labels.
    dnn_npz: a saved DNN run (with logits) on the same training subset.
    Returns a flat dict of metrics for the model and its softness-matched DNN."""
    mean = {k: v.mean(0) for k, v in ps.items()}
    H = {k: entropy(v) for k, v in mean.items()}
    MI = {k: H[k] - entropy(ps[k]).mean(0) for k in ps}

    T = fit_temperature_entropy(dnn_npz["val_logits"].astype(np.float64), H["val"].mean())
    Hd = {k: entropy(softmax(dnn_npz[f"{k}_logits"].astype(np.float64) / T, -1))
          for k in ["test"] + oods}

    out = dict(T=T, H_id=float(H["test"].mean()), **classification(mean["test"], ys["test"]))
    for o in oods:
        out[f"auroc_{o}_entropy"] = auroc(H["test"], H[o])
        out[f"auroc_{o}_mi"] = auroc(MI["test"], MI[o])
        out[f"auroc_{o}_dnnT"] = auroc(Hd["test"], Hd[o])
        out[f"delta_{o}"] = out[f"auroc_{o}_entropy"] - out[f"auroc_{o}_dnnT"]
        out[f"delta_{o}_mi"] = out[f"auroc_{o}_mi"] - out[f"auroc_{o}_dnnT"]
        out[f"fpr95_{o}_entropy"] = fpr95(H["test"], H[o])
        out[f"fpr95_{o}_mi"] = fpr95(MI["test"], MI[o])
        out[f"fpr95_{o}_dnnT"] = fpr95(Hd["test"], Hd[o])
    return out
