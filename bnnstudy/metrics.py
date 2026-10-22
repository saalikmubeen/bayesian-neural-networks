"""Uncertainty scores and metrics, all computed from saved per-sample outputs."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def entropy(p):
    return -(p * np.log(p + EPS)).sum(-1)


def scores(mean_p, mean_ent=None):
    """Uncertainty scores (higher = more uncertain). `mean_ent` is E_s[H(p_s)]."""
    out = {"entropy": entropy(mean_p), "msp": 1 - mean_p.max(-1)}
    if mean_ent is not None:
        out["mi"] = out["entropy"] - mean_ent
    return out


def auroc(u_id, u_ood):
    y = np.r_[np.zeros(len(u_id)), np.ones(len(u_ood))]
    return float(roc_auc_score(y, np.r_[u_id, u_ood]))


def fpr95(u_id, u_ood):
    """False-positive rate at 95% true-positive rate: the share of OOD inputs
    accepted when the threshold keeps 95% of in-distribution inputs (lower is better)."""
    return float((u_ood <= np.quantile(u_id, 0.95)).mean())


def ece(p, y, n_bins=15):
    conf, correct = p.max(-1), (p.argmax(-1) == y)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(
        (bins == b).mean() * abs(correct[bins == b].mean() - conf[bins == b].mean())
        for b in range(n_bins) if (bins == b).any()
    ))


def selective(u, correct, coverage=0.8):
    """AURC (area under risk-coverage, lower is better) and accuracy at a fixed coverage."""
    order = np.argsort(u, kind="stable")
    c = correct[order].astype(float)
    risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    k = int(round(coverage * len(c)))
    return float(risk.mean()), float(c[:k].mean())


def classification(p, y):
    correct = p.argmax(-1) == y
    return {
        "acc": float(correct.mean()),
        "nll": float(-np.log(p[np.arange(len(y)), y] + EPS).mean()),
        "ece": ece(p, y),
    }


# ── Temperature ────────────────────────────────────────────────────────────────


def fit_temperature_nll(logits, y):
    f = lambda t: -log_softmax(logits / t, -1)[np.arange(len(y)), y].mean()
    return float(minimize_scalar(f, bounds=(0.05, 20), method="bounded").x)


def fit_temperature_entropy(logits, target):
    """Temperature at which the mean predictive entropy equals `target` (monotone in T)."""
    f = lambda t: entropy(softmax(logits / t, -1)).mean() - target
    lo, hi = 0.05, 100.0
    if f(hi) < 0:
        return hi
    return float(brentq(f, lo, hi))
