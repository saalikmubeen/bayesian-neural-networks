"""Datasets as in-memory tensors: a stratified MNIST training subset, a disjoint
validation set, the MNIST test set and OOD test sets with MNIST normalisation."""

import numpy as np
import torch
from torchvision import datasets

DATA_ROOT = "./data"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_VAL = 5000
VAL_SEED = 12345  # validation pool is fixed across seeds

OOD_SETS = {
    "fashion": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}


def _to_tensor(ds):
    x = ds.data.float().div(255.0).sub(MNIST_MEAN).div(MNIST_STD).unsqueeze(1)
    y = torch.as_tensor(ds.targets).long()
    return x, y


def stratified_indices(labels, n, rng, exclude=()):
    """n indices with n // 10 per class (remainder filled at random), never from `exclude`."""
    labels = np.asarray(labels)
    allowed = np.ones(len(labels), bool)
    allowed[list(exclude)] = False
    idx = []
    for c in np.unique(labels):
        ci = np.where((labels == c) & allowed)[0]
        idx.extend(rng.choice(ci, n // 10, replace=False).tolist())
    rest = np.setdiff1d(np.where(allowed)[0], idx)
    idx.extend(rng.choice(rest, n - len(idx), replace=False).tolist())
    return np.array(idx)


def support_mask(x_train):
    """Pixels that vary in the training images (False = constant in every image)."""
    flat = x_train.flatten(1)
    return (flat.max(0).values > flat.min(0).values).view(x_train.shape[1:])


def load(train_size, seed, on_support=False):
    """Returns dict of (x, y) tensors: train, val, test, and one entry per OOD set.

    on_support=True also adds <ood>_onsupp: the OOD images with every pixel that
    is constant across the training subset reset to that constant (background),
    removing the "ink where training images never have ink" shortcut."""
    train = datasets.MNIST(DATA_ROOT, train=True, download=True)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True)
    x_all, y_all = _to_tensor(train)

    val_idx = stratified_indices(y_all.numpy(), N_VAL, np.random.RandomState(VAL_SEED))
    tr_idx = stratified_indices(
        y_all.numpy(), train_size, np.random.RandomState(seed), exclude=val_idx,
    )
    assert len(np.intersect1d(val_idx, tr_idx)) == 0

    out = {
        "train": (x_all[tr_idx], y_all[tr_idx]),
        "val": (x_all[val_idx], y_all[val_idx]),
        "test": _to_tensor(test),
    }
    for name, cls in OOD_SETS.items():
        out[name] = _to_tensor(cls(DATA_ROOT, train=False, download=True))
    if on_support:
        live = support_mask(out["train"][0])
        bg = out["train"][0].flatten(1).min(0).values.view(live.shape)  # constant value of dead pixels
        for name in OOD_SETS:
            x, y = out[name]
            out[f"{name}_onsupp"] = (torch.where(live, x, bg.expand_as(x)), y)
    return out


def batches(x, y, batch_size, generator):
    perm = torch.randperm(len(x), generator=generator, device=x.device)
    for i in range(0, len(x), batch_size):
        j = perm[i : i + batch_size]
        yield x[j], y[j]
