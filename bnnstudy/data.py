"""Datasets as in-memory tensors: a stratified training subset of the
in-distribution dataset (MNIST by default), a disjoint validation set, its test
set and OOD test sets, all normalised with the in-distribution statistics.

Set `ID = "fashion"` (train.py --id fashion; E21) to use Fashion-MNIST as the
in-distribution data, with MNIST and KMNIST as OOD sets, or `ID = "mnist04"`
(E22) to train on MNIST digits 0-4 only, with the test digits 5-9 as a near-OOD
set ("mnist59") next to Fashion-MNIST and KMNIST."""

import numpy as np
import torch
from torchvision import datasets

DATA_ROOT = "./data"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_VAL = 5000
VAL_SEED = 12345  # validation pool is fixed across seeds

FASHION_MEAN, FASHION_STD = 0.2860, 0.3530

ID = "mnist"  # in-distribution dataset: "mnist" or "fashion" (module global, set by train.py --id)
ID_SETS = {"mnist": (datasets.MNIST, MNIST_MEAN, MNIST_STD),
           "fashion": (datasets.FashionMNIST, FASHION_MEAN, FASHION_STD),
           "mnist04": (datasets.MNIST, MNIST_MEAN, MNIST_STD)}
HELD_OUT = 5  # with ID = "mnist04", classes >= HELD_OUT are out-of-distribution
OOD_SETS = {  # OOD sets for MNIST as in-distribution (used by most scripts)
    "fashion": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}
ALL_OOD = dict(OOD_SETS, mnist=datasets.MNIST, mnist59=None)  # every OOD set name any ID choice uses


def ood_names():
    """OOD sets used with the current in-distribution dataset."""
    return {"mnist": ["fashion", "kmnist"], "fashion": ["mnist", "kmnist"],
            "mnist04": ["mnist59", "fashion", "kmnist"]}[ID]


def _to_tensor(ds, mean=MNIST_MEAN, std=MNIST_STD):
    x = ds.data.float().div(255.0).sub(mean).div(std).unsqueeze(1)
    y = torch.as_tensor(ds.targets).long()
    return x, y


def stratified_indices(labels, n, rng, exclude=()):
    """n indices with n // (number of classes) per class (remainder filled at random), never from `exclude`."""
    labels = np.asarray(labels)
    allowed = np.ones(len(labels), bool)
    allowed[list(exclude)] = False
    idx = []
    classes = np.unique(labels)
    for c in classes:
        ci = np.where((labels == c) & allowed)[0]
        idx.extend(rng.choice(ci, n // len(classes), replace=False).tolist())
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
    cls_id, mean, std = ID_SETS[ID]
    train = cls_id(DATA_ROOT, train=True, download=True)
    test = cls_id(DATA_ROOT, train=False, download=True)
    x_all, y_all = _to_tensor(train, mean, std)
    x_test, y_test = _to_tensor(test, mean, std)
    if ID == "mnist04":  # keep digits 0-4 in distribution; test digits 5-9 become an OOD set
        keep = y_all < HELD_OUT
        x_all, y_all = x_all[keep], y_all[keep]
        held = y_test >= HELD_OUT
        x_test, y_test, x_held, y_held = x_test[~held], y_test[~held], x_test[held], y_test[held]

    val_idx = stratified_indices(y_all.numpy(), N_VAL, np.random.RandomState(VAL_SEED))
    tr_idx = stratified_indices(
        y_all.numpy(), train_size, np.random.RandomState(seed), exclude=val_idx,
    )
    assert len(np.intersect1d(val_idx, tr_idx)) == 0

    out = {
        "train": (x_all[tr_idx], y_all[tr_idx]),
        "val": (x_all[val_idx], y_all[val_idx]),
        "test": (x_test, y_test),
    }
    for name in ood_names():
        if name == "mnist59":
            out[name] = (x_held, y_held)
        else:
            out[name] = _to_tensor(ALL_OOD[name](DATA_ROOT, train=False, download=True), mean, std)
    if on_support:
        live = support_mask(out["train"][0])
        bg = out["train"][0].flatten(1).min(0).values.view(live.shape)  # constant value of dead pixels
        for name in ood_names():
            x, y = out[name]
            out[f"{name}_onsupp"] = (torch.where(live, x, bg.expand_as(x)), y)
    return out


def batches(x, y, batch_size, generator):
    perm = torch.randperm(len(x), generator=generator, device=x.device)
    for i in range(0, len(x), batch_size):
        j = perm[i : i + batch_size]
        yield x[j], y[j]
