"""
Step 2: CIFAR-10 full analysis.
Runtime: ~45 min GPU.

  - BNN Bayesian h1 distances (not just CNN backbone)
  - DNN distances for comparison (is this BNN-specific?)
  - AUROC for all signals
  - BNN vs DNN ratio (the critical question)

Outputs:
  figures_cifar/tsne_cifar_bnn.png
  figures_cifar/distance_distributions_cifar.png
  figures_cifar/cifar_results.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)
DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")
Path("figures_cifar").mkdir(exist_ok=True)

TRAIN_SIZE = 25000
N = 500
SEED = 0
FGSM_EPS = 0.03
PGD_EPS = 0.03
PGD_ALPHA = 0.007
PGD_STEPS = 20
EPOCHS_DNN = 50
EPOCHS_BNN = 80
LR_DNN = 1e-3
LR_BNN = 3e-4

# ── Data ───────────────────────────────────────────────────────────────────────
t_tr = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ],
)
t_te = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ],
)

c10_train = datasets.CIFAR10("./data", train=True, download=True, transform=t_tr)
c10_test = datasets.CIFAR10("./data", train=False, download=True, transform=t_te)
c100_test = datasets.CIFAR100("./data", train=False, download=True, transform=t_te)


# ── Architecture ───────────────────────────────────────────────────────────────
class BayesianLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(o, i) * 0.1)
        self.rho = nn.Parameter(torch.full((o, i), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(o))
        self.bias_rho = nn.Parameter(torch.full((o,), -3.0))

    @property
    def sigma_w(self):
        return F.softplus(self.rho)

    @property
    def sigma_b(self):
        return F.softplus(self.bias_rho)

    def forward(self, x):
        return F.linear(
            x,
            self.mu + self.sigma_w * torch.randn_like(self.mu),
            self.bias_mu + self.sigma_b * torch.randn_like(self.bias_mu),
        )

    def kl(self):
        kw = 0.5 * (
            self.sigma_w**2 + self.mu**2 - 1 - torch.log(self.sigma_w**2 + 1e-8)
        )
        kb = 0.5 * (
            self.sigma_b**2 + self.bias_mu**2 - 1 - torch.log(self.sigma_b**2 + 1e-8)
        )
        return kw.sum() + kb.sum()


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.1),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.1),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


FLAT = 128 * 4 * 4  # 32x32 -> 4x4 after 3 maxpools


class CIFAR_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_cnn()
        self.head = nn.Sequential(
            nn.Linear(FLAT, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        self._f = f
        return self.head(f)

    def get_features(self, x):
        self.eval()
        with torch.no_grad():
            return self.backbone(x).view(x.size(0), -1)


class CIFAR_BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_cnn()
        self.bay1 = BayesianLinear(FLAT, 256)
        self.bay2 = BayesianLinear(256, 10)

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        self._h1 = F.relu(self.bay1(f))
        return self.bay2(self._h1)

    def kl_total(self):
        return self.bay1.kl() + self.bay2.kl()

    def get_features(self, x):
        self.eval()
        with torch.no_grad():
            return self.backbone(x).view(x.size(0), -1)

    def get_bay_h1(self, x, n=20):
        self.eval()
        acts = []
        with torch.no_grad():
            for _ in range(n):
                f = self.backbone(x).view(x.size(0), -1)
                acts.append(F.relu(self.bay1(f)))
        return torch.stack(acts).mean(0)


# ── Training ───────────────────────────────────────────────────────────────────
def stratified(ds, n, seed):
    rng = np.random.RandomState(seed)
    labels = np.array(ds.targets)
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        idx.extend(
            rng.choice(ci, min(max(1, n // 10), len(ci)), replace=False).tolist(),
        )
    rem = list(set(range(len(ds))) - set(idx))
    rng.shuffle(rem)
    idx.extend(rem[: max(0, n - len(idx))])
    return idx[:n]


def train_dnn(m, loader):
    opt = torch.optim.Adam(m.parameters(), lr=LR_DNN, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS_DNN)
    m.train()
    for ep in range(EPOCHS_DNN):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"  DNN epoch {ep + 1}/{EPOCHS_DNN}", end="\r")
    print()


def train_bnn(m, loader):
    for p in m.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR_BNN)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS_BNN)
    nb = len(loader)
    m.train()
    for ep in range(EPOCHS_BNN):
        kw = min(1.0, (ep + 1) / 20) / nb
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y) + kw * m.kl_total()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"  BNN epoch {ep + 1}/{EPOCHS_BNN}", end="\r")
    print()


# ── Attacks ────────────────────────────────────────────────────────────────────
def fgsm(m, x, y, eps):
    x = x.clone().requires_grad_(True)
    F.cross_entropy(m(x), y).backward()
    return (x + eps * x.grad.sign()).detach()


def pgd(m, x, y, eps, alpha, steps):
    xa = x.clone() + torch.zeros_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        F.cross_entropy(m(xa), y).backward()
        xa = xa.detach() + alpha * xa.grad.sign()
        xa = torch.max(torch.min(xa, x + eps), x - eps)
    return xa.detach()


# ── Metrics ────────────────────────────────────────────────────────────────────
def nn_dist(q, r, k=5):
    return (
        NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
        .fit(r)
        .kneighbors(q)[0]
        .mean(1)
    )


def entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(-1)


def auroc(d_id, d_ood):
    s = np.concatenate([d_id, d_ood])
    l = np.concatenate([np.zeros(len(d_id)), np.ones(len(d_ood))])
    return float(roc_auc_score(l, s))


def bnn_predict(m, loader, ns=30):
    m.eval()
    ss, ls = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        bs = []
        with torch.no_grad():
            for _ in range(ns):
                bs.append(F.softmax(m(x), -1).cpu())
        ss.append(torch.stack(bs))
        ls.append(y)
    s = torch.cat(ss, 1)
    return s.mean(0), torch.cat(ls)


# ── Run ────────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
idx = stratified(c10_train, TRAIN_SIZE, SEED)
g = torch.Generator()
g.manual_seed(SEED)
tl = DataLoader(
    Subset(c10_train, idx),
    batch_size=128,
    shuffle=True,
    generator=g,
    num_workers=0,
)

print(f"Training DNN on CIFAR-10 (n={TRAIN_SIZE})...")
dnn = CIFAR_DNN().to(DEVICE)
train_dnn(dnn, tl)

print("Training BNN (Bayesian head, frozen backbone)...")
bnn = CIFAR_BNN().to(DEVICE)
bnn.backbone.load_state_dict(dnn.backbone.state_dict())  # init from DNN
train_bnn(bnn, tl)

# Accuracy
te_loader = DataLoader(c10_test, batch_size=256, shuffle=False, num_workers=0)
dnn.eval()
with torch.no_grad():
    dnn_acc = sum(
        (dnn(x.to(DEVICE)).argmax(-1).cpu() == y).sum().item() for x, y in te_loader
    ) / len(c10_test)
mp_id, lab_id = bnn_predict(bnn, te_loader)
bnn_acc = (mp_id.argmax(-1) == lab_id).float().mean().item()
print(f"DNN acc: {dnn_acc:.4f}  BNN acc: {bnn_acc:.4f}")

# Input samples
x_c10, y_c10 = next(
    iter(DataLoader(Subset(c10_test, list(range(N))), batch_size=N, num_workers=0)),
)
x_c100, y_c100 = next(
    iter(DataLoader(Subset(c100_test, list(range(N))), batch_size=N, num_workers=0)),
)

dnn.eval()
print("Generating FGSM...")
x_fgsm = torch.cat(
    [
        fgsm(
            dnn,
            x_c10[i : i + 128].to(DEVICE),
            y_c10[i : i + 128].to(DEVICE),
            FGSM_EPS,
        ).cpu()
        for i in range(0, N, 128)
    ],
)
print("Generating PGD...")
x_pgd_p = []
for i in range(0, N, 64):
    x_pgd_p.append(
        pgd(
            dnn,
            x_c10[i : i + 64].to(DEVICE),
            y_c10[i : i + 64].to(DEVICE),
            PGD_EPS,
            PGD_ALPHA,
            PGD_STEPS,
        ).cpu(),
    )
    print(f"  PGD {i // 64 + 1}/{N // 64}", end="\r")
x_pgd = torch.cat(x_pgd_p)
print()

with torch.no_grad():
    fgsm_acc = (dnn(x_fgsm.to(DEVICE)).argmax(-1).cpu() == y_c10).float().mean().item()
    pgd_acc = (dnn(x_pgd.to(DEVICE)).argmax(-1).cpu() == y_c10).float().mean().item()
print(f"DNN acc after attack: FGSM={fgsm_acc:.4f}  PGD={pgd_acc:.4f}")

# BNN entropy
mk = lambda x, y: DataLoader(
    TensorDataset(x, y),
    batch_size=256,
    shuffle=False,
    num_workers=0,
)
mp_c10, _ = bnn_predict(bnn, mk(x_c10, y_c10))
mp_c100, _ = bnn_predict(bnn, mk(x_c100, y_c100))
mp_fg, _ = bnn_predict(bnn, mk(x_fgsm, y_c10))
mp_pgd, _ = bnn_predict(bnn, mk(x_pgd, y_c10))

ents = {
    k: float(entropy(v).numpy().mean())
    for k, v in [
        ("CIFAR-10", mp_c10),
        ("FGSM", mp_fg),
        ("PGD", mp_pgd),
        ("CIFAR-100", mp_c100),
    ]
}

# ── Three distance spaces ──────────────────────────────────────────────────────
print("\nExtracting features...")
inputs_map = {"CIFAR-10": x_c10, "FGSM": x_fgsm, "PGD": x_pgd, "CIFAR-100": x_c100}


def batch_extract(fn, x):
    return np.concatenate(
        [fn(x[i : i + 128].to(DEVICE)).cpu().numpy() for i in range(0, len(x), 128)],
    )


# CNN backbone (shared between BNN and DNN)
feats_cnn = {k: batch_extract(bnn.get_features, v) for k, v in inputs_map.items()}
# Bayesian h1 (uncertainty-aware)
feats_bay = {
    k: batch_extract(lambda x: bnn.get_bay_h1(x), v) for k, v in inputs_map.items()
}
# DNN features
feats_dnn = {k: batch_extract(dnn.get_features, v) for k, v in inputs_map.items()}

# Training references
tl_ref = DataLoader(
    Subset(c10_train, idx),
    batch_size=128,
    shuffle=False,
    num_workers=0,
)
tr_cnn = np.concatenate(
    [bnn.get_features(x.to(DEVICE)).cpu().numpy() for x, _ in tl_ref],
)
tr_bay = np.concatenate([bnn.get_bay_h1(x.to(DEVICE)).cpu().numpy() for x, _ in tl_ref])
tr_dnn = np.concatenate(
    [dnn.get_features(x.to(DEVICE)).cpu().numpy() for x, _ in tl_ref],
)

dists_cnn = {k: nn_dist(v, tr_cnn) for k, v in feats_cnn.items()}
dists_bay = {k: nn_dist(v, tr_bay) for k, v in feats_bay.items()}
dists_dnn = {k: nn_dist(v, tr_dnn) for k, v in feats_dnn.items()}

# ── Results ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CIFAR-10 FULL RESULTS TABLE")
print("=" * 80)
print(
    f"\n{'Signal':<28} {'CIFAR-10':>10} {'FGSM':>8} {'PGD':>8} {'CIFAR-100':>10} {'AUROC':>8}",
)
print("-" * 75)

id_k = "CIFAR-10"
ood_k = "CIFAR-100"
rows = [
    ("BNN CNN distance", dists_cnn),
    ("BNN Bayesian h1 dist", dists_bay),
    ("DNN distance", dists_dnn),
]
for label, dists in rows:
    d_id = dists[id_k]
    d_ood = dists[ood_k]
    print(
        f"{label:<28} {dists[id_k].mean():>10.2f} {dists['FGSM'].mean():>8.2f} "
        f"{dists['PGD'].mean():>8.2f} {dists[ood_k].mean():>10.2f} "
        f"{auroc(d_id, d_ood):>8.3f}",
    )

ent_id_arr = entropy(mp_c10).numpy()
ent_ood_arr = entropy(mp_c100).numpy()
print(
    f"{'BNN entropy':<28} {ents['CIFAR-10']:>10.3f} {ents['FGSM']:>8.3f} "
    f"{ents['PGD']:>8.3f} {ents['CIFAR-100']:>10.3f} "
    f"{auroc(ent_id_arr, ent_ood_arr):>8.3f}",
)

# The critical ratio
bnn_ratio = dists_bay["FGSM"].mean() / dists_bay[ood_k].mean()
dnn_ratio = dists_dnn["FGSM"].mean() / dists_dnn[ood_k].mean()

print(f"\n{'=' * 60}")
print("IS THE EFFECT BNN-SPECIFIC?")
print(f"{'=' * 60}")
print(f"BNN Bayesian h1: FGSM/OOD ratio = {bnn_ratio:.3f}")
print(f"DNN features:    FGSM/OOD ratio = {dnn_ratio:.3f}")
print("(lower = adversarials more similar to training data than OOD)")
if bnn_ratio < dnn_ratio:
    print(
        f"\nBNN shows STRONGER manifold proximity effect ({bnn_ratio:.3f} < {dnn_ratio:.3f})",
    )
    print("Supports: effect is tied to Bayesian uncertainty, not just CNN geometry")
else:
    print(f"\nDNN shows similar/stronger effect ({dnn_ratio:.3f} <= {bnn_ratio:.3f})")
    print("Effect may be CNN geometry — revise paper framing accordingly")

# Ordering check
adv_ok = (
    dists_bay["FGSM"].mean() < dists_bay[ood_k].mean()
    and dists_bay["PGD"].mean() < dists_bay[ood_k].mean()
)
ent_ok = ents["FGSM"] < ents["CIFAR-100"] and ents["PGD"] < ents["CIFAR-100"]
print(
    f"\nDistance ordering (FGSM,PGD < CIFAR-100): {'CONFIRMED' if adv_ok else 'NOT CONFIRMED'}",
)
print(
    f"Entropy ordering (FGSM,PGD < CIFAR-100):  {'CONFIRMED' if ent_ok else 'NOT CONFIRMED'}",
)

if adv_ok and ent_ok:
    print("\nMECHANISM GENERALIZES TO CIFAR-10.")
else:
    print("\nOrdering does not hold on CIFAR-10. Report honestly and investigate.")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print("\nRunning t-SNE...")
import sklearn

sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
from sklearn.manifold import TSNE

kw = dict(n_components=2, perplexity=30, random_state=42)
kw["max_iter" if sk_ver >= (1, 2) else "n_iter"] = 1000

all_f = np.concatenate(
    [feats_bay["CIFAR-10"], feats_bay["FGSM"], feats_bay["CIFAR-100"]],
)
emb = TSNE(**kw).fit_transform(all_f)

fig, ax = plt.subplots(figsize=(8, 7))
for i, (c, l) in enumerate(
    zip(
        ["#2196F3", "#FF9800", "#F44336"],
        ["CIFAR-10 (ID)", f"FGSM ε={FGSM_EPS}", "CIFAR-100 (OOD)"],
    ),
):
    s = i * N
    ax.scatter(emb[s : s + N, 0], emb[s : s + N, 1], s=5, alpha=0.45, color=c, label=l)
ax.set_title(
    "t-SNE of BNN Bayesian h1 Activations (CIFAR-10)\n"
    "CIFAR-10 vs FGSM vs CIFAR-100 (OOD)",
    fontsize=12,
)
ax.legend(fontsize=11, markerscale=3)
ax.axis("off")
plt.tight_layout()
plt.savefig("figures_cifar/tsne_cifar_bnn.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures_cifar/tsne_cifar_bnn.png")

# ── Distance histogram ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("CIFAR-10: Distance to Training Manifold", fontsize=12)
colors_h = {
    "CIFAR-10": "#2196F3",
    "FGSM": "#FF9800",
    "PGD": "#FF5722",
    "CIFAR-100": "#F44336",
}
for ax, (title, dists) in zip(
    axes,
    [("BNN CNN", dists_cnn), ("BNN Bayesian h1", dists_bay), ("DNN", dists_dnn)],
):
    for nm, d in dists.items():
        ax.hist(
            d,
            bins=30,
            alpha=0.55,
            color=colors_h[nm],
            density=True,
            label=f"{nm} μ={d.mean():.1f}",
        )
        ax.axvline(d.mean(), color=colors_h[nm], ls="--", lw=1.5)
    ax.set_xlabel("k-NN distance")
    ax.set_title(title)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    "figures_cifar/distance_distributions_cifar.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()
print("Saved: figures_cifar/distance_distributions_cifar.png")

# ── Save ───────────────────────────────────────────────────────────────────────
with open("figures_cifar/cifar_results.json", "w") as f:
    json.dump(
        {
            "dnn_acc": dnn_acc,
            "bnn_acc": bnn_acc,
            "dnn_acc_fgsm": fgsm_acc,
            "dnn_acc_pgd": pgd_acc,
            "entropy": ents,
            "distances_bnn_cnn": {k: float(v.mean()) for k, v in dists_cnn.items()},
            "distances_bnn_bay": {k: float(v.mean()) for k, v in dists_bay.items()},
            "distances_dnn": {k: float(v.mean()) for k, v in dists_dnn.items()},
            "auroc": {
                "bnn_cnn": auroc(dists_cnn["CIFAR-10"], dists_cnn["CIFAR-100"]),
                "bnn_bay": auroc(dists_bay["CIFAR-10"], dists_bay["CIFAR-100"]),
                "dnn": auroc(dists_dnn["CIFAR-10"], dists_dnn["CIFAR-100"]),
                "entropy": auroc(ent_id_arr, ent_ood_arr),
            },
            "bnn_vs_dnn_ratio": {"bnn": bnn_ratio, "dnn": dnn_ratio},
            "ordering": {
                "distance": adv_ok,
                "entropy": ent_ok,
                "both": adv_ok and ent_ok,
            },
        },
        f,
        indent=2,
    )
print("Saved: figures_cifar/cifar_results.json")
print("\nPaste the full results table and verdict above into your notes.")
