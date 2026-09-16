"""
Step 1: PGD attack + fc2 layer analysis on MNIST.
Runtime: ~15 min GPU.

Outputs:
  figures_mechanism/pgd_fc2_distance_distributions.png
  figures_mechanism/paper_numbers_extended.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
Path("figures_mechanism").mkdir(exist_ok=True)

N, TRAIN_SIZE, SEED = 1000, 2000, 0
FGSM_EPS = 0.3
PGD_EPS = 0.3
PGD_ALPHA = 0.01
PGD_STEPS = 40

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
)
mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
fashion_test = datasets.FashionMNIST(
    "./data",
    train=False,
    download=True,
    transform=transform,
)


# ── Models ─────────────────────────────────────────────────────────────────────
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


class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(784, 256)
        self.fc2 = BayesianLinear(256, 256)
        self.fc3 = BayesianLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        self._h1 = F.relu(self.fc1(x))
        self._h2 = F.relu(self.fc2(self._h1))
        return self.fc3(self._h2)

    def kl_total(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()

    def get_act(self, x, layer, n=20):
        self.eval()
        acts = []
        with torch.no_grad():
            for _ in range(n):
                self.forward(x.view(x.size(0), -1))
                acts.append((self._h1 if layer == "fc1" else self._h2).clone())
        return torch.stack(acts).mean(0)


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


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


def train_dnn(m, loader, ep=40, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    m.train()
    for _ in range(ep):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
        sched.step()


def train_bnn(m, loader, ep=120, lr=1e-4):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    nb = len(loader)
    m.train()
    for e in range(ep):
        kw = min(1.0, (e + 1) / 20) / nb
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y) + kw * m.kl_total()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()


# ── Attacks ────────────────────────────────────────────────────────────────────
def fgsm(m, x, y, eps):
    x = x.clone().requires_grad_(True)
    F.cross_entropy(m(x), y).backward()
    return (x + eps * x.grad.sign()).detach().clamp(-3, 3)


def pgd(m, x, y, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    xa = x.clone() + torch.zeros_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        F.cross_entropy(m(xa), y).backward()
        xa = xa.detach() + alpha * xa.grad.sign()
        xa = torch.max(torch.min(xa, x + eps), x - eps).clamp(-3, 3)
    return xa.detach()


def nn_dist(q, r, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(r)
    return nbrs.kneighbors(q)[0].mean(1)


def entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(-1)


def bnn_predict(m, loader, ns=50):
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


# ── Setup ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
idx = stratified(mnist_train, TRAIN_SIZE, SEED)
g = torch.Generator()
g.manual_seed(SEED)
tl = DataLoader(
    Subset(mnist_train, idx),
    batch_size=64,
    shuffle=True,
    generator=g,
    num_workers=0,
)

print("Training DNN...")
dnn = DNN().to(DEVICE)
train_dnn(dnn, tl)
print("Training BNN...")
bnn = BNN().to(DEVICE)
train_bnn(bnn, tl)

x_mnist, y_mnist = next(
    iter(DataLoader(Subset(mnist_test, list(range(N))), batch_size=N, num_workers=0)),
)
x_fash, y_fash = next(
    iter(DataLoader(Subset(fashion_test, list(range(N))), batch_size=N, num_workers=0)),
)

dnn.eval()
print("Generating FGSM...")
x_fgsm = torch.cat(
    [
        fgsm(
            dnn,
            x_mnist[i : i + 256].to(DEVICE),
            y_mnist[i : i + 256].to(DEVICE),
            FGSM_EPS,
        ).cpu()
        for i in range(0, N, 256)
    ],
)
print("Generating PGD (slow ~10min)...")
x_pgd_p = []
for i in range(0, N, 64):
    x_pgd_p.append(
        pgd(dnn, x_mnist[i : i + 64].to(DEVICE), y_mnist[i : i + 64].to(DEVICE)).cpu(),
    )
    print(f"  PGD {i // 64 + 1}/{N // 64}", end="\r")
x_pgd = torch.cat(x_pgd_p)
print()

with torch.no_grad():
    fgsm_acc = (
        (dnn(x_fgsm.to(DEVICE)).argmax(-1).cpu() == y_mnist).float().mean().item()
    )
    pgd_acc = (dnn(x_pgd.to(DEVICE)).argmax(-1).cpu() == y_mnist).float().mean().item()
print(f"DNN acc after attack: FGSM={fgsm_acc:.4f}  PGD={pgd_acc:.4f}")

# Entropy
mk = lambda x, y: DataLoader(
    TensorDataset(x, y),
    batch_size=512,
    shuffle=False,
    num_workers=0,
)
mp_id, _ = bnn_predict(bnn, mk(x_mnist, y_mnist))
mp_f, _ = bnn_predict(bnn, mk(x_fash, y_fash))
mp_fg, _ = bnn_predict(bnn, mk(x_fgsm, y_mnist))
mp_pgd, _ = bnn_predict(bnn, mk(x_pgd, y_mnist))
ents = {
    k: float(entropy(v).numpy().mean())
    for k, v in [("MNIST", mp_id), ("FGSM", mp_fg), ("PGD", mp_pgd), ("Fashion", mp_f)]
}

# Activations + distances
inputs = {"MNIST": x_mnist, "FGSM": x_fgsm, "PGD": x_pgd, "Fashion": x_fash}


def get_acts(layer):
    out = {}
    for nm, x in inputs.items():
        chunks = [
            bnn.get_act(x[i : i + 256].to(DEVICE), layer).cpu().numpy()
            for i in range(0, len(x), 256)
        ]
        out[nm] = np.concatenate(chunks)
    return out


acts_fc1 = get_acts("fc1")
acts_fc2 = get_acts("fc2")

# Training reference activations
tl_ref = DataLoader(
    Subset(mnist_train, idx),
    batch_size=500,
    shuffle=False,
    num_workers=0,
)
tr_fc1 = np.concatenate(
    [bnn.get_act(xb.to(DEVICE), "fc1").cpu().numpy() for xb, _ in tl_ref],
)
tr_fc2 = np.concatenate(
    [bnn.get_act(xb.to(DEVICE), "fc2").cpu().numpy() for xb, _ in tl_ref],
)

dists_fc1 = {k: nn_dist(v, tr_fc1) for k, v in acts_fc1.items()}
dists_fc2 = {k: nn_dist(v, tr_fc2) for k, v in acts_fc2.items()}

# Print results
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)
print(f"\n{'Input':<12} {'fc1 dist':>10} {'fc2 dist':>10} {'entropy':>10}")
print("-" * 45)
for nm in ["MNIST", "FGSM", "PGD", "Fashion"]:
    print(
        f"{nm:<12} {dists_fc1[nm].mean():>10.3f} {dists_fc2[nm].mean():>10.3f} {ents[nm]:>10.3f}",
    )

fc1_ok = (
    dists_fc1["FGSM"].mean() < dists_fc1["Fashion"].mean()
    and dists_fc1["PGD"].mean() < dists_fc1["Fashion"].mean()
)
fc2_ok = (
    dists_fc2["FGSM"].mean() < dists_fc2["Fashion"].mean()
    and dists_fc2["PGD"].mean() < dists_fc2["Fashion"].mean()
)
ent_ok = ents["FGSM"] < ents["Fashion"] and ents["PGD"] < ents["Fashion"]

print(f"\nfc1 ordering: {'CONFIRMED' if fc1_ok else 'BROKEN'}")
print(
    f"fc2 ordering: {'CONFIRMED — not layer specific' if fc2_ok else 'BROKEN at fc2'}",
)
print(f"Entropy ordering: {'CONFIRMED' if ent_ok else 'BROKEN'}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Robustness: PGD + fc2 Check (MNIST)", fontsize=12)
colors = {"MNIST": "#2196F3", "FGSM": "#FF9800", "PGD": "#FF5722", "Fashion": "#F44336"}
for ax, (layer, dists) in zip(axes, [("fc1", dists_fc1), ("fc2", dists_fc2)]):
    for nm, d in dists.items():
        ax.hist(
            d,
            bins=35,
            alpha=0.55,
            color=colors[nm],
            density=True,
            label=f"{nm} μ={d.mean():.1f}",
        )
        ax.axvline(d.mean(), color=colors[nm], ls="--", lw=1.8)
    ax.set_xlabel("k-NN distance")
    ax.set_ylabel("Density")
    ax.set_title(f"BNN {layer}")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures_mechanism/pgd_fc2_distances.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: figures_mechanism/pgd_fc2_distances.png")

with open("figures_mechanism/pgd_fc2_results.json", "w") as f:
    json.dump(
        {
            "fc1": {k: float(v.mean()) for k, v in dists_fc1.items()},
            "fc2": {k: float(v.mean()) for k, v in dists_fc2.items()},
            "entropy": ents,
            "dnn_acc": {"fgsm": fgsm_acc, "pgd": pgd_acc},
            "ordering": {"fc1": fc1_ok, "fc2": fc2_ok, "entropy": ent_ok},
        },
        f,
        indent=2,
    )
print("Saved: figures_mechanism/pgd_fc2_results.json")
print("\nPaste the RESULTS table above into your notes.")
