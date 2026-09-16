"""
Step 1: Train all models on MNIST with fixed bugs.
Run this first. Takes ~20 min on GPU, ~1 hour on CPU.

Fixes applied vs original code:
  1. KL weight = 1/n_batches (not 1/n_samples)
  2. KL annealing over first 10 epochs
  3. Ensemble members seeded differently for diversity
  4. Cosine LR schedule for fair DNN comparison
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── Reproducibility ────────────────────────────────────────────────────────────
MASTER_SEED = 42
torch.manual_seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_SIZES = [500, 2000, 10000]
SEEDS = [0, 1, 2]
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 64
N_MC_SAMPLES = 50
N_ENSEMBLE = 5

# ── Data ───────────────────────────────────────────────────────────────────────
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

MNIST_TRAIN = datasets.MNIST("./data", train=True, download=True, transform=transform)
MNIST_TEST = datasets.MNIST("./data", train=False, download=True, transform=transform)
FASHION_TEST = datasets.FashionMNIST(
    "./data", train=False, download=True, transform=transform
)


def stratified_subset(dataset, n, seed):
    """Stratified random subset — ensures all classes represented."""
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.targets)
    classes = np.unique(labels)
    n_per = max(1, n // len(classes))
    idx = []
    for c in classes:
        ci = np.where(labels == c)[0]
        idx.extend(rng.choice(ci, min(n_per, len(ci)), replace=False).tolist())
    remaining = list(set(range(len(dataset))) - set(idx))
    rng.shuffle(remaining)
    idx.extend(remaining[: max(0, n - len(idx))])
    return idx[:n]


def make_loaders(train_size, seed):
    idx = stratified_subset(MNIST_TRAIN, train_size, seed)
    train = Subset(MNIST_TRAIN, idx)
    g = torch.Generator()
    g.manual_seed(seed)
    return (
        DataLoader(
            train, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0
        ),
        DataLoader(MNIST_TEST, batch_size=512, shuffle=False, num_workers=0),
        DataLoader(FASHION_TEST, batch_size=512, shuffle=False, num_workers=0),
    )


# ── Models ─────────────────────────────────────────────────────────────────────


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

    @torch.no_grad()
    def predict(self, loader):
        self.eval()
        probs, labels = [], []
        for xb, yb in loader:
            probs.append(F.softmax(self(xb.to(DEVICE)), -1).cpu())
            labels.append(yb)
        mp = torch.cat(probs)
        return mp, torch.zeros_like(mp), torch.cat(labels)


class BayesianLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_f))
        self.bias_rho = nn.Parameter(torch.full((out_f,), -3.0))

    @property
    def sigma_w(self):
        return F.softplus(self.rho)

    @property
    def sigma_b(self):
        return F.softplus(self.bias_rho)

    def forward(self, x):
        w = self.mu + self.sigma_w * torch.randn_like(self.mu)
        b = self.bias_mu + self.sigma_b * torch.randn_like(self.bias_mu)
        return F.linear(x, w, b)

    def kl(self):
        kl_w = 0.5 * (
            self.sigma_w**2 + self.mu**2 - 1 - torch.log(self.sigma_w**2 + 1e-8)
        )
        kl_b = 0.5 * (
            self.sigma_b**2 + self.bias_mu**2 - 1 - torch.log(self.sigma_b**2 + 1e-8)
        )
        return kl_w.sum() + kl_b.sum()


class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(784, 256)
        self.fc2 = BayesianLinear(256, 256)
        self.fc3 = BayesianLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def kl_total(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()

    def predict(self, loader, n_samples=N_MC_SAMPLES):
        self.eval()
        all_s, labels = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            bs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    bs.append(F.softmax(self(xb), -1).cpu())
            all_s.append(torch.stack(bs))  # (S, B, 10)
            labels.append(yb)
        s = torch.cat(all_s, 1)  # (S, N, 10)
        return s.mean(0), s.std(0), torch.cat(labels)


class MCDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, loader, n_samples=N_MC_SAMPLES):
        self.train()  # ← keep dropout active
        all_s, labels = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            bs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    bs.append(F.softmax(self(xb), -1).cpu())
            all_s.append(torch.stack(bs))
            labels.append(yb)
        s = torch.cat(all_s, 1)
        return s.mean(0), s.std(0), torch.cat(labels)


class DeepEnsemble(nn.Module):
    def __init__(self, n=N_ENSEMBLE):
        super().__init__()
        self.members = nn.ModuleList([DNN() for _ in range(n)])

    def predict(self, loader):
        for m in self.members:
            m.eval()
        all_s, labels = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            with torch.no_grad():
                ps = torch.stack([F.softmax(m(xb), -1).cpu() for m in self.members])
            all_s.append(ps)
            labels.append(yb)
        s = torch.cat(all_s, 1)  # (M, N, 10)
        return s.mean(0), s.std(0), torch.cat(labels)


# ── Training ───────────────────────────────────────────────────────────────────


def train_dnn(model, loader, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()


def train_bnn(model, loader, epochs=EPOCHS, lr=LR):
    """
    FIXED:
      - kl_weight = 1 / n_batches  (not 1/N)
      - KL annealing: weight ramps from 0→1 over first 10 epochs
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_batches = len(loader)
    model.train()

    for ep in range(epochs):
        # KL annealing — ramp up over first 10 epochs
        kl_weight = min(1.0, (ep + 1) / 10) / n_batches

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            kl = model.kl_total()
            loss = ce + kl_weight * kl
            loss.backward()
            opt.step()
        sched.step()


def train_ensemble(model, loader, epochs=EPOCHS, lr=LR, base_seed=0):
    """
    FIXED: each member gets a different seed for diversity.
    """
    for i, mem in enumerate(model.members):
        # Re-init with a different seed per member
        member_seed = base_seed + i * 1000
        torch.manual_seed(member_seed)
        for layer in mem.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        mem.to(DEVICE)
        opt = torch.optim.Adam(mem.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        mem.train()
        for ep in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                F.cross_entropy(mem(xb), yb).backward()
                opt.step()
            sched.step()


# ── Metrics ────────────────────────────────────────────────────────────────────


def predictive_entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(-1)


def ece(probs, labels, n_bins=15):
    conf = probs.max(-1).values.numpy()
    preds = probs.argmax(-1).numpy()
    corr = (preds == labels.numpy()).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece_v = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece_v += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece_v)


def auroc(ent_id, ent_ood):
    scores = np.concatenate([ent_id, ent_ood])
    labels = np.concatenate([np.zeros(len(ent_id)), np.ones(len(ent_ood))])
    return float(roc_auc_score(labels, scores))


def acc_coverage_curve(entropy, correct, n_pts=200):
    idx = np.argsort(entropy)
    corr = correct[idx]
    ts = np.linspace(1, len(corr), n_pts, dtype=int)
    return (ts / len(corr)).tolist(), [corr[:t].mean() for t in ts]


def compute_metrics(mp_id, lab_id, mp_ood, tau=0.5):
    ent_id = predictive_entropy(mp_id).numpy()
    ent_ood = predictive_entropy(mp_ood).numpy()
    correct = (mp_id.argmax(-1) == lab_id).numpy().astype(float)
    p_true = mp_id[torch.arange(len(lab_id)), lab_id].numpy()
    oh = F.one_hot(lab_id, 10).float().numpy()
    mask = ent_id < tau

    cov, acc_cov = acc_coverage_curve(ent_id, correct)

    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(p_true + 1e-8).mean()),
        "brier": float(((mp_id.numpy() - oh) ** 2).sum(-1).mean()),
        "ece": ece(mp_id, lab_id),
        "auroc": auroc(ent_id, ent_ood),
        "entropy_id": float(ent_id.mean()),
        "entropy_ood": float(ent_ood.mean()),
        "conf_id": float(mp_id.max(-1).values.numpy().mean()),
        "conf_ood": float(mp_ood.max(-1).values.numpy().mean()),
        "accept_rate": float(mask.mean()),
        "acc_accepted": float(correct[mask].mean()) if mask.sum() > 0 else 0.0,
        "cov_curve": cov,
        "acc_curve": acc_cov,
    }


# ── Main sweep ─────────────────────────────────────────────────────────────────

all_results = []

total = len(TRAIN_SIZES) * len(SEEDS) * 4
done = 0

for train_size in TRAIN_SIZES:
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, id_loader, ood_loader = make_loaders(train_size, seed)

        for model_type in ["dnn", "bnn", "mc_dropout", "deep_ensemble"]:
            done += 1
            ckpt = Path(f"checkpoints/{model_type}_n{train_size}_s{seed}.json")

            if ckpt.exists():
                print(
                    f"[{done}/{total}] SKIP (cached): {model_type} n={train_size} seed={seed}"
                )
                with open(ckpt) as f:
                    all_results.append(json.load(f))
                continue

            print(
                f"[{done}/{total}] Training {model_type} n={train_size} seed={seed} ...",
                end=" ",
                flush=True,
            )

            if model_type == "dnn":
                model = DNN().to(DEVICE)
                train_dnn(model, train_loader)
                mp_id, _, lab_id = model.predict(id_loader)
                mp_ood, _, _ = model.predict(ood_loader)

            elif model_type == "bnn":
                model = BNN().to(DEVICE)
                train_bnn(model, train_loader)
                mp_id, _, lab_id = model.predict(id_loader)
                mp_ood, _, _ = model.predict(ood_loader)

            elif model_type == "mc_dropout":
                model = MCDropout().to(DEVICE)
                train_dnn(model, train_loader)  # trains like DNN
                mp_id, _, lab_id = model.predict(id_loader)
                mp_ood, _, _ = model.predict(ood_loader)

            elif model_type == "deep_ensemble":
                model = DeepEnsemble().to(DEVICE)
                train_ensemble(model, train_loader, base_seed=seed)
                mp_id, _, lab_id = model.predict(id_loader)
                mp_ood, _, _ = model.predict(ood_loader)

            metrics = compute_metrics(mp_id, lab_id, mp_ood)
            result = dict(model=model_type, train_size=train_size, seed=seed, **metrics)

            with open(ckpt, "w") as f:
                json.dump(result, f)
            all_results.append(result)

            print(
                f"acc={metrics['accuracy']:.3f}  auroc={metrics['auroc']:.3f}  "
                f"ent_id={metrics['entropy_id']:.3f}  ent_ood={metrics['entropy_ood']:.3f}"
            )

# ── Save combined ──────────────────────────────────────────────────────────────
with open("results/baseline_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nDone. {len(all_results)} results saved to results/baseline_results.json")
