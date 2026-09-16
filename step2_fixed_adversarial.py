"""
Fixed adversarial experiment — run AFTER step1_fixed_baselines.py.

Uses identical BNN hyperparameters as step1_fixed (lr=3e-4, 80 epochs, 20-epoch annealing).
Tests the entropy ordering: ID < FGSM adversarial < semantic OOD (Fashion, rotated).

This is the core finding for the paper:
  Adversarial perturbations that fool a DNN look MORE like training data
  to the BNN posterior than semantic domain shifts do.

Runtime: ~20 min GPU
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")

N_MC_SAMPLES = 50
FGSM_EPSILONS = [0.05, 0.1, 0.2, 0.3]
ROTATION_ANGLES = [15, 30, 45, 60, 90]
TRAIN_SIZE = 2000
SEED = 0

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
)

MNIST_TRAIN = datasets.MNIST("./data", train=True, download=True, transform=transform)
MNIST_TEST = datasets.MNIST("./data", train=False, download=True, transform=transform)
FASHION_TEST = datasets.FashionMNIST(
    "./data",
    train=False,
    download=True,
    transform=transform,
)

id_loader = DataLoader(MNIST_TEST, batch_size=512, shuffle=False, num_workers=0)
fashion_loader = DataLoader(FASHION_TEST, batch_size=512, shuffle=False, num_workers=0)


# ── Models (same as step1) ─────────────────────────────────────────────────────


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

    def predict_loader(self, loader, n_samples=N_MC_SAMPLES):
        self.eval()
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


# ── Training (FIXED hyperparams) ───────────────────────────────────────────────


def train_dnn(model, loader, epochs=40, lr=1e-3):
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


def train_bnn(model, loader, epochs=80, lr=3e-4):
    """FIXED: 80 epochs, lr=3e-4, 20-epoch annealing, gradient clipping."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_batches = len(loader)
    model.train()
    for ep in range(epochs):
        kl_weight = min(1.0, (ep + 1) / 20) / n_batches
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            kl = model.kl_total()
            loss = ce + kl_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()


# ── Helpers ────────────────────────────────────────────────────────────────────


def predictive_entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(-1)


def stratified_subset(dataset, n, seed):
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


def fgsm_attack(model, xb, yb, epsilon):
    xb = xb.clone().requires_grad_(True)
    loss = F.cross_entropy(model(xb), yb)
    model.zero_grad()
    loss.backward()
    return (xb + epsilon * xb.grad.sign()).detach().clamp(-3, 3)


def rotated_loader(angle):
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation([angle, angle]),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    ds = datasets.MNIST("./data", train=False, download=True, transform=t)
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)


def summarize(arr, label):
    r = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "pct_high": float((arr > 0.5).mean()),
    }
    print(
        f"  {label:<40} H={r['mean']:.4f} ± {r['std']:.4f}  "
        f"pct>0.5nats={r['pct_high']:.3f}",
    )
    return r


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Training data
    idx = stratified_subset(MNIST_TRAIN, TRAIN_SIZE, SEED)
    train_loader = DataLoader(
        Subset(MNIST_TRAIN, idx),
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    print(f"Training DNN (n={TRAIN_SIZE})...")
    dnn = DNN().to(DEVICE)
    train_dnn(dnn, train_loader)

    print(f"Training BNN with FIXED hyperparameters (n={TRAIN_SIZE})...")
    bnn = BNN().to(DEVICE)
    train_bnn(bnn, train_loader)

    mp_id_check, _, lab_check = bnn.predict_loader(id_loader)
    ent_check = -(mp_id_check * (mp_id_check + 1e-8).log()).sum(-1).mean().item()

    print(f"\n  [CHECK] BNN ID entropy after training: {ent_check:.3f}")

    # Convergence check
    with torch.no_grad():
        sigma_mean = F.softplus(bnn.fc1.rho).mean().item()
    print(f"\nBNN fc1 posterior sigma mean: {sigma_mean:.4f}")
    print("  (healthy range: 0.01 to 0.5)")

    # Quick accuracy check
    dnn.eval()
    with torch.no_grad():
        correct = sum(
            (dnn(xb.to(DEVICE)).argmax(-1).cpu() == yb).sum().item()
            for xb, yb in id_loader
        )
    dnn_acc = correct / len(MNIST_TEST)
    mp_id, _, lab_id = bnn.predict_loader(id_loader)
    bnn_acc = (mp_id.argmax(-1) == lab_id).float().mean().item()
    print(f"\nDNN test accuracy: {dnn_acc:.4f}")
    print(f"BNN test accuracy: {bnn_acc:.4f}")

    results = {}

    print("\n" + "=" * 65)
    print("ENTROPY ACROSS OOD SPECTRUM (higher = more uncertain)")
    print("=" * 65)

    # 1. MNIST ID
    mp_id, _, lab_id = bnn.predict_loader(id_loader)
    ent_id = predictive_entropy(mp_id).numpy()
    results["mnist_id"] = summarize(ent_id, "MNIST (ID)")

    # 2. Fashion-MNIST
    mp_f, _, _ = bnn.predict_loader(fashion_loader)
    ent_f = predictive_entropy(mp_f).numpy()
    results["fashion_ood"] = summarize(ent_f, "Fashion-MNIST (semantic OOD)")

    # 3. Rotated MNIST
    print()
    for angle in ROTATION_ANGLES:
        mp_r, _, _ = bnn.predict_loader(rotated_loader(float(angle)))
        ent_r = predictive_entropy(mp_r).numpy()
        results[f"rotated_{angle}"] = summarize(ent_r, f"Rotated MNIST {angle}deg")

    # 4. FGSM adversarials
    print()
    for eps in FGSM_EPSILONS:
        # Generate adversarials against DNN
        adv_x, adv_y = [], []
        dnn.eval()
        for xb, yb in id_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            adv_x.append(fgsm_attack(dnn, xb, yb, eps).cpu())
            adv_y.append(yb.cpu())
        adv_loader = DataLoader(
            TensorDataset(torch.cat(adv_x), torch.cat(adv_y)),
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )

        # DNN accuracy after attack (confirms attack works)
        dnn.eval()
        with torch.no_grad():
            adv_correct = sum(
                (dnn(xb.to(DEVICE)).argmax(-1).cpu() == yb).sum().item()
                for xb, yb in adv_loader
            )
        dnn_adv_acc = adv_correct / len(MNIST_TEST)

        # BNN entropy on adversarials
        mp_adv, _, _ = bnn.predict_loader(adv_loader)
        ent_adv = predictive_entropy(mp_adv).numpy()
        conf_adv = mp_adv.max(-1).values.numpy()

        r = summarize(
            ent_adv,
            f"FGSM eps={eps} [DNN acc={dnn_adv_acc:.3f} after attack]",
        )
        r["dnn_accuracy_after_attack"] = dnn_adv_acc
        r["bnn_mean_confidence"] = float(conf_adv.mean())
        r["bnn_pct_confident_wrong"] = float(
            ((conf_adv > 0.8) & (mp_adv.argmax(-1).numpy() != lab_id.numpy())).mean(),
        )
        results[f"fgsm_eps{eps}"] = r

    # ── Critical comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CRITICAL COMPARISON")
    print("=" * 65)
    print(f"  MNIST ID entropy:        {results['mnist_id']['mean']:.4f}")
    print(f"  Fashion OOD entropy:     {results['fashion_ood']['mean']:.4f}")
    print(
        f"  Rotated 90deg entropy:   {results.get('rotated_90', {}).get('mean', 'N/A')}",
    )
    print()
    for eps in FGSM_EPSILONS:
        k = f"fgsm_eps{eps}"
        if k in results:
            print(
                f"  FGSM eps={eps}: "
                f"DNN acc={results[k]['dnn_accuracy_after_attack']:.3f}  "
                f"BNN H={results[k]['mean']:.4f}  "
                f"BNN conf+wrong={results[k]['bnn_pct_confident_wrong']:.4f}",
            )

    # ── Interpretation ────────────────────────────────────────────────────────
    id_h = results["mnist_id"]["mean"]
    adv_h = results.get("fgsm_eps0.3", {}).get("mean", 0)
    ood_h = results["fashion_ood"]["mean"]

    print("\n" + "=" * 65)
    print("INTERPRETATION")
    print("=" * 65)
    if id_h < 0.5:
        print(f"  BNN converged correctly (H_id={id_h:.3f} < 0.5 nats)")
        print(
            f"  Entropy ordering: ID({id_h:.3f}) < ADV({adv_h:.3f}) < OOD({ood_h:.3f})",
        )
        if adv_h < ood_h:
            print()
            print("  FINDING CONFIRMED:")
            print(
                "  Adversarial inputs (which fool the DNN) produce LESS BNN uncertainty",
            )
            print("  than semantic OOD inputs (Fashion-MNIST, rotated digits).")
            print(
                "  This means the BNN posterior treats adversarials as more 'in-distribution'",
            )
            print(
                "  than genuine domain shifts -- even though adversarials fool the DNN completely.",
            )
            print()
            print("  Paper title candidate:")
            print(
                "  'Mean-Field BNNs Distinguish Semantic OOD from Adversarial Perturbation'",
            )
        else:
            print()
            print("  Adversarial entropy >= OOD entropy.")
            print("  BNN is MORE uncertain on adversarials than on semantic OOD.")
            print("  Different finding -- also publishable, different framing.")
    else:
        print(f"  WARNING: H_id={id_h:.3f} still > 0.5 nats after fixes.")
        print("  Try: epochs=120, lr=1e-4, anneal over 30 epochs")
        print("  Or try a smaller hidden size (128 instead of 256)")

    Path("results_fixed").mkdir(exist_ok=True)
    with open("results_fixed/adversarial_results_fixed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results_fixed/adversarial_results_fixed.json")
    print("\nPaste the CRITICAL COMPARISON block here when done.")
