"""
Step 2: The key experiment — adversarial and structured OOD.
Run AFTER step1_train_baselines.py.

This is the experiment that determines whether you have a NeurIPS paper.
We test BNN entropy on a spectrum from easy OOD to hard structured OOD:

  Fashion-MNIST     → easy OOD (different semantics, same format)
  Rotated MNIST     → medium OOD (same digits, different orientation)
  FGSM adversarial  → hard OOD (structured perturbation exploiting the function)

The hypothesis:
  BNNs should be uncertain on Fashion-MNIST (we already know this works).
  BNNs may FAIL on adversarials — confident and wrong despite Bayesian weights.
  This is the finding. If confirmed, it's the paper.

What to look for in the output:
  - entropy_fashion >> entropy_id     → BNN works on easy OOD ✓
  - entropy_rotated  > entropy_id     → BNN partially works on medium OOD
  - entropy_adv     << entropy_fashion → BNN FAILS on adversarial ← THE FINDING

Runtime: ~15 min on GPU.
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
FGSM_EPSILONS = [0.05, 0.1, 0.2, 0.3]  # perturbation strengths
ROTATION_ANGLES = [15, 30, 45, 60, 90]  # degrees

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
)

MNIST_TEST = datasets.MNIST("./data", train=False, download=True, transform=transform)
FASHION_TEST = datasets.FashionMNIST(
    "./data",
    train=False,
    download=True,
    transform=transform,
)

id_loader = DataLoader(MNIST_TEST, batch_size=512, shuffle=False, num_workers=0)
fashion_loader = DataLoader(FASHION_TEST, batch_size=512, shuffle=False, num_workers=0)


# ── Re-define models (same as step 1) ─────────────────────────────────────────


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

    def predict_batch(self, xb, n_samples=N_MC_SAMPLES):
        """Predict on a single batch tensor, return (mean, std)."""
        self.eval()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(F.softmax(self(xb), -1).cpu())
        s = torch.stack(samples)  # (S, B, 10)
        return s.mean(0), s.std(0)

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

    @torch.no_grad()
    def predict_loader(self, loader):
        self.eval()
        probs, labels = [], []
        for xb, yb in loader:
            probs.append(F.softmax(self(xb.to(DEVICE)), -1).cpu())
            labels.append(yb)
        mp = torch.cat(probs)
        return mp, torch.cat(labels)


# ── FGSM adversarial attack ────────────────────────────────────────────────────


def fgsm_attack(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (Goodfellow et al. 2014).
    Generates adversarial examples against the DNN.
    We attack the DNN (not the BNN) — this is deliberate:
    we want to test if BNN uncertainty catches inputs that fool a DNN.
    """
    xb = xb.clone().requires_grad_(True)
    loss = F.cross_entropy(model(xb), yb)
    model.zero_grad()
    loss.backward()
    return (xb + epsilon * xb.grad.sign()).detach().clamp(-3, 3)


def generate_adversarials(dnn, loader, epsilon, device):
    """Generate FGSM adversarials for full loader. Returns TensorDataset."""
    dnn.eval()
    adv_x, adv_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        adv_x.append(fgsm_attack(dnn, xb, yb, epsilon).cpu())
        adv_y.append(yb.cpu())
    return TensorDataset(torch.cat(adv_x), torch.cat(adv_y))


# ── Rotated MNIST ──────────────────────────────────────────────────────────────


def rotated_mnist_loader(angle_deg: float) -> DataLoader:
    """MNIST test set with all images rotated by angle_deg degrees."""
    rot_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation([angle_deg, angle_deg]),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    ds = datasets.MNIST("./data", train=False, download=True, transform=rot_transform)
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)


# ── Metrics ────────────────────────────────────────────────────────────────────


def predictive_entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(-1)


def summarize_entropy(entropy_arr, label):
    print(
        f"  {label:<35} "
        f"mean={entropy_arr.mean():.4f}  "
        f"std={entropy_arr.std():.4f}  "
        f"median={np.median(entropy_arr):.4f}",
    )
    return {
        "mean": float(entropy_arr.mean()),
        "std": float(entropy_arr.std()),
        "median": float(np.median(entropy_arr)),
        "pct_high": float((entropy_arr > 0.5).mean()),  # fraction "uncertain"
    }


# ── Load trained models (train_size=2000, seed=0 — representative) ─────────────


def retrain_models(train_size=2000, seed=0):
    """Retrain BNN and DNN for the adversarial experiments."""
    from step1_train_baselines import DNN as DNN1
    from step1_train_baselines import make_loaders, train_dnn

    torch.manual_seed(seed)
    np.random.seed(seed)
    tl, _, _ = make_loaders(train_size, seed)

    print(f"  Training DNN (n={train_size}, seed={seed})...")
    dnn = DNN1().to(DEVICE)
    train_dnn(dnn, tl)

    print(f"  Training BNN (n={train_size}, seed={seed})...")
    bnn = BNN().to(DEVICE)

    # Inline BNN training (with fixes) to avoid import issues
    opt = torch.optim.Adam(bnn.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
    n_b = len(tl)
    bnn.train()
    for ep in range(40):
        kl_w = min(1.0, (ep + 1) / 10) / n_b
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            (F.cross_entropy(bnn(xb), yb) + kl_w * bnn.kl_total()).backward()
            opt.step()
        sched.step()

    return dnn, bnn


# ── Main experiment ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("KEY EXPERIMENT: BNN uncertainty on structured OOD")
    print("=" * 60)

    # Train fresh models for this experiment
    print("\nTraining models for adversarial experiments...")
    dnn = DNN().to(DEVICE)
    bnn = BNN().to(DEVICE)

    # Load from step1 checkpoint if available, else retrain
    ckpt_dnn = Path("checkpoints/dnn_n2000_s0.json")
    if not ckpt_dnn.exists():
        print("  Checkpoints not found — run step1 first, or retraining now...")

    # Always retrain for adversarial since we need model objects not just metrics
    torch.manual_seed(0)
    np.random.seed(0)

    # Quick inline data setup
    idx = np.random.choice(len(MNIST_TEST.data), 2000, replace=False)
    # Use full MNIST_TRAIN subset for training
    mnist_train_full = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )
    labels_all = np.array(mnist_train_full.targets)
    classes = np.unique(labels_all)
    n_per = 200  # 2000/10
    train_idx = []
    rng = np.random.RandomState(0)
    for c in classes:
        ci = np.where(labels_all == c)[0]
        train_idx.extend(rng.choice(ci, n_per, replace=False).tolist())
    train_loader = DataLoader(
        Subset(mnist_train_full, train_idx),
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    print("  Training DNN...")
    opt = torch.optim.Adam(dnn.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
    dnn.train()
    for ep in range(40):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(dnn(xb), yb).backward()
            opt.step()
        sched.step()

    print("  Training BNN...")
    lr = 1e-4
    epochs = 120

    opt2 = torch.optim.Adam(bnn.parameters(), lr=lr)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, epochs)
    n_b = len(train_loader)
    bnn.train()
    for ep in range(epochs):
        # kl_w = min(1.0, (ep + 1) / 10) / n_b
        kl_w = min(1.0, (ep + 1) / 20) / n_b
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt2.zero_grad()
            (F.cross_entropy(bnn(xb), yb) + kl_w * bnn.kl_total()).backward()
            opt2.step()
        sched2.step()

    # Quick accuracy check
    with torch.no_grad():
        dnn.eval()
        correct = 0
        total = 0
        for xb, yb in id_loader:
            preds = dnn(xb.to(DEVICE)).argmax(-1).cpu()
            correct += (preds == yb).sum().item()
            total += len(yb)
    print(f"\n  DNN test accuracy: {correct / total:.4f}")

    mp_id, _, lab_id = bnn.predict_loader(id_loader)
    bnn_acc = (mp_id.argmax(-1) == lab_id).float().mean().item()
    print(f"  BNN test accuracy: {bnn_acc:.4f}")

    results = {}

    # ── 1. Baseline: MNIST (ID) ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("ENTROPY SUMMARY (higher = more uncertain = better for OOD)")
    print("─" * 60)

    mp_id, _, lab_id = bnn.predict_loader(id_loader)
    ent_id = predictive_entropy(mp_id).numpy()
    results["mnist_id"] = summarize_entropy(ent_id, "MNIST (ID, baseline)")

    # ── 2. Fashion-MNIST (easy OOD) ────────────────────────────────────────────
    mp_f, _, _ = bnn.predict_loader(fashion_loader)
    ent_f = predictive_entropy(mp_f).numpy()
    results["fashion_ood"] = summarize_entropy(ent_f, "Fashion-MNIST (easy OOD)")

    # ── 3. Rotated MNIST (medium OOD spectrum) ─────────────────────────────────
    print()
    for angle in ROTATION_ANGLES:
        rot_loader = rotated_mnist_loader(float(angle))
        mp_r, _, _ = bnn.predict_loader(rot_loader)
        ent_r = predictive_entropy(mp_r).numpy()
        results[f"rotated_{angle}"] = summarize_entropy(
            ent_r,
            f"Rotated MNIST {angle}° (medium OOD)",
        )

    # ── 4. FGSM adversarials — THE KEY EXPERIMENT ──────────────────────────────
    print()
    for eps in FGSM_EPSILONS:
        adv_ds = generate_adversarials(dnn, id_loader, eps, DEVICE)
        adv_loader = DataLoader(adv_ds, batch_size=512, shuffle=False, num_workers=0)

        # DNN accuracy on adversarials (confirms attack works)
        with torch.no_grad():
            dnn.eval()
            adv_correct = 0
            adv_total = 0
            for xb, yb in adv_loader:
                preds = dnn(xb.to(DEVICE)).argmax(-1).cpu()
                adv_correct += (preds == yb).sum().item()
                adv_total += len(yb)
        dnn_adv_acc = adv_correct / adv_total

        # BNN entropy on adversarials
        mp_adv, std_adv, _ = bnn.predict_loader(adv_loader)
        ent_adv = predictive_entropy(mp_adv).numpy()
        conf_adv = mp_adv.max(-1).values.numpy()

        r = summarize_entropy(ent_adv, f"FGSM ε={eps} (DNN acc={dnn_adv_acc:.3f})")
        r["dnn_accuracy_after_attack"] = dnn_adv_acc
        r["bnn_mean_confidence"] = float(conf_adv.mean())
        r["bnn_pct_confident_wrong"] = float(
            ((conf_adv > 0.8) & (mp_adv.argmax(-1).numpy() != lab_id.numpy())).mean(),
        )
        results[f"fgsm_eps{eps}"] = r

    # ── 5. Print the critical comparison ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("CRITICAL COMPARISON")
    print("=" * 60)
    print(f"  Baseline ID entropy:          {results['mnist_id']['mean']:.4f}")
    print(f"  Fashion OOD entropy:          {results['fashion_ood']['mean']:.4f}")
    print(
        f"  Rotated 90° entropy:          {results.get('rotated_90', {}).get('mean', 'N/A')}",
    )

    for eps in FGSM_EPSILONS:
        k = f"fgsm_eps{eps}"
        if k in results:
            dnn_acc = results[k]["dnn_accuracy_after_attack"]
            bnn_ent = results[k]["mean"]
            pct_conf_wrong = results[k]["bnn_pct_confident_wrong"]
            print(
                f"  FGSM ε={eps}: DNN acc={dnn_acc:.3f}  "
                f"BNN entropy={bnn_ent:.4f}  "
                f"BNN conf+wrong={pct_conf_wrong:.3f}",
            )

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
  If BNN entropy on FGSM ≈ entropy on ID:
    → BNN FAILS to detect adversarial OOD
    → Mean-field posterior doesn't capture function-space uncertainty
    → THIS IS YOUR FINDING. Paper title:
      "Mean-Field BNNs Are Not Uncertainty-Aware Under Adversarial Perturbation"

  If BNN entropy on FGSM >> entropy on ID:
    → BNN succeeds even on adversarials (surprising positive result)
    → Paper angle changes: BNNs are more robust than expected
    → Still publishable, different framing

  If BNN entropy on FGSM is between ID and Fashion:
    → Partial failure — BNN less uncertain on structured vs random OOD
    → Paper: "Structured OOD inputs expose limitations of mean-field BNNs"
    """)

    # ── Save results ───────────────────────────────────────────────────────────
    with open("results/adversarial_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/adversarial_results.json")
    print("\nPaste the CRITICAL COMPARISON output here when you're done.")
