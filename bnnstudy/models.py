"""Models and their training loops. Every model exposes `predict(x)` returning
per-sample class probabilities of shape (S, n, 10); S = 1 for deterministic nets."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import batches

HIDDEN = 256


def mlp(p_drop=0.0):
    layers = [nn.Flatten()]
    for i, o in [(784, HIDDEN), (HIDDEN, HIDDEN)]:
        layers += [nn.Linear(i, o), nn.ReLU()]
        if p_drop:
            layers.append(nn.Dropout(p_drop))
    layers.append(nn.Linear(HIDDEN, 10))
    return nn.Sequential(*layers)


# ── Deterministic / MC dropout ──────────────────────────────────────────────────


class DNN(nn.Module):
    def __init__(self, p_drop=0.0, mc_samples=1):
        super().__init__()
        self.net = mlp(p_drop)
        self.mc_samples = mc_samples

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def logits(self, x, chunk=2048):
        self.eval()
        return torch.cat([self(x[i : i + chunk]) for i in range(0, len(x), chunk)])

    @torch.no_grad()
    def predict(self, x, chunk=2048):
        if self.mc_samples == 1:
            return F.softmax(self.logits(x, chunk), -1)[None]
        self.train()  # dropout on
        return torch.stack([
            torch.cat([F.softmax(self(x[i : i + chunk]), -1) for i in range(0, len(x), chunk)])
            for _ in range(self.mc_samples)
        ])


def train_dnn(model, x, y, epochs, lr, batch_size, gen, prior_sigma=None, clip=None):
    """prior_sigma: MAP under a N(0, prior_sigma^2) prior on all weights and biases,
    i.e. CE_mean + ||w||^2 / (2 prior_sigma^2 N), the per-example scaling BBB uses."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(x, y, batch_size, gen):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if prior_sigma is not None:
                sq = sum((p ** 2).sum() for p in model.parameters())
                loss = loss + sq / (2 * prior_sigma ** 2 * len(x))
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        sched.step()


class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    @torch.no_grad()
    def predict(self, x):
        return torch.cat([m.predict(x) for m in self.members])


# ── Last-layer Laplace (full GGN covariance over the final linear layer) ─────────


class LastLayerLaplace(nn.Module):
    """Gaussian posterior N(W_map, (H + lambda I)^-1) over the last layer of a
    trained DNN, with H the generalised Gauss-Newton of the training CE loss.
    lambda (prior precision) is chosen on validation NLL."""

    def __init__(self, dnn, mc_samples=50):
        super().__init__()
        self.body = dnn.net[:-1]
        last = dnn.net[-1]
        self.w_map = torch.cat([last.weight, last.bias[:, None]], 1).detach()  # (10, 257)
        self.mc_samples = mc_samples

    @torch.no_grad()
    def features(self, x, chunk=2048):
        self.body.eval()
        f = torch.cat([self.body(x[i : i + chunk]) for i in range(0, len(x), chunk)])
        return torch.cat([f, torch.ones(len(f), 1, device=f.device)], 1)

    @torch.no_grad()
    def fit(self, x_train, x_val, y_val, prior_precisions=np.logspace(-2, 4, 25)):
        phi = self.features(x_train).double()
        p = F.softmax(phi @ self.w_map.double().T, -1)
        k, d = self.w_map.shape
        # H[(c,a),(e,b)] = sum_i (diag(p_i) - p_i p_i^T)[c,e] phi_i[a] phi_i[b]
        H = torch.zeros(k, d, k, d, dtype=torch.float64, device=phi.device)
        for c in range(k):
            for e in range(k):
                lam = (p[:, c] * (float(c == e) - p[:, e]))[:, None]
                H[c, :, e, :] = (phi * lam).T @ phi
        self.H = H.reshape(k * d, k * d)
        phi_val = self.features(x_val)
        best = None
        for pp in prior_precisions:
            self.set_prior_precision(pp)
            g = torch.Generator(device=phi_val.device).manual_seed(0)
            probs = self._predict_features(phi_val, g).mean(0)
            nll = -torch.log(probs[torch.arange(len(y_val)), y_val] + 1e-12).mean().item()
            if best is None or nll < best[0]:
                best = (nll, pp)
        self.prior_precision = best[1]
        self.set_prior_precision(best[1])
        return self

    def set_prior_precision(self, pp):
        prec = self.H + pp * torch.eye(len(self.H), dtype=self.H.dtype, device=self.H.device)
        self.scale_tril = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(prec))).float()

    def _predict_features(self, phi, generator=None):
        k, d = self.w_map.shape
        eps = torch.randn(self.mc_samples, k * d, device=phi.device, generator=generator)
        W = self.w_map + (eps @ self.scale_tril.T).view(-1, k, d)  # (S, 10, 257)
        return F.softmax(torch.einsum("nd,skd->snk", phi, W), -1)

    @torch.no_grad()
    def predict(self, x):
        return self._predict_features(self.features(x))


# ── Bayes by Backprop (mean-field Gaussian, N(0, prior_sigma^2) prior) ───────────


class BayesianLinear(nn.Module):
    def __init__(self, i, o, prior_sigma=1.0, rho_init=-5.0, legacy_init=False):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(o, i))
        if legacy_init:  # small-init variant: mu ~ N(0, 0.1^2), rho = -3
            nn.init.normal_(self.mu, std=0.1)
            rho_init = -3.0
        else:
            nn.init.kaiming_uniform_(self.mu, a=5 ** 0.5)  # same init as nn.Linear
        self.rho = nn.Parameter(torch.full((o, i), rho_init))
        self.bias_mu = nn.Parameter(torch.zeros(o))
        self.bias_rho = nn.Parameter(torch.full((o,), rho_init))
        self.prior_sigma = prior_sigma

    def forward(self, x):
        w = self.mu + F.softplus(self.rho) * torch.randn_like(self.mu)
        b = self.bias_mu + F.softplus(self.bias_rho) * torch.randn_like(self.bias_mu)
        return F.linear(x, w, b)

    def kl(self):
        ps2 = self.prior_sigma ** 2
        total = 0.0
        for mu, rho in [(self.mu, self.rho), (self.bias_mu, self.bias_rho)]:
            s2 = F.softplus(rho) ** 2
            total = total + 0.5 * ((s2 + mu ** 2) / ps2 - 1 - torch.log(s2 / ps2)).sum()
        return total


class BBB(nn.Module):
    def __init__(self, mc_samples=50, **layer_kw):
        super().__init__()
        self.fc1 = BayesianLinear(784, HIDDEN, **layer_kw)
        self.fc2 = BayesianLinear(HIDDEN, HIDDEN, **layer_kw)
        self.fc3 = BayesianLinear(HIDDEN, 10, **layer_kw)
        self.mc_samples = mc_samples

    def forward(self, x):
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def kl(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()

    @torch.no_grad()
    def predict(self, x, chunk=2048):
        return torch.stack([
            torch.cat([F.softmax(self(x[i : i + chunk]), -1) for i in range(0, len(x), chunk)])
            for _ in range(self.mc_samples)
        ])


def train_bbb(model, x, y, epochs, lr, batch_size, gen, kl_scale, anneal_epochs):
    """Minimise CE_mean + anneal * kl_scale * KL.

    kl_scale = 1/N is the per-example negative ELBO. Using 1/n_batches instead
    (an easy slip when CE is averaged per batch) makes the KL weight about
    batch_size times too large; kept as the "bbb_bug" ablation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        w = min(1.0, (ep + 1) / anneal_epochs) * kl_scale
        model.train()
        for xb, yb in batches(x, y, batch_size, gen):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb) + w * model.kl()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
