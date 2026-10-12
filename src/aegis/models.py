"""Model definition, training and evaluation.

``SmallCNN`` is a compact convolutional classifier for 8x8 digits that trains in
seconds on CPU. It exposes ``features(x)`` returning the penultimate activations
used by the spectral-signatures backdoor detector.

``train_model`` supports optional Madry-style PGD adversarial training that mixes
clean and adversarial examples inside each batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .config import Config
from .data import make_loader


class SmallCNN(nn.Module):
    """conv(1->16)-relu-conv(16->32)-relu-maxpool2-flatten-fc(512->64)-relu-fc(64->10)."""

    def __init__(
        self, conv1_out: int = 16, conv2_out: int = 32, fc_hidden: int = 64, num_classes: int = 10
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # 8x8 -> conv (same) -> conv (same) -> pool2 -> 4x4
        self.flat_dim = conv2_out * 4 * 4
        self.fc1 = nn.Linear(self.flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate (fc_hidden-d) activations. Used by spectral signatures."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


@dataclass
class TrainResult:
    model: SmallCNN
    final_train_loss: float
    epochs: int


def build_model(cfg: Config, num_classes: int = 10) -> SmallCNN:
    return SmallCNN(
        conv1_out=cfg.model.conv1_out,
        conv2_out=cfg.model.conv2_out,
        fc_hidden=cfg.model.fc_hidden,
        num_classes=num_classes,
    )


def _pgd_perturb(
    model: SmallCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool,
) -> torch.Tensor:
    """Inner PGD maximisation for adversarial training. Returns adversarial x."""
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
    if was_training:
        model.train()
    return x_adv


def train_model(
    cfg: Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = 10,
    adversarial: bool = False,
    model: SmallCNN | None = None,
) -> TrainResult:
    """Train a SmallCNN. With ``adversarial=True`` a fraction of each batch is
    replaced by PGD adversarial examples (Madry et al., arXiv:1706.06083).
    """
    if model is None:
        model = build_model(cfg, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    loader = make_loader(x_train, y_train, cfg.model.batch_size, shuffle=True, seed=int(cfg.seed))
    adv = cfg.adv_training
    last_loss = 0.0
    model.train()
    for _epoch in range(int(cfg.model.epochs)):
        running = 0.0
        n_batches = 0
        for xb, yb in loader:
            if adversarial:
                k = max(1, int(adv.adv_weight * xb.size(0)))
                x_adv = _pgd_perturb(
                    model,
                    xb[:k],
                    yb[:k],
                    adv.eps,
                    adv.step_size,
                    adv.steps,
                    adv.random_start,
                )
                xb = torch.cat([x_adv, xb[k:]], dim=0)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        last_loss = running / max(1, n_batches)
    model.eval()
    return TrainResult(model=model, final_train_loss=last_loss, epochs=int(cfg.model.epochs))


def train_on_arrays(
    cfg: Config,
    model: SmallCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int | None = None,
) -> SmallCNN:
    """Plain (non-adversarial) training loop used by cross-validation. Mutates
    and returns ``model``.
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    loader = make_loader(x, y, cfg.model.batch_size, shuffle=True, seed=int(cfg.seed))
    model.train()
    for _ in range(epochs if epochs is not None else int(cfg.model.epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def features_matrix(model: SmallCNN, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Penultimate activations for a batch as a NumPy array (N, fc_hidden)."""
    model.eval()
    out = []
    for i in range(0, x.size(0), batch_size):
        out.append(model.features(x[i : i + batch_size]).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict(model: SmallCNN, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, x.size(0), batch_size):
        out.append(model(x[i : i + batch_size]).argmax(dim=1))
    return torch.cat(out)


@torch.no_grad()
def predict_proba(model: SmallCNN, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, x.size(0), batch_size):
        out.append(F.softmax(model(x[i : i + batch_size]), dim=1))
    return torch.cat(out)


@torch.no_grad()
def evaluate(model: SmallCNN, x: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = predict(model, x)
    return float((preds == y).float().mean().item())
